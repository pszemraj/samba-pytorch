import gzip
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from samba_pytorch import GPT, Config
from samba_pytorch.utils import get_default_supported_precision

# Constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3
PRINT_EVERY = 10
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

# Get optimal precision for training
PRECISION = get_default_supported_precision(training=True)
USE_AMP = PRECISION in ["16-mixed", "bf16-mixed"]
DTYPE = (
    torch.bfloat16
    if "bf16" in PRECISION
    else torch.float16
    if "16" in PRECISION
    else torch.float32
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_here = Path(__file__).resolve().parent


# helpers
def exists(v):
    return v is not None


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# sampling helpers
def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1, keepdim=True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(
        dim=dim, keepdim=keepdim
    )


def min_p_filter(logits, min_p=0.1):
    probs = logits.softmax(dim=-1)
    max_probs = probs.amax(dim=-1, keepdim=True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float("-inf"), logits)


def base_decoding(
    net,
    prompt: torch.Tensor,
    seq_len: int,
    temperature=1.3,
    min_p=1e-1,
):
    B = prompt.size(0)  # Batch size
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)[:, -1]  # Get last token logits [B, vocab_size]
        logits = min_p_filter(logits, min_p=min_p)

        # Sample and reshape to [B, 1]
        sample = gumbel_sample(logits, temperature=temperature, dim=-1).view(B, 1)
        out = torch.cat([out, sample], dim=1)

    return out[:, prompt_seq_len:]


# Dataset preparation
with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8)

train_data, val_data = np.split(data, [int(90e6)])
train_data = np.copy(train_data)
val_data = np.copy(val_data)
train_data, val_data = torch.from_numpy(train_data), torch.from_numpy(val_data)


class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = random.randint(0, len(self.data) - self.seq_len - 1)
        seq = self.data[start : start + self.seq_len + 1].long()
        return seq.to(device)


train_dataset = TextDataset(train_data, SEQ_LEN)
val_dataset = TextDataset(val_data, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# Model initialization
config = Config(
    name="CustomSamba",
    block_size=SEQ_LEN,  # match sequence length
    vocab_size=256,  # byte-level tokenization
    n_layer=8,
    n_head=8,
    n_embd=512,
    padding_multiple=64,
    mb_per_layer=2,
    fused_add_norm=False,
    parallel_residual=False,
    norm_eps=1e-5,
    rotary_percentage=1.0,
    _mlp_class="LLaMAMLP",
    _norm_class="RMSNorm",
    intermediate_size=1536,
    mamba_init=True,
)

model = GPT(config).to(device)

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, fused=torch.cuda.is_available())
criterion = nn.CrossEntropyLoss()

out_dir = _here / "out" / "samba"
best_loss = float("inf")
last_checkpoint_step = -1

# Training loop
for batch_num in tqdm.tqdm(range(NUM_BATCHES), mininterval=5.0, desc="Training"):
    model.train()
    optimizer.zero_grad()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)
        inputs = data[:, :-1]
        targets = data[:, 1:]

        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=USE_AMP):
            logits = model(inputs)
            loss = (
                criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                / GRAD_ACCUM_EVERY
            )

            loss.backward()

    if batch_num % PRINT_EVERY == 0:
        print(f"Loss at step {batch_num}: {loss.item():.4f}")

    optimizer.step()
    optimizer.zero_grad()

    if batch_num % VALIDATE_EVERY == 0:
        model.eval()
        with torch.inference_mode():
            val_data = next(val_loader)
            val_inputs = val_data[:, :-1]
            val_targets = val_data[:, 1:]
            val_logits = model(val_inputs)
            val_loss = criterion(
                val_logits.reshape(-1, val_logits.size(-1)), val_targets.reshape(-1)
            )
            print(f"Validation Loss at step {batch_num}: {val_loss.item():.4f}")

            if loss < best_loss and batch_num > 0:
                best_loss = loss
                out_dir.mkdir(exist_ok=True, parents=True)
                print(f"Best loss updated: {best_loss.item():.3f}")
                print(f"Saving model to {str(out_dir)}")

                # Save both model weights and config
                torch.save(model.state_dict(), out_dir / "best.pt")
                with open(out_dir / "config.json", "w") as f:
                    json.dump(config.__dict__, f, indent=2)

                last_checkpoint_step = batch_num

    if batch_num % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"{prime} \n\n {'*' * 100}")

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)
        base_decode_output = decode_tokens(sampled[0])

        print(f"\n\n{base_decode_output}\n")

print("Training complete")
print(f"Last checkpoint step:\t{last_checkpoint_step}")
print(f"Best loss:\t{best_loss.item():.3f}")
