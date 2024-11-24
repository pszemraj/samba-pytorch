import json
from pathlib import Path

import fire
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import trange

from samba_pytorch import GPT, Config
from samba_pytorch.tokenizer import Tokenizer
from samba_pytorch.utils import (
    activate_tf32_if_available,
    get_default_supported_precision,
    model_summary,
)

# Constants
NUM_BATCHES = int(5e4)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3
PRINT_EVERY = 10
VALIDATE_EVERY = 100
PRIME_LENGTH = 32
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
activate_tf32_if_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_here = Path(__file__).resolve().parent


# helpers
def exists(v):
    return v is not None


def cycle(loader):
    while True:
        for data in loader:
            yield data


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
    tokenizer,
    prompt: torch.Tensor,
    seq_len: int,
    temperature=0.7,
    min_p=1e-1,
):
    net.eval()  # Ensure model is in eval mode
    generated_sequence = prompt.clone()

    # Debug info
    print("\nStarting generation:")
    print(f"- Input shape: {prompt.shape}")
    print(f"- Target length: {seq_len}")
    print(f"- Temperature: {temperature}")

    with torch.inference_mode():
        for _ in range(seq_len - prompt.shape[1]):
            # Get predictions
            outputs = net(generated_sequence)
            next_token_logits = outputs[:, -1, :] / temperature

            # Get probabilities and filter
            probs = torch.softmax(next_token_logits, dim=-1)
            if min_p > 0:
                max_probs = probs.max(dim=-1, keepdim=True).values
                probs[probs < min_p * max_probs] = 0
                probs.div_(probs.sum(dim=-1, keepdim=True))

            # Sample from the filtered distribution
            next_token = torch.multinomial(probs, num_samples=1)

            # Stop if we hit the EOS token
            if next_token.item() == tokenizer.eos_id:
                break

            # Concatenate the new token
            generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

    # Get just the generated part
    generated_part = generated_sequence[:, prompt.shape[1] :]

    try:
        # Try to decode the full sequence for better context
        full_text = tokenizer.decode(generated_sequence[0])
        prompt_text = tokenizer.decode(prompt[0])

        # Get just the newly generated text by removing the prompt
        if full_text.startswith(prompt_text):
            generated_text = full_text[len(prompt_text) :]
        else:
            # Fallback to decoding just the new part
            generated_text = tokenizer.decode(generated_part[0])
    except Exception as e:
        print(f"Decoding error: {str(e)}")
        return generated_part, "[Decoding failed]"

    return generated_part, generated_text


class WikiTextDataset(Dataset):
    def __init__(self, dataset, tokenizer, seq_len):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Try getting text from dataset up to 3 times
        for attempt in range(3):
            try:
                text = self.dataset[(idx + attempt) % len(self)]["text"]
                if not text or len(text.strip()) < 10:  # Skip empty or very short texts
                    continue

                # Get tokens - cast to long here to ensure correct dtype
                tokens = self.tokenizer.encode(
                    text.strip(), device=device, bos=True, eos=False
                ).long()  # Force long dtype

                # Handle sequence length
                if len(tokens) >= self.seq_len + 1:
                    # Take slice from start
                    tokens = tokens[: self.seq_len + 1]
                else:
                    # Pad with EOS if needed
                    padding = torch.full(
                        (self.seq_len + 1 - len(tokens),),
                        self.tokenizer.eos_id,
                        dtype=torch.long,
                        device=device,
                    )
                    tokens = torch.cat([tokens, padding])

                return tokens.long()  # Ensure long dtype again

            except Exception as e:
                if attempt == 2:
                    print(f"Error processing idx {idx}: {str(e)}")
                    # Return fallback sequence
                    return torch.tensor(
                        [self.tokenizer.bos_id]
                        + [self.tokenizer.eos_id] * self.seq_len,
                        dtype=torch.long,
                        device=device,
                    )
                continue

        # Ultimate fallback
        return torch.tensor(
            [self.tokenizer.bos_id] + [self.tokenizer.eos_id] * self.seq_len,
            dtype=torch.long,
            device=device,
        )


def main(
    tokenizer_dir,
    dataset_name: str = "pszemraj/simple_wikipedia_LM",
    dataset_config: str = "default",
    ):
    """
    Main function for training a language model using a specified tokenizer and dataset.

    Args:
        tokenizer_dir (str): Directory path where the tokenizer checkpoint is stored.
        dataset_name (str, optional): Name of the dataset to load. Defaults to "pszemraj/simple_wikipedia_LM".
        dataset_config (str, optional): Configuration for the dataset. Defaults to "default".
    """
    tokenizer_dir = Path(tokenizer_dir)
    assert tokenizer_dir.is_dir(), f"{tokenizer_dir} is not a directory"

    # Load dataset
    print(f"Loading dataset... {dataset_name} {dataset_config}")
    ds = load_dataset(dataset_name, dataset_config)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = Tokenizer(tokenizer_dir)

    # Dataset preparation
    train_dataset = WikiTextDataset(ds["train"], tokenizer, SEQ_LEN)
    val_dataset = WikiTextDataset(ds["validation"], tokenizer, SEQ_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    train_loader = cycle(train_loader)
    val_loader = cycle(val_loader)

    # Model initialization
    config = Config(
        name="WikiSamba",
        block_size=SEQ_LEN,
        vocab_size=tokenizer.vocab_size,  # Use tokenizer's vocab size
        padding_multiple=64,
        n_layer=8,
        n_head=8,
        n_embd=512,
        mb_per_layer=2,
        rotary_percentage=1.0,
        fused_add_norm=False,
        parallel_residual=False,
        bias=False,
        _norm_class="RMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1536,
        mamba_init=True,
    )
    print(config)

    model = GPT(config).to(device)
    model_summary(model, max_depth=5)

    # Optimizer and loss function
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, fused=torch.cuda.is_available()
    )
    criterion = nn.CrossEntropyLoss()

    out_dir = _here / "out" / config.name
    best_loss = float("inf")
    last_checkpoint_step = -1

    # Training loop
    for batch_num in trange(NUM_BATCHES, mininterval=10.0, desc="training"):
        model.train()
        optimizer.zero_grad()

        accumulated_loss = 0.0

        for _ in range(GRAD_ACCUM_EVERY):
            data = next(train_loader)
            inputs = data[:, :-1]
            targets = data[:, 1:]

            with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=USE_AMP):
                logits = model(inputs)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                accumulated_loss += loss.item()
                loss = loss / GRAD_ACCUM_EVERY
                loss.backward()

        if batch_num % PRINT_EVERY == 0:
            average_loss = accumulated_loss / GRAD_ACCUM_EVERY
            print(f"Loss at step {batch_num}: {average_loss:.4f}")

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

                if val_loss.item() < best_loss and batch_num > 0:
                    best_loss = val_loss.item()
                    out_dir.mkdir(exist_ok=True, parents=True)
                    print(f"Best val_loss updated: {best_loss:.3f}")
                    print(f"Saving model to {str(out_dir)}")

                    torch.save(model.state_dict(), out_dir / "best.pt")
                    with open(out_dir / "config.json", "w") as f:
                        json.dump(config.__dict__, f, indent=2)

                    last_checkpoint_step = batch_num

        if batch_num % GENERATE_EVERY == 0:
            print("\n" + "=" * 40 + " Generation " + "=" * 40)

            try:
                # Get sample from validation set
                val_sample = next(val_loader)[0]  # This includes EOS token!
                prime_tokens = val_sample[:PRIME_LENGTH]

                # Remove EOS tokens if present at the end
                while prime_tokens[-1] == tokenizer.eos_id:
                    prime_tokens = prime_tokens[:-1]

                # Debug info
                print("\nPrime tokens info:")
                print(f"- Shape: {prime_tokens.shape}")
                print(
                    f"- Value range: {prime_tokens.min().item()} to {prime_tokens.max().item()}"
                )
                print(f"- First few tokens: {prime_tokens[:10].tolist()}")
                print(f"- Last few tokens: {prime_tokens[-10:].tolist()}")
                print(f"- EOS token id: {tokenizer.eos_id}")

                # Decode and print prime text
                print("\nDecoding prime text...")
                prime_text = tokenizer.decode(prime_tokens)
                print(f"\nPrime text:\n{prime_text}")
                print("\n" + "=" * 80)

                # Generate continuation
                print("\nStarting generation...")
                prompt = prime_tokens.unsqueeze(0)  # Add batch dimension

                generated_tokens, continuation = base_decoding(
                    model, tokenizer, prompt, GENERATE_LENGTH, temperature=0.7
                )

                if continuation:
                    print(f"\nGenerated continuation:\n{continuation}")
                else:
                    print("\nNo continuation generated")

            except Exception as e:
                print(f"\nError in generation loop: {str(e)}")
                import traceback

                print(traceback.format_exc())

    print("Training complete")
    print(f"Last checkpoint step:\t{last_checkpoint_step}")
    print(f"Best loss:\t{best_loss:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
