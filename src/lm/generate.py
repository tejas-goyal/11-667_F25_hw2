import argparse
import json
import os
import math

import tiktoken
import torch
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    scaled_logits = logits / temperature
    return torch.nn.functional.softmax(scaled_logits, dim=-1)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes; computes perplexity

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """
    model.eval()
    generations = []
    #perplexity = ...
    all_generated_tokens = []

    #batch
    for batch_start in trange(0, len(prefixes), batch_size, desc="Generating"):
        batch_prefixes = prefixes[batch_start : batch_start + batch_size]
        # Tokenize the prefixes
        tokenized_prefixes = [tokenizer.encode(prefix) for prefix in batch_prefixes]
        max_prefix_len = max(len(tp) for tp in tokenized_prefixes)
        # Left-pad with eot_token (which is 0)
        pad_token_id = tokenizer.eot_token
        input_ids = []
        attention_mask = []
        for tokens in tokenized_prefixes:
            padding_length = max_prefix_len - len(tokens)
            input_ids.append([pad_token_id] * padding_length + tokens)
            attention_mask.append([0] * padding_length + [1] * len(tokens))
        #convert to tensors and move to device
        input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float).to(device)
        #generate tokens
        generated_ids = input_ids.clone()
        generated_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            logits = model(generated_ids, generated_mask)
            next_token_logits = logits[:, -1, :]  # (B, V), Get logits for the last position
            probs = softmax_with_temperature(next_token_logits, temperature) # Apply temperature and softmax
            next_token_id = torch.multinomial(probs, num_samples=1)  # (B, 1), Sample from the distribution
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)# Append to generated sequence
            new_mask = torch.ones((generated_ids.shape[0], 1), dtype=torch.float).to(device)# Extend attention mask (new tokens are always attended to)
            generated_mask = torch.cat([generated_mask, new_mask], dim=1)
        # Extract only the newly generated tokens (after the prefix)
        new_tokens = generated_ids[:, max_prefix_len:]
        all_generated_tokens.append(new_tokens)
        
        # Decode the generated tokens
        for i, tokens in enumerate(new_tokens):
            generated_text = tokenizer.decode(tokens.tolist())
            generations.append(generated_text)
    # Compute perplexity on the generated text and concatenate all generated tokens
    all_generated = torch.cat(all_generated_tokens, dim=0)
    total_loss = 0.0
    total_tokens = 0
    
    for batch_start in range(0, all_generated.shape[0], batch_size):
        batch_tokens = all_generated[batch_start:batch_start + batch_size]
        # Generate logits for the batch
        logits = model(batch_tokens)
        loss = compute_language_modeling_loss(batch_tokens, logits)
        # Accumulate
        num_tokens = (batch_tokens.shape[0] * (batch_tokens.shape[1] - 1))
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss)
    print(f"Perplexity: {perplexity}")
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
