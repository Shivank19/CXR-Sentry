import argparse
import csv
from pathlib import Path

import torch

from .dataset import CXRManifestDataset, VALID_SPLITS
from .generate import generate_token_ids, load_generator

@torch.inference_mode()
def score_greedy_draft(generator, images: torch.Tensor, sequence: torch.Tensor,) -> tuple[float | None, int]:
    """ Score one generated sequence under its image and fixed prompt"""
    config = generator.t5.config

    # T5 generation returns the decoder start token in the first slot.
    
    if sequence[0].item() != config.decoder_start_token_id:
        raise ValueError("Expected a T5 decoder start token")

    generated_ids = sequence[1:]
    eos_positions = (generated_ids == config.eos_token_id).nonzero(as_tuple=True)[0]

    if eos_positions.numel():
        generated_ids = generated_ids[: eos_positions[0].item() + 1]

    # T5 ignores -100 when computing cross-entropy. Keeping the padding slots here preserves the decoder's token positions while excluding them from loss.
    labels = generated_ids.masked_fill(generated_ids == config.pad_token_id, -100)
    n_tokens = int((labels != -100).sum().item())
    if n_tokens == 0:
        return None, 0

    # Using the exact generated IDs. Decoding and re-tokenizing the draft could change the sequence we score.
    loss = generator(images, labels.unsqueeze(0)).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("Generated draft has non-finite mean token NLL")

    return float(loss.item()), n_tokens

def main():
    parser = argparse.ArgumentParser(description="Export greedy CXR-Sentry drafts.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xrv-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    dataset = CXRManifestDataset(args.manifest, args.image_root, args.split)
    generator, tokenizer = load_generator(args.checkpoint, args.device, args.xrv_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "study_id", "split", "generated_text", "checkpoint_id", "mean_nll", "n_tokens"
    ]
    with args.output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for index in range(len(dataset)):
            example = dataset[index]

            # A dataset item is [1, 224, 224]. Add back the batch dimension.
            images = example["image"].unsqueeze(0).to(args.device)
            sequence = generate_token_ids(
                generator, images, max_new_tokens=args.max_new_tokens
            )[0]
            mean_nll, n_tokens = score_greedy_draft(generator, images, sequence)
            generated_text = tokenizer.decode(
                sequence.cpu().tolist(), skip_special_tokens=True
            )

            writer.writerow({
                "study_id": example["study_id"],
                "split": args.split,
                "generated_text": generated_text,
                "checkpoint_id": args.checkpoint_id,
                "mean_nll": mean_nll,
                "n_tokens": n_tokens,
            })

    print(f"Exported {len(dataset)} greedy drafts to {args.output}")


if __name__ == "__main__":
    main()