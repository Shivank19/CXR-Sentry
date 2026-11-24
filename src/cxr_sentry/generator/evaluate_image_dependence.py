import argparse
import csv
import random
from pathlib import Path

import torch

from .dataset import CXRManifestDataset, VALID_SPLITS, collate_reports
from .generate import generate_token_ids, load_generator


@torch.inference_mode()
def score_condition(generator, tokenizer, image, labels, device, max_new_tokens):
    # Each dataset image is [1, 224, 224]. The model expects a batch.
    images = image.unsqueeze(0).to(device)

    # The reference is used only to score the predictions.
    loss = generator(images, labels).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError("Reference loss is not finite")

    # Generation still gets only the image and the fixed prompt.
    sequence = generate_token_ids(generator, images, max_new_tokens=max_new_tokens)
    draft = tokenizer.decode(sequence[0].cpu().tolist(), skip_special_tokens=True)
    return float(loss.item()), draft


def main():
    parser = argparse.ArgumentParser(description="Compare three image conditions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xrv-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    dataset = CXRManifestDataset(args.manifest, args.image_root, args.split)
    if len(dataset) < 2:
        parser.error("The selected split needs at least two studies")

    # Shuffle the study order once, then pair each study with its successor.
    # This gives each study a different image and uses every image once.
    order = list(range(len(dataset)))
    random.Random(args.seed).shuffle(order)
    shuffled_index = dict(zip(order, order[1:] + order[:1]))

    generator, tokenizer = load_generator(args.checkpoint, args.device, args.xrv_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    conditions = ("matched", "shuffled", "constant")
    loss_totals = dict.fromkeys(conditions, 0.0)
    changed_drafts = {"shuffled": 0, "constant": 0}
    total_tokens = 0

    columns = [
        "study_id", "split", "condition", "image_study_id",
        "reference_mean_nll", "n_reference_tokens",
        "generated_text", "draft_changed_vs_matched",
    ]
    with args.output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for index in range(len(dataset)):
            study = dataset[index]
            other = dataset[shuffled_index[index]]

            # Tokenize this study's reference once and reuse it in all conditions.
            labels = collate_reports([study], tokenizer)["labels"].to(args.device)
            n_tokens = int((labels != -100).sum().item())
            total_tokens += n_tokens

            variants = (
                ("matched", study["image"], study["study_id"]),
                ("shuffled", other["image"], other["study_id"]),
                ("constant", torch.zeros_like(study["image"]), ""),
            )
            matched_draft = ""
            for condition, image, image_id in variants:
                loss, draft = score_condition(
                    generator, tokenizer, image, labels,
                    args.device, args.max_new_tokens,
                )
                loss_totals[condition] += loss * n_tokens

                if condition == "matched":
                    matched_draft = draft
                changed = draft != matched_draft
                if condition != "matched":
                    changed_drafts[condition] += int(changed)

                writer.writerow({
                    "study_id": study["study_id"],
                    "split": args.split,
                    "condition": condition,
                    "image_study_id": image_id,
                    "reference_mean_nll": loss,
                    "n_reference_tokens": n_tokens,
                    "generated_text": draft,
                    "draft_changed_vs_matched": changed,
                })

    for condition in conditions:
        mean_loss = loss_totals[condition] / total_tokens
        print(f"{condition}: reference NLL per token {mean_loss:.4f}")

    for condition, count in changed_drafts.items():
        print(f"{condition}: {count}/{len(dataset)} drafts changed vs. matched")

    print(f"Saved {3 * len(dataset)} rows to {args.output}")


if __name__ == "__main__":
    main()