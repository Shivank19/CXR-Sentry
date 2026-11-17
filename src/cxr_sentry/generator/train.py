import argparse
import csv
import time
from functools import partial
from pathlib import Path
import hashlib
import json
import math

import torch
import torchxrayvision as xrv
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from .dataset import CXRManifestDataset, collate_reports
from .model import CXRReportGenerator

MODEL_NAME = "google/flan-t5-small"
IMAGE_WEIGHTS = "densenet121-res224-nih"
PROMPT = "Write a chest X-Ray report with Findings and Impression."

def parse_args():
    parser = argparse.ArgumentParser(description="Train the CXR-Sentry report generator.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xrv-cache", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment.")
    return args


def train_step(generator, images, labels, optimizer) -> float:
    # Clear old gradients, compute new ones, then update the trainable weights.
    optimizer.zero_grad(set_to_none=True)
    loss = generator(images, labels).loss
    if not torch.isfinite(loss).item():
        raise RuntimeError(f"Training loss is not finite: {loss.item()}")
    loss.backward()
    optimizer.step()
    return loss.item()


def run_epoch(generator, loader, device, optimizer=None):
    training = optimizer is not None
    generator.train(training)

    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        # Count targets on the CPU before moving the batch to the model device.
        # The batch builder uses -100 for padding, which T5 ignores.
        token_count = (batch["labels"] != -100).sum().item()
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)

        if training:
            batch_loss = train_step(generator, images, labels, optimizer)
        else:
            # Dev measures the current model without computing gradients.
            with torch.inference_mode():
                batch_loss = generator(images, labels).loss.item()

        # T5 returns a mean over this batch's scored tokens. Multiply by the token count before combining batches of different lengths.
        total_loss += batch_loss * token_count
        total_tokens += token_count

    return total_loss / total_tokens, total_tokens

def save_best_checkpoint(generator, tokenizer, checkpoint_dir, metadata):
    """Save the learned weights and settings needed to rebuild the generator."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # PEFT saves the learned T5 adapters and their LoRA settings.
    generator.t5.save_pretrained(checkpoint_dir/"lora")
    tokenizer.save_pretrained(checkpoint_dir/"tokenizer")

    # The visual projection and learned positions live outside PEFT.
    visual_state = {
        name: tensor.detach().cpu()
        for name, tensor in generator.encoder_inputs.visual_tokenizer.state_dict().items()
    }
    torch.save(visual_state, checkpoint_dir / "visual_tokenizer.pt")

    # Record the base weights, preprocessing, and selection metric for reload.
    with (checkpoint_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    train_data = CXRManifestDataset(args.manifest, args.image_root, "train")
    dev_data = CXRManifestDataset(args.manifest, args.image_root, "dev")

    # Use a fresh directory so a rerun cannot overwrite an earlier metrics CSV.
    args.run_dir.mkdir(parents=True, exist_ok=False)

    # Hub downloads are stored under snapshots/<commit>/.
    # Use that commit for both the tokenizer and the base model.
    
    config_path = Path(hf_hub_download(repo_id=MODEL_NAME, filename="config.json"))
    t5_revision = config_path.parent.name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=t5_revision)
    batch_builder = partial(collate_reports, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=batch_builder
    )
    dev_loader = DataLoader(
        dev_data, batch_size=args.batch_size, shuffle=False, collate_fn=batch_builder
    )

    t5 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, revision=t5_revision)
    t5 = get_peft_model(
        t5,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q", "v"],
            r=8,
            lora_alpha=16,
            bias="none",
        ),
    )

    # Keep machine-specific paths in the command. HF_HOME controls the
    # FLAN-T5 download cache. This separate option controls DenseNet's cache.
    cache_dir = None
    if args.xrv_cache is not None:
        args.xrv_cache.mkdir(parents=True, exist_ok=True)
        cache_dir = str(args.xrv_cache)

    densenet = xrv.models.DenseNet(weights=IMAGE_WEIGHTS, cache_dir=cache_dir)
    generator = CXRReportGenerator(densenet, t5, tokenizer, PROMPT).to(device)

    # PEFT freezes the base T5 weights. The generator freezes DenseNet.
    # AdamW receives the remaining trainable parameters.
    trainable = [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=0.0
    )

    print(f"Device: {device} | Train studies: {len(train_data)} | Dev studies: {len(dev_data)}")

    # The hash lets the reload code verify the exact DenseNet weights.
    with Path(densenet.weights_filename_local).open("rb") as file:
        image_weights_sha256 = hashlib.file_digest(file, "sha256").hexdigest()

    best_dev_loss = math.inf
    checkpoint_dir = args.run_dir / "best_checkpoint"
    checkpoint_config = {
        "base_model": MODEL_NAME,
        "base_model_revision": t5_revision,
        "image_weights": IMAGE_WEIGHTS,
        "image_weights_sha256": image_weights_sha256,
        "prompt": PROMPT,
        "preprocessing": "8-bit grayscale, XRV normalize(255), center crop, resize to 224",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_studies": len(train_data),
        "dev_studies": len(dev_data),
    }

    metrics_path = args.run_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["epoch", "train_loss", "dev_loss", "train_tokens", "dev_tokens",
             "learning_rate", "seconds"]
        )

        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss, train_tokens = run_epoch(generator, train_loader, device, optimizer)
            dev_loss, dev_tokens = run_epoch(generator, dev_loader, device)
            elapsed = time.perf_counter() - start

            writer.writerow(
                [epoch, train_loss, dev_loss, train_tokens, dev_tokens,
                 optimizer.param_groups[0]["lr"], elapsed]
            )
            file.flush()
            print(f"Epoch {epoch}: train loss {train_loss:.4f} | dev loss {dev_loss:.4f}")

            if not math.isfinite(dev_loss):
                raise RuntimeError(f"Dev loss is not finite at epoch {epoch}")

            if dev_loss < best_dev_loss:
                save_best_checkpoint(
                    generator,
                    tokenizer,
                    checkpoint_dir,
                    {**checkpoint_config, "epoch": epoch, "dev_loss": dev_loss},
                )
                best_dev_loss = dev_loss
                print(f"Saved best checkpoint from epoch {epoch} to {checkpoint_dir}")

    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
