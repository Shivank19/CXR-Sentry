import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchxrayvision as xrv
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .dataset import load_xray_png
from .model import CXRReportGenerator

def load_generator(checkpoint_dir: Path, device, xrv_cache: Path | None = None):
    with (checkpoint_dir / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    base_t5 = AutoModelForSeq2SeqLM.from_pretrained( config["base_model"], revision=config["base_model_revision"],)

    t5 = PeftModel.from_pretrained(base_t5, checkpoint_dir / "lora")

    densenet = xrv.models.DenseNet(weights=config["image_weights"], cache_dir=str(xrv_cache) if xrv_cache is not None else None,)

    # Verify that the frozen DenseNet weights match the training run.
    with Path(densenet.weights_filename_local).open("rb") as file:
        actual_hash = hashlib.file_digest(file, "sha256").hexdigest()
    if actual_hash != config["image_weights_sha256"]:
        raise ValueError("DenseNet weights do not match the training checkpoint")

    generator = CXRReportGenerator(densenet, t5, tokenizer, config["prompt"])

    # These weights belong to the visual tokenizer, outside the LoRA checkpoint.
    visual_state = torch.load(checkpoint_dir / "visual_tokenizer.pt", map_location="cpu", weights_only=True,)

    generator.encoder_inputs.visual_tokenizer.load_state_dict(visual_state)

    return generator.to(device).eval(), tokenizer

@torch.inference_mode()
def generate_token_ids(generator, images: torch.Tensor, *,  max_new_tokens: int = 128, num_samples: int = 0, temperature: float = 0.7, top_p: float = 0.9, ) -> torch.Tensor:
    # The shared model code builds encoder inputs from images and the prompt.
    inputs_embeds, attention_mask = generator.prepare_encoder_inputs(images)
    options = {"do_sample": False, "num_beams": 1}
    if num_samples > 0:
        options.update(
            do_sample=True,
            num_return_sequences=num_samples,
            temperature=temperature,
            top_k=0,
            top_p=top_p,
        )

    return generator.t5.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        **options,
    )

def main():
    parser = argparse.ArgumentParser(description="Generate drafts from one X-ray.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--xrv-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.samples < 0 or args.max_new_tokens < 1:
        parser.error("--samples must be >= 0 and --max-new-tokens must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")

    torch.manual_seed(args.seed)

    generator, tokenizer = load_generator(args.checkpoint, args.device, args.xrv_cache)
    images = load_xray_png(args.image).to(args.device)
    token_ids = generate_token_ids(generator, images, max_new_tokens=args.max_new_tokens, num_samples=args.samples)

    drafts = tokenizer.batch_decode(token_ids.cpu(), skip_special_tokens=True)
    for index, draft in enumerate(drafts, start=1):
        print(f"Draft {index}:{draft}")

if __name__ == "__main__":
    main()