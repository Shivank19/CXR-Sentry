from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# from cxr_sentry.generator.model import GeneratorEncoderInputs
from cxr_sentry.generator.model import CXRReportGenerator
from cxr_sentry.generator.dataset import load_xray_png


# def load_xray_png(image_path: Path) -> torch.Tensor:
#     with Image.open(image_path) as image:
#         pixels = np.asarray(image.convert("L"), dtype=np.float32)

#     # Map 8-bit pixel values from [0, 255] to the range expected by the pretrained model, then add the channel dimension.
#     pixels = xrv.datasets.normalize(pixels, 255)
#     pixels = pixels[None, :, :]  # Add channel dimension

#     # Crop to a square before resizing to avoid stretching the image.
#     pixels = xrv.datasets.XRayCenterCrop()(pixels)
#     pixels = xrv.datasets.XRayResizer(224)(pixels)

#         # Add a batch dimension so DenseNet receives [batch, channel, height, width].
#     return torch.from_numpy(pixels).float().unsqueeze(0)

# def main():
#     model_name = "google/flan-t5-small"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     t5.eval()

#     prompt = "Write a chest X-Ray report with Findings and Impression."
#     target_report = (
#         "Findings: No focal airspace opacity. "
#         "Impression:  No acute cardiopulmonary abnormality." 
#     )

#     # The prompt conditions the encoder. The report supplies the decoder's
#     # correct answers for this teacher-forced pass.
#     prompt_tokens = tokenizer(prompt, return_tensors="pt")
#     labels = tokenizer(text_target=target_report, return_tensors="pt")["input_ids"]

#     # Simulate one DenseNet feature map, read the embedding width from T5 so the image projection matches the LM's input
#     image_path = Path("data/iu/images/CXR1002_IM-0004-1001.png")
#     image_batch = load_xray_png(image_path)
#     assert image_batch.shape == (1, 1, 224, 224)

#     # TorchXRayVision has its own weight cache, separate from HF_HOME.
#     # Set it explicitly so the NIH weights also download to D:.
#     cache_dir = Path(r"D:\ml-cache\torchxrayvision")
#     cache_dir.mkdir(parents=True, exist_ok=True)

#     densenet = xrv.models.DenseNet(
#         weights="densenet121-res224-nih",
#         cache_dir=str(cache_dir),
#     )

#     # eval() uses the pretrained batch-normalization statistics.
#     # requires_grad_(False) freezes DenseNet's parameters.
#     densenet.eval()
#     densenet.requires_grad_(False)

#     with torch.no_grad():
#         # Keep the 7x7 grid. DenseNet's classifier would average it away.
#         image_features = densenet.features(image_batch).relu()

#     assert image_features.shape == (1, 1024, 7, 7)
#     print(f"DenseNet input shape: {tuple(image_batch.shape)}")
#     print(f"DenseNet feature shape: {tuple(image_features.shape)}")    
#     encoder_inputs = GeneratorEncoderInputs(image_channels=1024, embedding_dim=t5.config.d_model,)

#     # Skip gradient tracking, since this is just a forward pass test
#     with torch.inference_mode():
#         prompt_embeddings = t5.get_input_embeddings()(prompt_tokens["input_ids"])
#         inputs_embeds, attention_mask = encoder_inputs(image_features, prompt_embeddings, prompt_tokens["attention_mask"])

#         # T5 creates the shifted decoder inputs from labels and computes cross-entropy against the original target token IDs.

#         output = t5(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,)
    
#      # Expect a vocabulary score at every target position and one finite loss.
#     expected_shape = (1, labels.shape[1], t5.config.vocab_size)
#     assert output.logits.shape == expected_shape
#     assert output.loss.ndim == 0 and torch.isfinite(output.loss).item()

#     print(f"Encoder positions: {inputs_embeds.shape[1]}")
#     print(f"Target tokens: {labels.shape[1]}")
#     print(f"Logits shape: {tuple(output.logits.shape)}")
#     print(f"Loss: {output.loss.item():.4f}")
#     print("FLAN-T5 forward check passed.")


def main():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    cache_dir = Path(r"D:\ml-cache\torchxrayvision")
    cache_dir.mkdir(parents=True, exist_ok=True)
    densenet = xrv.models.DenseNet(
        weights="densenet121-res224-nih",
        cache_dir=str(cache_dir),
    )

    prompt = "Write a chest X-Ray report with Findings and Impression."
    generator = CXRReportGenerator(densenet, t5, tokenizer, prompt)
    generator.eval()

    image_path = Path("data/iu/images/CXR1002_IM-0004-1001.png")
    images = load_xray_png(image_path)
    assert images.shape == (1, 1, 224, 224)

    # A synthetic target gives T5 tokens to score for this check.
    target_report = (
        "Findings: No focal airspace opacity. "
        "Impression: No acute cardiopulmonary abnormality."
    )
    labels = tokenizer(text_target=target_report, return_tensors="pt")["input_ids"]

    # Call the same model interface that train.py will use.
    with torch.inference_mode():
        output = generator(images, labels)

    assert output.logits.shape == (1, labels.shape[1], t5.config.vocab_size)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss).item()

    print(f"Image batch: {tuple(images.shape)}")
    print(f"Logits shape: {tuple(output.logits.shape)}")
    print(f"Loss: {output.loss.item():.4f}")
    print("CXRReportGenerator forward check passed.")

if __name__ == "__main__":
    main()