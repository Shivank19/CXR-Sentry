from pathlib import Path, PureWindowsPath
import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image
import csv

from torch.utils.data import Dataset

def load_xray_png(image_path: Path) -> torch.Tensor:
    """Load one xray image froma PNG file as a [1, 1, 224, 224] tensor for DenseNet input"""
    with Image.open(image_path ) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float32)

    #DenseNet expects normalized greyscale pizels and a channel dimension.
    pixels = xrv.datasets.normalize(pixels, 255)[None, :, :]

    #Crop to a square before resizing to avoid stretching the image.
    pixels = xrv.datasets.XRayCenterCrop()(pixels)
    pixels = xrv.datasets.XRayResizer(224)(pixels)

    #Add a batch dimension so DenseNet receives [batch, channel, height, width].
    return torch.from_numpy(pixels).float().unsqueeze(0)

REQUIRED_COLUMNS = {"study_id", "image_path", "split", "reference_text"}
VALID_SPLITS = {"train", "dev", "prob_cal", "risk_cal", "test"}

class CXRManifestDataset(Dataset):
    """Read one assigned split and load its PNGs as examples are requested."""

    def __init__(self, manifest_path: Path, image_root: Path, split: str):
        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {split}")

        self.image_root = Path(image_root)
        self.rows = []
        seen_studies = set()

        # DictReader handles quoted report text that contains commas or newlines.
        # utf-8-sig also handles an optional encoding marker from CSV editors.
        with Path(manifest_path).open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

            for row in reader:
                study_id = (row["study_id"] or "").strip()
                image_path = (row["image_path"] or "").strip()
                row_split = (row["split"] or "").strip()
                reference_text = (row["reference_text"] or "").strip()
                location = f"CSV line {reader.line_num}"

                if not study_id or not image_path or not reference_text:
                    raise ValueError(f"{location}: missing study ID, path, or report")
                if row_split not in VALID_SPLITS:
                    raise ValueError(f"{location}: unknown split {row_split!r}")

                # Check across the entire manifest, including other splits.
                if study_id in seen_studies:
                    raise ValueError(f"{location}: duplicate study_id {study_id!r}")
                seen_studies.add(study_id)

                # Manifest paths must stay relative to the supplied image root.
                relative_path = PureWindowsPath(image_path)
                if relative_path.anchor or ".." in relative_path.parts:
                    raise ValueError(f"{location}: expected a relative image path")

                if row_split == split:
                    self.rows.append((study_id, relative_path.parts, reference_text))

        if not self.rows:
            raise ValueError(f"Manifest has no rows for split {split!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        study_id, path_parts, reference_text = self.rows[index]
        image_path = self.image_root.joinpath(*path_parts)
        if not image_path.is_file():
            raise FileNotFoundError(f"Study {study_id}: image missing at {image_path}")

        # The PNG helper returns [1, 1, 224, 224]. Remove its one-image batch dimension so the batch builder can stack several examples.
        return {
            "study_id": study_id,
            "image": load_xray_png(image_path).squeeze(0),
            "reference_text": reference_text,
        }

def collate_reports(examples, tokenizer):
    """Stack images and turn variable-length reports into a target batch."""
    images = torch.stack([example["image"] for example in examples])
    references = [example["reference_text"] for example in examples]

    # Reports can have different lengths. Pad only to the longest report in this batch so they can share one tensor.
    targets = tokenizer(text_target=references, padding=True, return_tensors="pt")

    # T5 ignores -100 when computing loss. Padding should contribute neither to the training objective nor to the reported token count.
    labels = targets["input_ids"].masked_fill(targets["attention_mask"] == 0, -100)

    return {
        "study_ids": [example["study_id"] for example in examples], 
        "images": images, 
        "labels": labels,
    }
