import argparse
import csv
from pathlib import Path

from .dataset import CXRManifestDataset


def prepare_manifest(handoff_root: Path, output: Path):
    rows = []
    seen_studies = set()

    for split in ("train", "dev"):
        source = handoff_root / "data" / "splits" / f"{split}.csv"

        with source.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required = {"study_id", "image_path", "target_text", "view_label"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{source}: missing columns {sorted(missing)}")

            for record in reader:
                study_id = record["study_id"].strip()
                image_path = record["image_path"].strip()
                reference_text = record["target_text"]

                if not study_id or not image_path or not reference_text.strip():
                    raise ValueError(f"{source}: incomplete study record")

                # Check across both files to catch overlap between splits.
                if study_id in seen_studies:
                    raise ValueError(f"Duplicate study ID: {study_id}")
                seen_studies.add(study_id)

                if record["view_label"] != "FRONTAL":
                    raise ValueError(f"Expected a frontal image: {study_id}")

                # Paths in the handoff are relative to its outer folder.
                relative_path = Path(image_path)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"Expected a relative path: {image_path}")
                if not (handoff_root / relative_path).is_file():
                    raise FileNotFoundError(handoff_root / relative_path)

                # Preserve the supplied target text and split assignment.
                rows.append({
                    "study_id": study_id,
                    "image_path": image_path,
                    "split": split,
                    "reference_text": reference_text,
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["study_id", "image_path", "split", "reference_text"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Check that the generated file passes the generator's existing parser.
    for split in ("train", "dev"):
        dataset = CXRManifestDataset(output, handoff_root, split)
        print(f"{split}: {len(dataset)} studies")

    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the prepared data handoff.")
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_manifest(args.handoff_root, args.output)