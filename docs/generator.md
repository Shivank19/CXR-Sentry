# CXR-Sentry generator guide

[Back to the project overview](../README.md)

The generator uses a frozen NIH-trained TorchXRayVision DenseNet-121, a trainable projection and learned positions for 16 spatial tokens, and FLAN-T5 Small with query/value LoRA adapters. Training uses the reference as the decoder target. Draft generation uses the image and a fixed prompt.

## Environment setup

Run commands from the repository root. Create and activate a Python 3.13 environment.

The package lives under `src/` and is run directly from the checkout. The `PYTHONPATH` line makes `cxr_sentry` importable in the current shell. Repeat that line in each new shell.

```cmd
py -3.13 -m venv .venv
.venv\Scripts\activate.bat
set "PYTHONPATH=%CD%\src"
```

Install a matching `torch` and `torchvision` pair for the target OS and accelerator. Select a CUDA build for an NVIDIA GPU, or a CPU build for CPU runs. [PyTorch installation selector](https://pytorch.org/get-started/locally/)

Install the pinned project dependencies:

```text
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use `--device cuda` when the check prints `CUDA available: True`. The examples below use `--device cpu`. Training additionally supports `--device auto`.

## Dataset and run paths

Use the prepared CSV and PNGs from the dataset owner. The manifest should have one row per study and contain `study_id`, `image_path`, `split`, and `reference_text`. `image_path` is relative to the image root. The dataset owner selects one frontal image per eligible study and supplies the shared split assignments. Use `train`, `dev`, `prob_cal`, `test`, and `risk_cal` if provided. Reference text uses Findings followed by Impression, with headings for the sections present. Exclude rows where both sections are empty.

Example layout under the repo root:

```text
data/iu/
├── images/
└── manifest.csv
```

The commands use example paths relative to the repository root. Substitute the actual manifest, image root, and run directory as needed. Use a fresh run directory for each training run and a distinct checkpoint ID for each model you export.

The first run downloads any missing pretrained weights into the libraries’ default caches.

Keep real data, model caches, checkpoints, and run outputs out of Git. Use the team’s shared storage to hand off a trained checkpoint and its exports.

## Train and plot

The settings below are example run settings. Start with a tiny paired-data run to check loading and confirm the model can overfit a few examples before the full run. Training uses the `train` split, and the lowest loss on `dev` selects the checkpoint. The test split is held for evaluation.

```text
python -m cxr_sentry.generator.train --manifest data/iu/manifest.csv --image-root data/iu/images --run-dir runs/baseline-01 --device cpu --epochs 3 --batch-size 2 --learning-rate 0.0001 --seed 42
python -m cxr_sentry.generator.plot_metrics --run-dir runs/baseline-01
```

The run folder contains `metrics.csv`, `loss_curves.png`, `loss_curves.svg`, and `best_checkpoint/`. The metrics record train and dev cross-entropy per non-padding target token. The checkpoint contains LoRA adapters, the tokenizer, the visual projection and position weights, and metadata identifying the frozen base models, prompt, and preprocessing. Keep the full checkpoint directory together for reload.

## Generate and export

Replace `data/iu/images/example.png` with an existing PNG. The first generation command prints one greedy draft. The second prints five sampled drafts, using temperature 0.7 and top-p 0.9.

```text
python -m cxr_sentry.generator.generate --checkpoint runs/baseline-01/best_checkpoint --image data/iu/images/example.png --device cpu --max-new-tokens 128
python -m cxr_sentry.generator.generate --checkpoint runs/baseline-01/best_checkpoint --image data/iu/images/example.png --device cpu --max-new-tokens 128 --samples 5 --seed 42
```

`--samples 5` generates five drafts from the same image. For corpus-wide sampling, load the checkpoint once with `load_generator()`, iterate over the required manifest splits, and call `generate_token_ids(..., num_samples=5)` for each study. Record the `study_id` and sample index with each draft. The `generate` CLI takes one image at a time.

Export one greedy draft per study. Run the command once per split you need, giving each export a distinct output path.

```text
python -m cxr_sentry.generator.export --checkpoint runs/baseline-01/best_checkpoint --checkpoint-id baseline-01 --manifest data/iu/manifest.csv --image-root data/iu/images --split test --output runs/baseline-01/greedy_test.csv --device cpu --max-new-tokens 128
```

The export columns are `study_id`, `split`, `generated_text`, `checkpoint_id`, `mean_nll`, and `n_tokens`. Mean NLL scores the exact generated IDs under the study image and prompt. It excludes the decoder start token and padding, includes the first EOS token if emitted, and ignores anything after EOS. If there are zero scorable tokens, `mean_nll` is blank and `n_tokens` is zero.

## Evaluate image dependence

The selected split needs at least two studies. Each study’s reference text stays fixed while the script compares its own image, a different study’s image, and a constant normalized input. It reports token-weighted reference NLL and how often the greedy draft changes relative to the matched image.

```text
python -m cxr_sentry.generator.evaluate_image_dependence --checkpoint runs/baseline-01/best_checkpoint --manifest data/iu/manifest.csv --image-root data/iu/images --split test --output runs/baseline-01/image_dependence_test.csv --device cpu --max-new-tokens 128 --seed 42
python -m cxr_sentry.generator.plot_image_dependence --input runs/baseline-01/image_dependence_test.csv
```

The plot command saves the condition summary CSV and PNG/SVG figures beside the per-study evaluation CSV.

## Reproducing a run

Keep the Git commit, exact manifest and split assignments, command-line arguments, environment snapshot (`python -m pip freeze`), and hardware details with the run artifacts. Preserve the checkpoint's `config.json`, which records the resolved FLAN-T5 revision and the SHA-256 hash of the DenseNet weights. Checkpoint reload uses the recorded FLAN-T5 revision and verifies the DenseNet hash. A new training run resolves the current Hub revision, so check that it matches the recorded revision when reproducing an earlier experiment.

