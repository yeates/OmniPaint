<!-- # OmniPaint -->

<!-- <p align="center">
  <img src="examples/logo/omnipaint-logo.png" height="120">
</p> -->

<h1 align="center">OmniPaint: Mastering Object-Oriented Editing via Disentangled Insertion-Removal Inpainting</h1>
<h3 align="center">ICCV 2025</h3>
<!-- <h3 align="center">arXiv 2025</h3> -->

<p align="center">
  <a href="https://arxiv.org/pdf/2503.08677"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg"></a>
  &nbsp;
  <a href="https://www.yongshengyu.com/OmniPaint-Page/"><img src="https://img.shields.io/badge/Website-Project-6535a0"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/zengziyun/OmniPaint"><img src="https://img.shields.io/badge/HuggingFace-Space-FFD21E?logo=huggingface&logoColor=FFD21E"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/yeates/omnipaint-bench"><img src="https://img.shields.io/badge/HuggingFace-Dataset-FFD21E?logo=huggingface&logoColor=FFD21E"></a>
</p>



<p align="center">
  <img src="assets/removal_demo.gif" style="width:85%;" />
</p>

<p align="center">
  <img src="assets/insertion_demo.gif" style="width:85%;" />
</p>


## Features

- 🧹 **Object Removal** - Remove foreground objects and their effects using only object masks
- 🖼️ **Object Insertion** - Seamless generative insertion of objects into existing scenes
- 📊 **Novel CFD Metric** - Reference-free evaluation of object removal quality


## Setup

Install dependencies and download OmniPaint weights:
  ```bash
  bash scripts/setup.sh
  ```
  Run this from the repository root. 


## Usage

### CLI - Object Removal
- Single image:
```bash
python scripts/omnipaint_remove.py \
  --input ./demo_assets/removal_samples/images/5.jpg \
  --mask ./demo_assets/removal_samples/masks/5.png \
  --output_dir ./outputs \
  --seed 42 \
  --steps 28 \
  --device cuda:0
```

- Directory:
```bash
python scripts/omnipaint_remove.py \
  --input ./demo_assets/removal_samples/images \
  --mask ./demo_assets/removal_samples/masks \
  --output_dir ./outputs \
  --seed 42 \
  --steps 28 \
  --device cuda:0
```

### CLI - Object Insertion
- Single image:
```bash
python scripts/omnipaint_insert.py \
  --background ./demo_assets/insertion_samples/backgrounds/background-2.png \
  --mask ./demo_assets/insertion_samples/masks/mask-2.png \
  --subject ./demo_assets/insertion_samples/subjects/subject-2.png \
  --output_dir ./outputs \
  --seed 42 \
  --steps 28 \
  --device cuda:0 \
  --carvekit_device cuda:0
```

- Directory:
```bash
python scripts/omnipaint_insert.py \
  --background ./demo_assets/insertion_samples/backgrounds \
  --mask ./demo_assets/insertion_samples/masks \
  --subject ./demo_assets/insertion_samples/subjects \
  --output_dir ./outputs \
  --seed 42 \
  --steps 28 \
  --device cuda:0 \
  --carvekit_device cuda:0
```

### Demo App

The demo app supports both manual mask drawing and automatic mask generation using segmentation model.

- Gradio environment setup:
  ```bash
  bash scripts/app_setup.sh
  ```
  Run this if you plan to launch `app.py`. The installation may take around 15 minutes due to SAM2 setup and weight downloads.

- Run the app:
  ```bash
  python app.py
  ```

### Notes
- Directory mode expects the following structure by default:
  - Removal: `demo_assets/removal_samples/images/*.{jpg,png,...}` with matching masks in `demo_assets/removal_samples/masks/*.{jpg,png,...}` (same basenames).
  - Insertion: `demo_assets/insertion_samples/backgrounds/background-XX.png`, masks in `demo_assets/insertion_samples/masks/mask-XX.png`, subjects in `demo_assets/insertion_samples/subjects/subject-XX.png`. Direct basename alignment also works.
- Mask quality strongly affects insertion performance. Prefer a single connected-component mask; avoid multiple disconnected masks.

### Evaluation - CFD Score
See [cfd_score](cfd_score/) for setup and usage.


<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=yeates.OmniPaint" width="1">
</div>
