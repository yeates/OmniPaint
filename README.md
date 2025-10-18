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
  <a href="https://7d71823049c6198a3f.gradio.live"><img src="https://img.shields.io/badge/Demo-Gradio-f07700"></a>
</p>

## 🔥🔥🔥 News!!
[October 18, 2025] **CFD Score** is released! CFD Score is a reference-free evaluation metric for object removal quality.

[July 21, 2025] **OmniPaint** is now live! You can try editing images directly in the [online demo](https://7d71823049c6198a3f.gradio.live)! For batch inference requests, please use this [form](https://forms.gle/pADR9j9P189Ag8sTA).

---
> **🚨 CODE COMING SOON! Please stay tuned...**

This repository will provide the official PyTorch implementation of **OmniPaint**, a framework that re-conceptualizes object removal and insertion as interdependent processes.

<p align="center">
  <img src="assets/removal_demo.gif" style="width:85%;" />
</p>

<p align="center">
  <img src="assets/insertion_demo.gif" style="width:85%;" />
</p>

## CFD Score
Follow the steps below to use CFD score, set up the environment, and install dependencies. The code is tested on **Python 3.10**.
```bash
cd cfd_score
./environment.sh
./run_cfd.sh
```


## Features

- 🧹 **Object Removal** - Remove foreground objects and their effects using only object masks
- 🖼️ **Object Insertion** - Seamless generative insertion of objects into existing scenes
- 📊 **Novel CFD Metric** - Reference-free evaluation of object removal quality

## Coming Soon

- [x] Demo application
- [x] Batch inference request form
- [ ] Model weights
- [ ] Training and inference code
- [ ] Dataset
- [ ] Evaluation metrics
- [ ] CFD evaluation code


## ⚠️ Disclaimer

This repository is part of an open-source research initiative provided for academic and research purposes only.

<div align="center">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=yeates.OmniPaint" width="1">
</div>
