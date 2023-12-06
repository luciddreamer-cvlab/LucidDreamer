<p align="center">
    <img src="assets/logo_color.png" height=180>
</p>

# 😴 LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes 😴

<div align="center">

[![Project](https://img.shields.io/badge/Project-LucidDreamer-green)](https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/LICENSE)
[![ArXiv](https://img.shields.io/badge/Arxiv-2311.13384-red)](https://arxiv.org/abs/2311.13384)
[![Github](https://img.shields.io/github/stars/luciddreamer-cvlab/LucidDreamer?label=Github&color=blue)]()
[![X](https://img.shields.io/twitter/url?url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ironjr/luciddreamer)
[![LICENSE](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/LICENSE)

</div>

<div align="center">
  <tr>
    <td><video width="200" height="200" src="https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/assets/animestreet2_back_rgb.mp4"></video></td>
    <td><video width="200" height="200" src="https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/assets/fig5ours_360_rgb.mp4"></video></td>
    <td><video width="200" height="200" src="https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/assets/waterfall_back_rgb.mp4"></video></td>
  </tr>
</div>

> #### [LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2311.13384)
> ##### \*[Jaeyoung Chung](https://robot0321.github.io/), \*[Suyoung Lee](https://esw0116.github.io/), [Hyeongjin Nam](https://hygenie1228.github.io/), [Jaerin Lee](http://jaerinlee.com/), [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)
> ###### \*Denotes equal contribution.

<p align="center">
    <img src="assets/logo_cvlab.png" height=90>
</p>


---

## 📋 **Table of Contents** <!-- omit in toc -->
- [**Usage**](#usage)
  - [Prerequisite](#prerequisite)
  - [Install](#install)
  - [Run](#run)
- [**Updates**](#updates)
- [**Citation**](#citation)
- [**Acknowledgement**](#acknowledgement)
- [**Contact**](#contact)

---


## ⚡ Usage

### Prerequisite

- Linux: Ubuntu>18.04

### Intall

```bash
conda create -n lucid python=3.9
conda activate lucid
pip install peft diffusers scipy numpy imageio[ffmpeg] Pillow open3d torchvision gradio
pip install torch==2.0.1 timm==0.6.7 # ZoeDepth
pip install plyfile==0.8.1 # Gaussian splatting
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas # FAISS

cd submodules/depth-diff-gaussian-rasterization-min
python setup.py install
cd ../simple-knn
python setup.py install
```

### Run

```bash
python run.py
``` 

---

## 🚩 **Updates**
 
- ✅ December 6, 2023: Code release!
- ✅ November 22, 2023: We have released our paper, LucidDreamer on [arXiv](https://arxiv.org/abs/2311.13384).

## 🌏 Citation

Please cite us if you find our project useful!

```latex
@article{chung2023luciddreamer,
    title={LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes},
    author={Chung, Jaeyoung and Lee, Suyoung and Nam, Hyeongjin and Lee, Jaerin and Lee, Kyoung Mu},
    journal={arXiv preprint arXiv:2311.13384},
    year={2023}
}
```

## 🤗 Acknowledgement

We deeply appreciate [ZoeDepth](https://github.com/isl-org/ZoeDepth), [Stability AI](), and [Runway](https://huggingface.co/runwayml/stable-diffusion-v1-5) for their models.

## 📧 Contact

If you have any questions, please email `robot0321@snu.ac.kr`, `esw0116@snu.ac.kr`.
