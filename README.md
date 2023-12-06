<p align="center">
    <img src="assets/logo_color.png" height=180>
</p>

# 😴 LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes 😴

<div align="center">

[![Project](https://img.shields.io/badge/Project-LucidDreamer-green)](https://luciddreamer-cvlab.github.io/)
[![ArXiv](https://img.shields.io/badge/Arxiv-2311.13384-red)](https://arxiv.org/abs/2311.13384)
[![Github](https://img.shields.io/github/stars/luciddreamer-cvlab/LucidDreamer)](https://github.com/luciddreamer-cvlab/LucidDreamer)
[![X](https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![LICENSE](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://github.com/luciddreamer-cvlab/LucidDreamer/blob/master/LICENSE)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ironjr/luciddreamer)

</div>

<p align="center">
    <img src="assets/demo.gif" height=256>
</p>

> #### [LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2311.13384)
> ##### \*[Jaeyoung Chung](https://robot0321.github.io/), \*[Suyoung Lee](https://esw0116.github.io/), [Hyeongjin Nam](https://hygenie1228.github.io/), [Jaerin Lee](http://jaerinlee.com/), [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)
###### \*Denotes equal contribution.

<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>

---


## ⚡ Usage

### Prerequisite

- Linux: Ubuntu>18.04

### Intall

```bash
conda create -n lucid python=3.9
conda activate lucid
pip install peft diffusers scipy numpy imageio[ffmpeg] opencv-python Pillow open3d torchvision gradio
pip install torch==2.0.1 timm==0.6.7 # ZoeDepth
pip install plyfile==0.8.1 # Gaussian splatting
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas # FAISS

cd submodules/depth-diff-gaussian-rasterization-min
python setup.py install
cd ../simple-knn
python setup.py install
cd ../..
```

### Run with your own samples

```bash
# Default Example
python run.py
``` 

To run with your own inputs and prompts, attach following arguments after ``run.py``.

- ``-img`` : path of input image.
- ``-t`` : text prompt. Can be either path to txt file or the text iteslf.
- ``-nt`` : negative text prompt. Can be either path to txt file or the text iteslf.
- ``-cg`` : camera extrinsic path for generating scenes. Can be one of "Rotate_360", "LookAround", or "LookDown".
- ``-cr`` : camera extrinsic path for rendering videos. Can be one of "Back_and_forth", "LLFF", or "Headbanging".
- ``--seed`` : manual seed for Stable Diffusion inpainting.
- ``--diff_steps`` : number of denoising steps for Stable Diffusion inpainting. Default is 50.
- ``-s`` : path to save results. 

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
