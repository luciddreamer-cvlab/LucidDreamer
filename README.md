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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ironjr/LucidDreamer-mini)

</div>

https://github.com/luciddreamer-cvlab/LucidDreamer/assets/12259041/35004aaa-dffc-4133-b15a-05224e68b91e


> #### [LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2311.13384)
> ##### \*[Jaeyoung Chung](https://robot0321.github.io/), \*[Suyoung Lee](https://esw0116.github.io/), [Hyeongjin Nam](https://hygenie1228.github.io/), [Jaerin Lee](http://jaerinlee.com/), [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/~kmlee/)
###### \*Denotes equal contribution.

<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>

---


## 🤖 Install

### Ubuntu

#### Prerequisite

- CUDA>=11.4 (higher version is OK).
- Python==3.9 (cannot use 3.10 due to open3d compatibility)

#### Installation script

```bash
conda create -n lucid python=3.9
conda activate lucid
pip install peft diffusers scipy numpy imageio[ffmpeg] opencv-python Pillow open3d torchvision gradio
# ZoeDepth
pip install torch==2.0.1 timm==0.6.7
# Gaussian splatting
pip install plyfile==0.8.1

cd submodules/depth-diff-gaussian-rasterization-min
# sudo apt-get install libglm-dev # may be required for the compilation.
python setup.py install
cd ../simple-knn
python setup.py install
cd ../..
```

### Windows (Experimental, Tested on Windows 11 with VS2022)

#### Checklist

- Make sure that the versions of your installed [**CUDA**](https://developer.nvidia.com/cuda-11-8-0-download-archive), [**cudatoolkit**](https://anaconda.org/nvidia/cudatoolkit), and [**pytorch**](https://pytorch.org/get-started/previous-versions/) match. We have tested on CUDA==11.8.
- Make sure you download and install C++ (>=14) from the [Visual Studio build tools](https://visualstudio.microsoft.com/downloads/).

#### Installation script

```bash
conda create -n lucid python=3.9
conda activate lucid
conda install pytorch=2.0.1 torchvision=0.15.2 torchaudio=2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install peft diffusers scipy numpy imageio[ffmpeg] opencv-python Pillow open3d gradio
# ZoeDepth
pip install timm==0.6.7
# Gaussian splatting
pip install plyfile==0.8.1

cd submodules\depth-diff-gaussian-rasterization-min\third_party
git clone https://github.com/g-truc/glm.git
cd ..\
python setup.py install
cd ..\simple-knn
python setup.py install
cd ..\..
```

## ⚡ Usage

We offer several ways to interact with LucidDreamer:

1. A demo is available on [`ironjr/LucidDreamer` HuggingFace Space](https://huggingface.co/spaces/ironjr/LucidDreamer) (including custom SD ckpt) and [`ironjr/LucidDreamer-mini` HuggingFace Space](https://huggingface.co/spaces/ironjr/LucidDreamer-mini) (minimal features / try at here in case of the former is down)
(We appreciate all the HF / Gradio team for their support).

https://github.com/luciddreamer-cvlab/LucidDreamer/assets/12259041/745bfc46-8215-4db2-80d5-4825e91316bc

2. Another demo is available on a [Colab](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb), implemented by [@camenduru](https://github.com/camenduru)
(We greatly thank [@camenduru](https://github.com/camenduru) for the contribution).
3. You can use the gradio demo locally by running [`CUDA_VISIBLE_DEVICES=0 python app.py`](app.py) (full feature including huggingface model download, requires ~15GB) or [`CUDA_VISIBLE_DEVICES=0 python app_mini.py`](app_mini.py) (minimum viable demo, uses only SD1.5).
4. You can also run this with command line interface as described below.

### Run with your own samples

```bash
# Default Example
python run.py
``` 

To run with your own inputs and prompts, attach following arguments after ``run.py``.

- ``-img`` : path of input image.
- ``-t`` : text prompt. Can be either path to txt file or the text itself.
- ``-nt`` : negative text prompt. Can be either path to txt file or the text itself.
- ``-cg`` : camera extrinsic path for generating scenes. Can be one of "Rotate_360", "LookAround", or "LookDown".
- ``-cr`` : camera extrinsic path for rendering videos. Can be one of "Back_and_forth", "LLFF", or "Headbanging".
- ``--seed`` : manual seed for Stable Diffusion inpainting.
- ``--diff_steps`` : number of denoising steps for Stable Diffusion inpainting. Default is 50.
- ``-s`` : path to save results. 

### Guideline for the prompting / Troubleshoot

#### General guides

1. If your image is indoors with specific scene (and possible character in it), you can **just put the most simplest representation of the scene first**, like a cozy livingroom for christmas, or a dark garage, etc. Please avoid prompts like 1girl because it will generate many humans for each inpainting task.
2. If you want to start from already hard-engineered image from e.g., StableDiffusion model, or a photo taken from other sources, you can try **using [WD14 tagger](https://github.com/toriato/stable-diffusion-webui-wd14-tagger) ([huggingface demo](https://huggingface.co/spaces/deepghs/wd14_tagging_online)) to extract the danbooru tags from an image**. Please ensure you remove some comma separated tags if you don't want them to appear multiple times. This include human-related objects, e.g., 1girl, white shirt, boots, smiling face, red eyes, etc. Make sure to specify the objects you want to have multiples of them.

#### Q. I generate unwanted objects everywhere, e.g., photo frames.

1. Manipulate negative prompts to set harder constraints for the frame object. You may try adding tags like twitter thumbnail, profile image, instagram image, watermark, text to the negative prompt. In fact, negative prompts are the best thing to try if you want some things not to be appeared in the resulting image.
2. Actually, our SD1.5 model uses the standard stable-diffusion-inpainting-v1.5 model, which I think has vulnerability to generate framed photos. To avoid this, I'd rather try other custom checkpoint models, which employs different pipeline methods: LaMa inpainting -> ControlNet-inpaint guided image inpainting. I empirically find that this setting is more stable than the naive application of standard inpainting model.

### Visualize `.ply` files

There are multiple available viewers / editors for Gaussian splatting `.ply` files.

1. [@playcanvas](https://github.com/playcanvas)'s [Super-Splat](https://github.com/playcanvas/super-splat) project ([Live demo](https://playcanvas.com/super-splat)). This is the viewer we have used for our debugging along with MeshLab.

![image](https://github.com/luciddreamer-cvlab/LucidDreamer/assets/12259041/89c4b5dd-c66f-4ad2-b1be-e5f951273049)

2. [@antimatter15](https://github.com/antimatter15)'s [WebGL viewer](https://github.com/antimatter15/splat) for Gaussian splatting ([Live demo](https://antimatter15.com/splat/)).

3. [@splinetool](https://github.com/splinetool)'s [web-based viewer](https://spline.design/) for Gaussian splatting. This is the version we have used in our project page's demo.

## 🚩 **Updates**

- ✅ December 11, 2023: We have updated installation guides for Windows. Thank you [@Maoku](https://twitter.com/Maoku) for your great contribution!
- ✅ December 8, 2023: [HuggingFace Space demo](https://huggingface.co/spaces/ironjr/LucidDreamer) is out. We deeply thank all the HF team for their support!
- ✅ December 7, 2023: [Colab](https://colab.research.google.com/github/camenduru/LucidDreamer-Gaussian-colab/blob/main/LucidDreamer_Gaussian_colab.ipynb) implementation is now available thanks to [@camenduru](https://github.com/camenduru)!
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

If you have any questions, please email `robot0321@snu.ac.kr`, `esw0116@snu.ac.kr`, `jarin.lee@gmail.com`.

## ⭐ Star History

<a href="https://star-history.com/#luciddreamer-cvlab/LucidDreamer&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=luciddreamer-cvlab/LucidDreamer&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=luciddreamer-cvlab/LucidDreamer&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=luciddreamer-cvlab/LucidDreamer&type=Date" />
  </picture>
</a>
