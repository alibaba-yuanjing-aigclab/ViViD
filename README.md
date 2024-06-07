# ViViD
ViViD: Video Virtual Try-on using Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2405.11794-b31b1b.svg)](https://arxiv.org/abs/2405.11794)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://alibaba-yuanjing-aigclab.github.io/ViViD)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/fangzx/ViViD)

## ❗️Note
The weights are still under review by the company. They will be released soon after approval.


## Installation

```
git clone https://github.com/alibaba-yuanjing-aigclab/ViViD
cd ViViD
```

### Environtment
```
conda create -n vivid python=3.10
conda activate vivid
pip install -r requirements.txt  
```

### Weights
You can place the weights anywhere you like, for example, ```./ckpts```. If you put them somewhere else, you just need to update the path in ```./configs/prompts/*.yaml```.


#### Stable Diffusion Image Variations
```
cd ckpts

git lfs install
git clone https://huggingface.co/lambdalabs/sd-image-variations-diffusers
```
#### SD-VAE-ft-mse
```
git lfs install
git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
```
#### Motion Module
Download [mm_sd_v15_v2](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt)

#### ViViD
```
git lfs install
git clone https://huggingface.co/fangzx/ViViD
```
## Inference
We provide two demos in ```./configs/prompts/```, run the following commands to have a try😼.

```
python vivid.py --config ./configs/prompts/upper1.yaml

python vivid.py --config ./configs/prompts/lower1.yaml
```

## Data
As illustrated in ```./data```, the following data should be provided.
```text
./data/
|-- agnostic
|   |-- video1.mp4
|   |-- video2.mp4
|   ...
|-- agnostic_mask
|   |-- video1.mp4
|   |-- video2.mp4
|   ...
|-- cloth
|   |-- cloth1.jpg
|   |-- cloth2.jpg
|   ...
|-- cloth_mask
|   |-- cloth1.jpg
|   |-- cloth2.jpg
|   ...
|-- densepose
|   |-- video1.mp4
|   |-- video2.mp4
|   ...
|-- videos
|   |-- video1.mp4
|   |-- video2.mp4
|   ...
```

### Agnostic and agnostic_mask video
This part is a bit complex, you can obtain them through any of the following three ways:
1. Follow [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) to extract them frame-by-frame.(recommended)
2. Use [SAM](https://github.com/facebookresearch/segment-anything) + Gaussian Blur.(see ```./tools/sam_agnostic.py``` for an example)
3. Mask editor tools.

Note that the shape and size of the agnostic area may affect the try-on results.

### Densepose video
See [vid2densepose](https://github.com/Flode-Labs/vid2densepose).(Thanks)

### Cloth mask
Any detection tool is ok for obtaining the mask, like [SAM](https://github.com/facebookresearch/segment-anything).

## BibTeX
```text
@misc{fang2024vivid,
        title={ViViD: Video Virtual Try-on using Diffusion Models}, 
        author={Zixun Fang and Wei Zhai and Aimin Su and Hongliang Song and Kai Zhu and Mao Wang and Yu Chen and Zhiheng Liu and Yang Cao and Zheng-Jun Zha},
        year={2024},
        eprint={2405.11794},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```

## Contact Us
**Zixun Fang**: [zxfang1130@gmail.com](mailto:zxfang1130@gmail.com)  
**Yu Chen**: [chenyu.cheny@alibaba-inc.com](mailto:chenyu.cheny@alibaba-inc.com)  

