<h2 align="center">SpatialReasoner: The first universal 3D reasoning framework
</a>

<h5 align="center">
<div align="center">

[Hongbo Jin](https://hongbo-jin.github.io/)<sup></sup>,
[Tinghong Ye]()<sup>*</sup>,
[Yaochen Liu]()<sup></sup>,
[Bo Tang]()<sup></sup>,
[Xinhua Wang]()<sup></sup>,
[Ge Li](https://openreview.net/profile?id=~Ge_Li2)<sup>✉</sup>

<sup></sup>School of Electronic and Computer Engineering, Peking University<br>

</div>


## 📰 News

- [2025-05] 🔊 Our [SpatialReasoner](https://github.com/Hongbo-Jin/HBLLaVA) repository is released!

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> About
**SpatialReasoner** is a universal 3D reasoning framework. It leverages reinforcement learning to enhance reasoning abilities while maintaining 3D perception ability. 
The model and training process are fully traceable, ensuring reproducibility and reliability. This repository provides the model, code, and experimental setups for easy replication.


## 🛠️ Installation

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/Hongbo-Jin/HBLLaVA.git
cd HBLLaVA
```

2. Create a conda environment, activate it and install Packages
```Shell
conda create -n hbllava python=3.10 -y
conda activate hbllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages (Find your corresponding architecture at this [link](https://github.com/Dao-AILab/flash-attention/releases))
```Shell
pip install flash-attn==2.7.3 --no-build-isolation
```
##### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## 📌 Usage

### 1. Data Preparation
The training data is based on scannet dataset, which can be downloaded from [here](https://huggingface.co/datasets/YiquanLi/ScanNet_for_ScanQA_SQA3D)

#### Organize Data

Organize the files and annotation files as follows in ``path/to/your/dataset``:

```Shell
data
├── Scannet
│   ├── downsample_32
├── ├── ScanQA-v1.0
├── NextQA
├── 
```

### 2. Train

#### 1. Cold Start

You can train the model yourself: 
Download [Qwen2.5-vl-3B-instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) as base model.

Replace data paths and model paths with yours in `scripts/train/coldstart.sh`

```bash
bash scripts/train/coldstart.sh
```

#### 2. GRPO Training

Replace data paths and output_dir with yours in `scripts/train/reason.sh`

```bash
bash scripts/train/reason.sh
```

### 3. Evaluation

We currently provide evaluations on 2 benchmarks, including [ScanQA](https://github.com/ATR-DBI/ScanQA) and [SQA3D](https://zenodo.org/records/7792397#.ZCkprfFBx3g)

#### ScanQA

1. Download [Scannet](https://github.com/ScanNet/ScanNet) and put it under ``path/to/your/dataset/Scannet``.
2. Please change ``model-path``, ``num-frame``, ``gt-file`` and ``data-folder`` in ``scripts/eval/eval_scanqa.sh``.
3. Please use the following command for inference.
   ```bash
   bash scripts/eval/eval_scanqa.sh
   ```
#### SQA3D

1. Download SQA3D gt files from its [official website](https://zenodo.org/records/7792397#.ZCkprfFBx3g).
2. Please change ``model-path``, ``num-frame``, ``gt-file`` and ``data-folder`` in ``scripts/eval/eval_sqa3d.sh``.
3. Please use the following command for inference.
   ```bash
   bash scripts/eval/eval_sqa3d.sh
   ```

### Quick Inference Scripts

1. Please change ``model_path``, ``prompt`` and ``video_file`` in ``eval.py``.
2.  Please use the following command for single-gpu inference.
   ```bash
   CUDA_VISIBLE_DEVICES=0 python eval.py
   ```

## 📝 Citation

If you find our work interesting and helpful, please consider giving our repo a star. Additionally, if you would like to cite our work, please use the following format:
```bibtex

```

## 📨 Contact

If you have any questions or suggestions, please feel free to contact us at ``hongbo@hust.edu.cn``.
