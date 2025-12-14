# AutoMix-finetuning
For fine tuning musicgen and audioldm

## 🚀 1. Environment Setup
AutoMix-finetuning uses both AudioLDM and Audiocraft, installed separately.
### 1.1.0 Install AudioLDM Environment
Use this as reference and download the necessaries checkpoints here: https://github.com/haoheliu/AudioLDM-training-finetuning
⚠️ Important: INSIDE AudioLDMFineTune folder
```
cd AudioLDMFineTune

# Create conda environment
conda create -n audioldm_train python=3.10
conda activate audioldm_train

# Clone the repo
git clone https://github.com/haoheliu/AudioLDM-training-finetuning.git; cd AudioLDM-training-finetuning

# Install running environment
pip install poetry
poetry install
```
### 1.2.0 Install Audiocraft (MusicGen) Environment
⚠️ Important: Create a separate conda env so it does not conflict with AudioLDM.\
Use this as reference: https://github.com/facebookresearch/audiocraft
⚠️ Important: INSIDE AutoMixMusicGen folder
```
cd AutoMixMusicGen

conda create -n audioldm python=3.9
conda activate automix

python -m pip install 'torch==2.1.0'

# You might need the following before trying to install the packages
python -m pip install setuptools wheel

# Install MusicGen / Audiocraft
python -m pip install -U audiocraft  # stable release

# Install ffmpeg for audio I/O
conda install "ffmpeg<5" -c conda-forge
```
### 1.2.1 Patch MusicGen Transformer (Required for Bidirectional Generation)
The default MusicGen transformer only supports forward generation. To generate transitions that blend both previous and upcoming audio (bidirectional context), AutoMix requires a modified transformer.py download [here](https://1drv.ms/u/c/c9df01ff3f99aa39/IQBG0rS4Cp7oSqiwjKQz83ajAY6pUejcgCOa8wiYbtYTc0Y?e=ZqdjYq)

Locate Audiocraft instalation path and replace the transformer file, example path:
```
/home/user/miniconda3/envs/audioldm/lib/python3.9/site-packages/audiocraft/modules/transformer.py
```
⚠️ Ensure you replace the file in the automix environment, not the audioldm environment.

## 🎵 2. Dataset Structure
dataset/
│
├── chunks/ # 10-second WAV files, 32 kHz
│
└── raw/    # Full-length tracks

## 🎛️ 3. Running
To run finetuning for musicgen run AutoMixMusicGen/musicgen.ipynb\
For audioLDM read and run AudioLDMFineTune/main.ipynb

## ✨ Credits
- AudioLDM — https://github.com/haoheliu/AudioLDM
- MusicGen / Audiocraft — https://github.com/facebookresearch/audiocraft


