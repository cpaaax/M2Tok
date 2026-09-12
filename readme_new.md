
# M2Tok: Getting Started

This guide provides instructions for setting up the environment, training the M2Tok action tokenizer, and training the M2Tok-based VLA model.

---

## 1. Environment Setup

Create a Conda environment and install the required dependencies:

```bash  
# Create and activate the conda environment  
conda create -n moeactok python=3.10 -y  
conda activate m2tok  

# Install PyTorch with CUDA 12.4 support
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124  

# Install Flash Attention  
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  

# Install remaining dependencies  
pip install -r requirements.txt  
```

---

## 2. Training M2Tok Action Tokenizer

### Dataset Preparation
We provide the preprocessed dataset for training the action tokenizer. Download it from Hugging Face:
- [RoboTwin_actions_chunk_8.pkl](https://huggingface.co/datasets/cpxu/M2Tok_data/blob/main/RoboTwin_actions_chunk_8.pkl)

Place the downloaded file under your data directory and ensure the `train_data_path` in the config file (`action_tokenizer/pretrain_config_epoch200.yaml`) is set correctly.

### Training
Train the action tokenizer using an 8-GPU distributed setup via `torchrun`:

```bash
cd src/action_tokenizer
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 pretrain_vqvae.py config=pretrain_config_epoch200.yaml
```

---

## 3. Training M2Tok-based VLA

### Dataset Preparation
The training data consists of 12 demonstration tasks collected on the Piper robotic arm in RoboTwin 2.0. Download the dataset from Hugging Face:
- [RoboTwin Piper 12 Tasks Dataset](https://huggingface.co/datasets/cpxu/RoboTwin_piper_12_tasks_data)

### Model Backbones
The VLA model utilizes the following pretrained components:
- **Vision Encoder**: [google/siglip-so400m-patch14-224](https://huggingface.co/google/siglip-so400m-patch14-224)
- **VLA Backbone**: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B)

### Configuration
Before starting the training, update the following paths in `src/vla/configs/training_based_on_qwen2.5_0.5B.yaml`:
- `img_encoder`: Path for the SigLIP vision encoder.
- `vla_model.vlm_model_path`: Path for Qwen2.5-0.5B.
- `action_vae.model_path`: Checkpoint path of your trained M2Tok action tokenizer.
- `data_root_dir`: Directory containing the downloaded RoboTwin trajectories.
- `instruction_path`: Path to task instructions.

### Training Execution
We train the VLA model on 4 $\times$ H800 GPUs using Accelerate with DeepSpeed ZeRO-2:

```bash
cd src/vla
accelerate launch \
  --config_file accelerate_configs/4_gpus_deepspeed_zero2.yaml \
  --main_process_port=8980 \
  vla-scripts/train_mine.py config=configs/training_based_on_qwen2.5_0.5B.yaml
```
```