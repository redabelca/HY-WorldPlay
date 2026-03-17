#!/bin/bash
export T2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export T2V_REWRITE_MODEL_NAME="<your_model_name>"
export I2V_REWRITE_BASE_URL="<your_vllm_server_base_url>"
export I2V_REWRITE_MODEL_NAME="<your_model_name>"
export PYTHONPATH=$PYTHONPATH:.
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- PROFILE MODE ---
# Set MODE to "fast" (distilled) or "quality" (standard).
# This single flag now controls steps, flow_shift, and chooses the correct sub-folders for models.
MODE="fast" 

PROMPT='A paved pathway leads towards a stone arch bridge spanning a calm body of water. Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky. The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone. The overall composition emphasizes the peaceful and harmonious nature of the landscape.'

IMAGE_PATH=./assets/img/test.png
SEED=1
ASPECT_RATIO=16:9
RESOLUTION=480p 
OUTPUT_PATH=./outputs/

# Base directories for the models. 
# The Python script will now automatically find the '480p_i2v' or 'step_distilled' subfolders based on your MODE.
MODEL_BASE_PATH=/root/.cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/snapshots/9b49404b3f5df2a8f0b31df27a0c7ab872e7b038/transformer
ACTION_BASE_PATH=/root/.cache/huggingface/hub/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8

POSE='w-31'                   
NUM_FRAMES=125
WIDTH=832
HEIGHT=480

# Parallel inference GPU count.
N_INFERENCE_GPU=1 

# Features (Optional)
REWRITE=false   
ENABLE_SR=false 

# --- EXECUTION ---
# Note: "$@" at the end allows you to override ANY setting from your terminal (e.g., bash run.sh --seed 999)
torchrun --nproc_per_node=$N_INFERENCE_GPU hyvideo/generate.py  \
  --mode $MODE \
  --prompt "$PROMPT" \
  --image_path $IMAGE_PATH \
  --resolution $RESOLUTION \
  --aspect_ratio $ASPECT_RATIO \
  --video_length $NUM_FRAMES \
  --seed $SEED \
  --rewrite $REWRITE \
  --sr $ENABLE_SR --save_pre_sr_video \
  --pose "$POSE" \
  --output_path $OUTPUT_PATH \
  --model_path $MODEL_BASE_PATH \
  --action_ckpt $ACTION_BASE_PATH \
  --width $WIDTH \
  --height $HEIGHT \
  --model_type 'ar' \
  --offloading true \
  --use_sageattn true \
  --use_vae_parallel true \
  --group_offloading true \
  "$@"
