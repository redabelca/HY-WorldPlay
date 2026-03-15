# 🎮 HY-WorldPlay: Interactive Streaming API Guide

This guide provides everything you need to know about the custom Streaming API developed for **HY-World 1.5 (WorldPlay)**. This architecture allows you to explore generative 3D worlds in real-time using a "game-like" explore-on-demand interface.

---

## 🏗️ 1. Architecture: How it Works

Unlike standard video generation, which creates a fixed-length clip and then "forgets" the state, this API uses **Reconstituted Context Memory**.

### **The "Memory" State**
A project's state is preserved in the GPU's VRAM (and backed up to disk) as a dictionary containing:
*   **Latents History:** The raw visual data of every chunk generated so far.
*   **Pose History:** The exact camera trajectory (viewmats, Ks, and actions).
*   **Embeddings:** Pre-computed text and vision features (so they only run once).

When you call `/step`, the model looks at its history, "remembers" the environment, and generates a new **Chunk** of 16 frames (0.66 seconds) that is perfectly consistent with everything you've seen before.

---

## ⚡ 2. Real-Time Performance & Achievement Guide

To achieve the fastest response times, follow this "Recipe for Speed":

### **The Performance Recipe**
| Feature | Setting | Why it matters |
| :--- | :--- | :--- |
| **Model** | **AR Distilled** | Reduces denoising from 50 steps to **4 steps** (12x faster). |
| **Quantization** | **FP8 (8-bit)** | Reduces VRAM usage by ~50%, fitting the model on a single 24GB card. |
| **Attention** | **SageAttention** | Highly optimized kernels for RTX 30/40 series GPUs. |
| **Resolution** | **480p** | Native resolution for interactive exploration; keeps latency low. |
| **SR** | **False** | Upscaling to 720p/1080p is too slow for real-time play. |

### **Hardware Latency Expectations (480p)**
| Setup | Estimated FPS | Latency for 2 Chunks (1.3s) |
| :--- | :--- | :--- |
| **1x RTX 3090** (Distilled + FP8) | 10-15 FPS | **~2.5 - 3.0 Seconds** |
| **1x RTX 4090** (Distilled + FP8) | 18-24 FPS | **~1.4 - 1.8 Seconds** |
| **Multi-GPU** (SP Parallelism) | **24 FPS** | **~1.3 Seconds (True Real-Time)** |

### **How to achieve True 24 FPS (Real-Time)**
To reach the "locked" 24 FPS experience where generation matches video speed perfectly:
1.  **Multiple GPUs:** You need a multi-GPU environment (e.g., 4x or 8x H100/A100).
2.  **Sequence Parallelism:** Start the server with `--sp_size X` where X is the number of GPUs.
3.  **No Offloading:** Set `--offloading false`. This requires ~80GB of VRAM across your cards to keep the full model in memory.
4.  **High-Speed PCIe:** Ensure the cards are connected via NVLink or PCIe Gen4/5 to minimize data transfer overhead.

---

## 📡 3. API Reference

The server runs on **Port 8000** by default.

### **POST `/start`**
Initializes a new project, processes the image/prompt, and generates the first video segment.
*   **Params:** `prompt` (text), `image` (file), `pose_string` (default: `"w-8"`), `num_chunks` (default: `2`).
*   **Returns:** `session_id`, `video_path` (the first 1.3s of video).

### **POST `/step`**
Generates the next segment of the video based on your camera movement.
*   **Payload:**
    ```json
    {
      "session_id": "uuid",
      "pose_string": "w-4, right-4", 
      "num_chunks": 2
    }
    ```
*   **Rule:** `num_chunks * 4` must equal the total duration in `pose_string`.

### **POST `/resume` (Time Travel)**
Loads a previous checkpoint from the disk into VRAM so you can continue from any point in the past.
*   **Payload:** `{ "checkpoint_path": "./outputs/api_sessions/uuid_chunk_X_state.pt" }`

### **POST `/close` (Memory Cleanup)**
Deletes the session from RAM and **forcefully purges VRAM**. Call this when you are finished with a project.
*   **Payload:** `{ "session_id": "uuid" }`

---

## 💾 4. Resumability & Checkpoints

The server automatically saves a **Complete Checkpoint** (`.pt` file) to `./outputs/api_sessions/` after **every single click**.

*   **Format:** `session_id_chunk_X_state.pt`
*   **Why:** If your server crashes or you want to "branch" a world in two different directions, you can use these files to resume from that exact frame.

---

## 🛠️ 5. Setup & Execution

### **Installation**
```bash
pip install fastapi uvicorn python-multipart angelslim==0.2.2
```
*(Ensure SageAttention is also installed as per the main README)*

### **Starting the Server**
```bash
python api_server.py \
  --model_path /path/to/hunyuanvideo_1_5 \
  --action_ckpt /path/to/ar_distilled_action_model/diffusion_pytorch_model.safetensors \
  --use_fp8_gemm true \
  --offloading true
```

---

## 💡 6. Units & Conversion Table

| Unit | Value |
| :--- | :--- |
| **1 Chunk** | 16 Frames |
| **1 Chunk (Time)** | 0.66 Seconds |
| **1 Chunk (Latents)** | 4 Latents |
| **`num_chunks: 1`** | Use `duration=4` in pose string (e.g., `w-4`) |
| **`num_chunks: 2`** | Use `duration=8` in pose string (e.g., `w-8`) |

---

*Documentation updated on March 15, 2026.*
