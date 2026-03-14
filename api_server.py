import os
import torch
import uuid
import argparse
import gc
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Set rank for single-gpu init requirements
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

from hyvideo.generate import pose_to_input
from hyvideo.commons.infer_state import initialize_infer_state
from hyvideo.utils.data_utils import save_video
from hyvideo.pipelines.streaming_pipeline import StreamingWorldPlayPipeline

app = FastAPI(title="HY-WorldPlay Streaming API")

# Global variables
ACTIVE_PIPELINE = None
SESSIONS = {}
OUTPUT_DIR = "./outputs/api_sessions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class StepRequest(BaseModel):
    session_id: str
    pose_string: str = "w-4" # Default 1 latent chunk
    num_chunks: int = 1 # Number of 4-latent chunks to generate

class CloseRequest(BaseModel):
    session_id: str

class ResumeRequest(BaseModel):
    checkpoint_path: str

def move_to_device(obj, device):
    """Recursively move tensors in state dictionary to the specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj

def save_checkpoint(state, session_id, chunk_index):
    """Saves the complete state to disk for future resumability."""
    # Prepare state for saving (extract generator state)
    save_state = {k: v for k, v in state.items() if k != 'generator'}
    save_state['generator_state'] = state['generator'].get_state()
    
    # Save with chunk index to maintain history
    backup_path = os.path.join(OUTPUT_DIR, f"{session_id}_chunk_{chunk_index}_state.pt")
    
    # Move to CPU before saving to save VRAM and avoid mapping issues later
    cpu_state = move_to_device(save_state, "cpu")
    torch.save(cpu_state, backup_path)
    
    return backup_path

@app.post("/start")
async def start_session(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    pose_string: str = Form("w-8"), # Default generate 2 chunks (1.3 seconds)
    num_chunks: int = Form(2)
):
    global ACTIVE_PIPELINE, SESSIONS
    
    if ACTIVE_PIPELINE is None:
        raise HTTPException(status_code=500, detail="Server model not initialized. Please start the server with the correct model arguments.")
    
    session_id = str(uuid.uuid4())
    
    # Save uploaded image temporarily
    image_path = os.path.join(OUTPUT_DIR, f"{session_id}_init.png")
    with open(image_path, "wb") as f:
        f.write(await image.read())
        
    print(f"Starting new session {session_id} with prompt: {prompt}")
    
    # Initialize the streaming state (runs encoders)
    state = ACTIVE_PIPELINE.init_stream(
        prompt=prompt,
        image_path=image_path,
        aspect_ratio="16:9",
        num_inference_steps=50,
        guidance_scale=1.0,
        seed=123,
        user_height=480,
        user_width=832,
        chunk_latent_frames=4,
        device="cuda"
    )
    
    SESSIONS[session_id] = state
    
    video_path = None
    backup_path = None
    if num_chunks > 0:
        latent_num = num_chunks * ACTIVE_PIPELINE.chunk_latent_frames
        try:
            viewmats, Ks, action = pose_to_input(pose_string, latent_num)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse initial pose string: {e}")
            
        print(f"Generating initial {num_chunks} chunks ({latent_num} latents) for session {session_id}")
        videos_np, updated_state = ACTIVE_PIPELINE.step_stream(
            state=state,
            new_viewmats=viewmats,
            new_Ks=Ks,
            new_action=action,
            num_chunks=num_chunks
        )
        SESSIONS[session_id] = updated_state
        
        chunk_index = updated_state["state_latents"].shape[2] // ACTIVE_PIPELINE.chunk_latent_frames
        
        # Save complete state checkpoint
        backup_path = save_checkpoint(updated_state, session_id, chunk_index)
        
        video_path = os.path.join(OUTPUT_DIR, f"{session_id}_chunk_{chunk_index}.mp4")
        videos_tensor = torch.from_numpy(videos_np).unsqueeze(0)
        save_video(videos_tensor, video_path)
    
    return {
        "status": "success", 
        "session_id": session_id, 
        "video_path": video_path,
        "backup_path": backup_path,
        "message": "Project initialized and first video generated."
    }

@app.post("/step")
async def step_session(req: StepRequest):
    global ACTIVE_PIPELINE, SESSIONS
    
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session ID.")
    
    state = SESSIONS[req.session_id]
    
    if state["state_latents"] is None and req.num_chunks == 0:
        raise HTTPException(status_code=400, detail="State not fully initialized. Generate at least 1 chunk.")

    # Calculate required frames for this step (chunk_latent_frames = 4 by default)
    latent_num = req.num_chunks * ACTIVE_PIPELINE.chunk_latent_frames
    
    # Parse pose to inputs
    try:
        viewmats, Ks, action = pose_to_input(req.pose_string, latent_num)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse pose string. Ensure the pose duration matches the requested chunk length. Expected {latent_num} latents total. Error: {e}")

    print(f"Generating {req.num_chunks} chunks ({latent_num} latents) for session {req.session_id}")
    
    # Run the streaming generation
    videos_np, updated_state = ACTIVE_PIPELINE.step_stream(
        state=state,
        new_viewmats=viewmats,
        new_Ks=Ks,
        new_action=action,
        num_chunks=req.num_chunks
    )
    SESSIONS[req.session_id] = updated_state
    
    chunk_index = updated_state["state_latents"].shape[2] // ACTIVE_PIPELINE.chunk_latent_frames
    
    # Save complete state checkpoint
    backup_path = save_checkpoint(updated_state, req.session_id, chunk_index)
    
    # Save video chunk
    video_path = os.path.join(OUTPUT_DIR, f"{req.session_id}_chunk_{chunk_index}.mp4")
    
    # Format for save_video utility: expects [B, C, T, H, W]
    videos_tensor = torch.from_numpy(videos_np).unsqueeze(0)
    save_video(videos_tensor, video_path)
    
    return {"status": "success", "video_path": video_path, "backup_path": backup_path}

@app.post("/resume")
async def resume_session(req: ResumeRequest):
    """
    Loads a full state from a .pt checkpoint file on disk, creates a new session in VRAM, 
    and allows you to continue generating from that point.
    """
    global ACTIVE_PIPELINE, SESSIONS
    
    if ACTIVE_PIPELINE is None:
        raise HTTPException(status_code=500, detail="Server model not initialized.")
        
    if not os.path.exists(req.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {req.checkpoint_path}")
        
    print(f"Loading checkpoint from: {req.checkpoint_path}")
    
    # Load state from disk to CPU first
    save_state = torch.load(req.checkpoint_path, map_location="cpu")
    
    # Reconstruct generator
    generator = torch.Generator(device=ACTIVE_PIPELINE.execution_device)
    generator.set_state(save_state['generator_state'])
    
    # Move tensors back to GPU execution device
    state = move_to_device(save_state, ACTIVE_PIPELINE.execution_device)
    
    # Replace generator state with actual generator object
    del state['generator_state']
    state['generator'] = generator
                
    # Assign a new active session
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = state
    
    chunk_index = state["state_latents"].shape[2] // ACTIVE_PIPELINE.chunk_latent_frames if state["state_latents"] is not None else 0
    
    print(f"Successfully resumed session {session_id} at chunk {chunk_index}")
    
    return {
        "status": "success",
        "session_id": session_id,
        "message": f"Session resumed successfully. Ready to generate from chunk {chunk_index}.",
        "chunk_index": chunk_index
    }

@app.post("/close")
async def close_session(req: CloseRequest):
    """
    Deletes the project session from memory and calls garbage collection to free up VRAM.
    """
    global SESSIONS
    if req.session_id in SESSIONS:
        del SESSIONS[req.session_id]
        
        # Free memory forcefully
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return {"status": "success", "message": f"Session {req.session_id} closed and memory freed."}
    else:
        raise HTTPException(status_code=404, detail="Session not found.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--action_ckpt", type=str, required=True, help="Path to pretrained action model")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--offloading", action="store_true", default=True)
    args, unknown = parser.parse_known_args()

    # Create dummy args for initialize_infer_state compatibility
    class DummyArgs:
        use_sageattn = False
        sage_blocks_range = ""
        enable_torch_compile = False
        use_fp8_gemm = False
        quant_type = "fp8-per-block"
        include_patterns = "double_blocks"
        use_vae_parallel = False
    
    initialize_infer_state(DummyArgs())

    transformer_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print("Loading Base Pipeline into Memory...")
    global ACTIVE_PIPELINE
    ACTIVE_PIPELINE = StreamingWorldPlayPipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_version="480p_i2v",
        enable_offloading=args.offloading,
        enable_group_offloading=args.offloading,
        create_sr_pipeline=False,
        force_sparse_attn=False,
        transformer_dtype=transformer_dtype,
        action_ckpt=args.action_ckpt,
    )
    print("Pipeline Loaded Successfully!")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
