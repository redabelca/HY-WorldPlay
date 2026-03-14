import torch
import numpy as np
from PIL import Image
from typing import Union, List, Optional, Dict, Any
import copy
from tqdm import tqdm

from hyvideo.commons import auto_offload_model
from hyvideo.utils.retrieval_context import select_aligned_memory_frames
from hyvideo.pipelines.worldplay_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.pipelines.pipeline_utils import retrieve_timesteps

class StreamingWorldPlayPipeline(HunyuanVideo_1_5_Pipeline):
    """
    A custom wrapper around HunyuanVideo_1_5_Pipeline that breaks down the __call__ 
    method to allow streaming generation, chunk by chunk, keeping state in a dictionary
    to easily support multiple projects.
    """

    def init_stream(
        self,
        prompt: str,
        image_path: str,
        aspect_ratio: str = "16:9",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int = 123,
        user_height: int = 480,
        user_width: int = 832,
        chunk_latent_frames: int = 4, # 4 latents = 16 frames
        device: str = "cuda",
    ):
        """
        Runs the heavy encoders (Qwen, ByT5, SigLIP, VAE) once and sets up the initial empty state dict.
        """
        self.chunk_latent_frames = chunk_latent_frames
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.execution_device = device
        self.target_dtype = self.transformer.dtype
        self.autocast_enabled = True
        
        # Determine target resolution
        target_resolution = self.config.get("base_resolution", "480p")
        
        # Load image
        reference_image = Image.open(image_path).convert("RGB")
        task_type = "i2v"
        semantic_images_np = np.array(reference_image)

        # Setup Scheduler
        flow_shift = self.config.flow_shift
        self.scheduler = self._create_scheduler(flow_shift)
        generator = torch.Generator(device=self.execution_device).manual_seed(seed)

        height, width = user_height, user_width
        
        # Pre-compute text embeds
        with auto_offload_model(self.text_encoder, self.execution_device, enabled=self.enable_offloading):
            (
                prompt_embeds,
                negative_prompt_embeds,
                prompt_mask,
                negative_prompt_mask,
            ) = self.encode_prompt(
                prompt,
                device,
                1, # num_videos_per_prompt
                self.do_classifier_free_guidance,
                negative_prompt="",
                clip_skip=self.clip_skip,
                data_type="video",
            )
        
        extra_kwargs = {}
        if self.config.glyph_byT5_v2:
            with auto_offload_model(self.byt5_model, self.execution_device, enabled=self.enable_offloading):
                extra_kwargs = self._prepare_byt5_embeddings(prompt, device)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])

        # Get visual embeds
        dummy_latents = torch.zeros((1, 1, 1, height // 16, width // 16), device=device) # dummy shape for prepare
        with auto_offload_model(self.vision_encoder, self.execution_device, enabled=self.enable_offloading):
            vision_states = self._prepare_vision_states(
                semantic_images_np, target_resolution, dummy_latents, device
            )
            
        with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
            image_cond = self.get_image_condition_latents(
                task_type, reference_image, height, width
            )

        # Init Caches
        _kv_cache = []
        _kv_cache_neg = []
        transformer_num_layers = len(self.transformer.double_blocks)
        for i in range(transformer_num_layers):
            _kv_cache.append({"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None})
            _kv_cache_neg.append({"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None})
        
        state = {
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "extra_kwargs": extra_kwargs,
            "vision_states": vision_states,
            "image_cond": image_cond,
            "state_latents": None,
            "state_cond_latents": None,
            "state_viewmats": None,
            "state_Ks": None,
            "state_action": None,
            "_kv_cache": _kv_cache,
            "_kv_cache_neg": _kv_cache_neg,
            "generator": generator,
            "task_type": task_type,
            "height": height,
            "width": width
        }
        
        return state

    def step_stream(self, state: dict, new_viewmats, new_Ks, new_action, num_chunks=1):
        """
        Appends new poses and generates exactly 'num_chunks' of video using the provided state dict.
        Returns the new video frames and the updated state dict.
        """
        device = self.execution_device
        task_type = state["task_type"]
        generator = state["generator"]
        
        # Prepare timesteps
        extra_set_timesteps_kwargs = {"n_tokens": 1000} # dummy n_tokens
        timesteps, _ = retrieve_timesteps(
            self.scheduler, self.num_inference_steps, device, **extra_set_timesteps_kwargs
        )
        
        batch_size = 1
        num_channels_latents = self.transformer.config.in_channels
        
        _, latent_height, latent_width = self.get_latent_size(16, state["image_cond"].shape[-2]*16, state["image_cond"].shape[-1]*16)
        
        # 1. Expand our state tensors by 'num_chunks'
        frames_to_add = num_chunks * self.chunk_latent_frames
        
        new_latents = self.prepare_latents(
            batch_size, num_channels_latents, latent_height, latent_width, frames_to_add, 
            self.target_dtype, device, generator
        )
        
        # Ensure image_cond time dimension matches frames_to_add
        expanded_image_cond = state["image_cond"].repeat(1, 1, frames_to_add, 1, 1)
        multitask_mask = self.get_task_mask(task_type, frames_to_add)
        
        new_cond_latents = self._prepare_cond_latents(task_type, expanded_image_cond, new_latents, multitask_mask)

        # Append to state
        if state["state_latents"] is None:
            state["state_latents"] = new_latents
            state["state_cond_latents"] = new_cond_latents
            state["state_viewmats"] = new_viewmats.to(device)
            state["state_Ks"] = new_Ks.to(device)
            state["state_action"] = new_action.to(device)
            
            start_chunk_idx = 0
            
            # Initial KV Cache Setup (Chunk 0 logic from ar_rollout)
            positive_idx = 1 if self.do_classifier_free_guidance else 0
            with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled), auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
                extra_kwargs_pos = {
                    "byt5_text_states": state["extra_kwargs"]["byt5_text_states"][positive_idx, None, ...],
                    "byt5_text_mask": state["extra_kwargs"]["byt5_text_mask"][positive_idx, None, ...],
                }
                t_expand_txt = torch.tensor([0]).to(device).to(self.target_dtype)
                state["_kv_cache"] = self.transformer(
                    bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                    timestep_txt=t_expand_txt, text_states=state["prompt_embeds"][positive_idx, None, ...],
                    encoder_attention_mask=state["prompt_mask"][positive_idx, None, ...],
                    vision_states=state["vision_states"][positive_idx, None, ...], mask_type=task_type,
                    extra_kwargs=extra_kwargs_pos, kv_cache=state["_kv_cache"], cache_txt=True,
                )
                if self.do_classifier_free_guidance:
                    extra_kwargs_neg = {
                        "byt5_text_states": state["extra_kwargs"]["byt5_text_states"][0, None, ...],
                        "byt5_text_mask": state["extra_kwargs"]["byt5_text_mask"][0, None, ...],
                    }
                    state["_kv_cache_neg"] = self.transformer(
                        bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
                        timestep_txt=t_expand_txt, text_states=state["prompt_embeds"][0, None, ...],
                        encoder_attention_mask=state["prompt_mask"][0, None, ...],
                        vision_states=state["vision_states"][0, None, ...], mask_type=task_type,
                        extra_kwargs=extra_kwargs_neg, kv_cache=state["_kv_cache_neg"], cache_txt=True,
                    )
        else:
            start_chunk_idx = state["state_latents"].shape[2] // self.chunk_latent_frames
            state["state_latents"] = torch.cat([state["state_latents"], new_latents], dim=2)
            state["state_cond_latents"] = torch.cat([state["state_cond_latents"], new_cond_latents], dim=2)
            state["state_viewmats"] = torch.cat([state["state_viewmats"], new_viewmats.to(device)], dim=1)
            state["state_Ks"] = torch.cat([state["state_Ks"], new_Ks.to(device)], dim=1)
            state["state_action"] = torch.cat([state["state_action"], new_action.to(device)], dim=1)

        # 2. Run the chunk generation loop
        stabilization_level = 15
        
        for c in range(num_chunks):
            chunk_i = start_chunk_idx + c
            current_frame_idx = chunk_i * self.chunk_latent_frames
            
            selected_frame_indices = []
            if chunk_i > 0:
                for chunk_start_idx in range(current_frame_idx, current_frame_idx + self.chunk_latent_frames, 4):
                    selected_history_frame_id = select_aligned_memory_frames(
                        state["state_viewmats"][0].cpu().detach().numpy(),
                        chunk_start_idx,
                        memory_frames=20,
                        temporal_context_size=12,
                        pred_latent_size=4,
                        points_local=self.points_local,
                        device=device,
                    )
                    selected_frame_indices += selected_history_frame_id
                selected_frame_indices = sorted(list(set(selected_frame_indices)))
                to_remove = list(range(current_frame_idx, current_frame_idx + self.chunk_latent_frames))
                selected_frame_indices = [x for x in selected_frame_indices if x not in to_remove]

                context_latents = state["state_latents"][:, :, selected_frame_indices]
                context_cond_latents_input = state["state_cond_latents"][:, :, selected_frame_indices]
                context_latents_input = torch.concat([context_latents, context_cond_latents_input], dim=1)

                context_viewmats = state["state_viewmats"][:, selected_frame_indices]
                context_Ks = state["state_Ks"][:, selected_frame_indices]
                context_action = state["state_action"][:, selected_frame_indices]

                context_timestep = torch.full((len(selected_frame_indices),), stabilization_level - 1, device=device, dtype=timesteps.dtype)
                
                with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled), auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
                    state["_kv_cache"] = self.transformer(
                        bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                        hidden_states=context_latents_input, timestep=context_timestep, timestep_r=None,
                        mask_type=task_type, return_dict=False, viewmats=context_viewmats.to(self.target_dtype),
                        Ks=context_Ks.to(self.target_dtype), action=context_action.to(self.target_dtype),
                        kv_cache=state["_kv_cache"], cache_vision=True, rope_temporal_size=context_latents_input.shape[2],
                        start_rope_start_idx=0,
                    )
                    if self.do_classifier_free_guidance:
                        state["_kv_cache_neg"] = self.transformer(
                            bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                            hidden_states=context_latents_input, timestep=context_timestep, timestep_r=None,
                            mask_type=task_type, return_dict=False, viewmats=context_viewmats.to(self.target_dtype),
                            Ks=context_Ks.to(self.target_dtype), action=context_action.to(self.target_dtype),
                            kv_cache=state["_kv_cache_neg"], cache_vision=True, rope_temporal_size=context_latents_input.shape[2],
                            start_rope_start_idx=0,
                        )

            self.scheduler.set_timesteps(self.num_inference_steps, device=device)

            start_idx = current_frame_idx
            end_idx = current_frame_idx + self.chunk_latent_frames

            with auto_offload_model(self.transformer, self.execution_device, enabled=self.enable_offloading):
                for i, t in enumerate(tqdm(timesteps, desc=f"Denoising chunk {chunk_i}")):
                    timestep_input = torch.full((self.chunk_latent_frames,), t, device=device, dtype=timesteps.dtype)
                    
                    latent_model_input = state["state_latents"][:, :, start_idx:end_idx]
                    cond_latents_input = state["state_cond_latents"][:, :, start_idx:end_idx]

                    viewmats_input = state["state_viewmats"][:, start_idx:end_idx]
                    Ks_input = state["state_Ks"][:, start_idx:end_idx]
                    action_input = state["state_action"][:, start_idx:end_idx]

                    latents_concat = torch.concat([latent_model_input, cond_latents_input], dim=1)
                    latents_concat = self.scheduler.scale_model_input(latents_concat, t)

                    with torch.autocast(device_type="cuda", dtype=self.target_dtype, enabled=self.autocast_enabled):
                        noise_pred = self.transformer(
                            bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                            hidden_states=latents_concat, timestep=timestep_input, timestep_r=None,
                            mask_type=task_type, return_dict=False, viewmats=viewmats_input.to(self.target_dtype),
                            Ks=Ks_input.to(self.target_dtype), action=action_input.to(self.target_dtype),
                            kv_cache=state["_kv_cache"], cache_vision=False,
                            rope_temporal_size=latents_concat.shape[2] + len(selected_frame_indices),
                            start_rope_start_idx=len(selected_frame_indices),
                        )[0]
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond = self.transformer(
                                bi_inference=False, ar_txt_inference=False, ar_vision_inference=True,
                                hidden_states=latents_concat, timestep=timestep_input, timestep_r=None,
                                mask_type=task_type, return_dict=False, viewmats=viewmats_input.to(self.target_dtype),
                                Ks=Ks_input.to(self.target_dtype), action=action_input.to(self.target_dtype),
                                kv_cache=state["_kv_cache_neg"], cache_vision=False,
                                rope_temporal_size=latents_concat.shape[2] + len(selected_frame_indices),
                                start_rope_start_idx=len(selected_frame_indices),
                            )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)

                    latent_model_input = self.scheduler.step(noise_pred, t, latent_model_input, return_dict=False)[0]
                    state["state_latents"][:, :, start_idx:end_idx] = latent_model_input[:, :, -self.chunk_latent_frames:]

        # Decode newly generated frames only
        newly_generated_latents = state["state_latents"][:, :, -frames_to_add:]
        
        # VAE decoding
        with auto_offload_model(self.vae, self.execution_device, enabled=self.enable_offloading):
            decode_latents = newly_generated_latents
            if len(decode_latents.shape) == 4:
                decode_latents = decode_latents.unsqueeze(2)
            decode_latents = decode_latents / self.vae.config.scaling_factor
            
            # temporal decoding optimization
            decode_latents = decode_latents.to(self.vae.dtype)
            videos = self.vae.decode(decode_latents, return_dict=False)[0]
            
            videos = (videos / 2 + 0.5).clamp(0, 1)
            videos = videos.cpu().float().numpy()
            
        return videos, state
