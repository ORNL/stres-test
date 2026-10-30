import os
import gc
import sys
import json
import time
import shutil
import argparse

import numpy as np

from pathlib import Path
from huggingface_hub import login as hfh_login
from typing import Dict, Any, Callable

from transformers import get_linear_schedule_with_warmup

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

main_script_file_path = Path(__file__)
absolute_main_script_file_path = main_script_file_path.resolve()
module_directory = absolute_main_script_file_path.parent.parent
sys.path.append(str(module_directory))
from usgs_data.text_time_series_dataset import get_text_time_series_hdf5_data_loader
from OpenTSLM.curriculum_learning import CurriculumTrainer
from OpenTSLM.src.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo # from OpenTSLM.src.model.llm.OpenTSLMSP import OpenTSLMSP
from OpenTSLM.src.model_config import (#BATCH_SIZE, 
    EARLY_STOP_PAT, GRAD_CLIP_NORM, LR_ENCODER, LR_PROJECTOR, NUM_EPOCHS, WARMUP_FRAC, WEIGHT_DECAY, TRANSFORMER_INPUT_DIM, #PATCH_SIZE, 
)

TIME_SERIES_SAMPLE_RT = 50#hz, 
INCLUDE_GEN_KNOWLEDGE = True
CONDENSE_TS_DESCRIPTIONS = True

# Global stage configuration - users can modify this to mix and match stages
CURRICULUM_STAGES = [
    "stageSE_seismic_events", #"stage6_seismic_pretrain", #"stage1_mcq", "stage2_captioning", "stage3_cot", "stage4_sleep_cot", "stage5_ecg_cot",
]
CURRICULUM_SEISMIC_DATA_TRAIN_FILE_NAME = "train_data.h5"
CURRICULUM_SEISMIC_DATA_VAL_FILE_NAME = "val_data.h5"
HUGGING_FACE_MODEL_TO_EOS_STR_MAP = {
    'OpenTSLM/llama-3.2-1b-': '<|end_of_text|>', # 'meta-llama/Llama-3.2-1B::<|end_of_text|>': 128001, # 'meta-llama/Llama-3.2-1B::<|begin_of_text|>': 128000, 
    'OpenTSLM/llama-3.2-3b-': '<|end_of_text|>', # 'meta-llama/Llama-3.2-3B::<|end_of_text|>': 128001, # 'meta-llama/Llama-3.2-3B::<|begin_of_text|>': 128000, 
    'OpenTSLM/gemma-3-270m-': '<eos>', # 'google/gemma-3-270m::<eos>': 1, # 'google/gemma-3-270m::<bos>': 2, # 'google/gemma-3-270m::<unk>': 3, # 'google/gemma-3-270m::<start_of_image>': 255999, # 'google/gemma-3-270m::<end_of_image>': 256000, # 'google/gemma-3-270m::<image_soft_token>': 262144, 
    'OpenTSLM/gemma-3-1b-pt-': '<eos>', # 'google/gemma-3-1b-pt::<eos>': 1, # 'google/gemma-3-1b-pt::<bos>': 2, # 'google/gemma-3-1b-pt::<unk>': 3, # 'google/gemma-3-1b-pt::<start_of_image>': 255999, # 'google/gemma-3-1b-pt::<end_of_image>': 256000, # 'google/gemma-3-1b-pt::<image_soft_token>': 262144, 
}
HUGGINGFACE_OPENTSLM_FLAMINGO_CHKPTS = [
    # gemma-3-270m
    "OpenTSLM/gemma-3-270m-ecg-flamingo", 
    "OpenTSLM/gemma-3-270m-sleep-flamingo", 
    "OpenTSLM/gemma-3-270m-har-flamingo", 
    "OpenTSLM/gemma-3-270m-m4-flamingo", 
    "OpenTSLM/gemma-3-270m-tsqa-flamingo", 
    # gemma-3-1b-pt
    "OpenTSLM/gemma-3-1b-pt-ecg-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-sleep-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-har-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-m4-flamingo", 
    "OpenTSLM/gemma-3-1b-pt-tsqa-flamingo", 
    # llama-3.2-1b
    "OpenTSLM/llama-3.2-1b-ecg-flamingo", 
    "OpenTSLM/llama-3.2-1b-sleep-flamingo", 
    "OpenTSLM/llama-3.2-1b-har-flamingo", 
    "OpenTSLM/llama-3.2-1b-m4-flamingo", 
    "OpenTSLM/llama-3.2-1b-tsqa-flamingo", 
    # llama-3.2-3b
    "OpenTSLM/llama-3.2-3b-ecg-flamingo", 
    "OpenTSLM/llama-3.2-3b-sleep-flamingo", 
    "OpenTSLM/llama-3.2-3b-har-flamingo", 
    "OpenTSLM/llama-3.2-3b-m4-flamingo", 
    "OpenTSLM/llama-3.2-3b-tsqa-flamingo", 
]


class SeismicCurriculumTrainer: # inspired by, but does not inherit directly from,  OpenTSLM.curriculum_learning.CurriculumTrainer

    def __init__(
        self, 
        log_file_path: str, 
        model_type: str,
        save_dir: str,
        pretrained_model_checkpoint: str = None, 
        hf_cache_dir: str = None,
        device: str = None,
        gradient_checkpointing: bool = False, 
        llm_id: str = None,
        patch_size: int = None, 
    ):
        """
        Initialize the curriculum trainer.
        Args:
            model_type: Either 'OpenTSLMSP' or 'OpenTSLMFlamingo'
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            gradient_checkpointing: Enable gradient checkpointing #dist_url: URL used to set up distributed training #dist_backend: Distributed backend #local_rank: Local GPU rank
            llm_id: LLM model ID (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')
        """
        with open(log_file_path, "a") as logfile: 
            logfile.write("SeismicCurriculumTrainer.__init__: START \n")
            self.log_file_path = log_file_path
            self.model_type = model_type
            self.device = device if device is not None else self._get_device()
            if self.device == "mps":
                logfile.write("🚨 Warning: Using MPS, might not be fully compatible with the model. Use CUDA for best results.\n")
            self.llm_id = llm_id
            self.patch_size = patch_size

            # Distributed training parameters
            self.gradient_checkpointing = gradient_checkpointing

            self.pretrained_model_checkpoint = pretrained_model_checkpoint
            self.hf_cache_dir = hf_cache_dir
            self.model = self._initialize_model(log_file_path, pretrained_model_checkpoint) 
            if patch_size is not None and self.model.model.vision_encoder.patch_size < patch_size:
                logfile.write("SeismicCurriculumTrainer.__init__: 🚨 Warning! OVERWRITING PATCH EMBEDDING LAYER PARAMETERS.\n")
                self.model.model.vision_encoder.patch_size = patch_size
                # 1) Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L/patch_size)
                self.model.model.vision_encoder.patch_embed = torch.nn.Conv1d(
                    in_channels=1,
                    out_channels=TRANSFORMER_INPUT_DIM,
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=False,
                ).to(self.device)
            self.results_dir = save_dir
            self._setup_results_dir()
            logfile.write("SeismicCurriculumTrainer.__init__: STOP \n")
        pass


    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _initialize_model(self, log_file_path, chkpt_path):
        """Initialize the specified model type."""
        from OpenTSLM.src.model.llm.OpenTSLM import OpenTSLM as OpenTSLM_ModelClass
        model = OpenTSLM_ModelClass.load_pretrained(chkpt_path, cache_dir=self.hf_cache_dir)
        model = model.to(self.device)
        return model

    def _setup_results_dir(self):
        """Create the results directory structure."""
        os.makedirs(self.results_dir, exist_ok=True)
        # model_dir used to include llm_id_safe
        model_dir = self.results_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.join(model_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(model_dir, "results"), exist_ok=True)
        pass

    def _get_optimizer(self, batch_size: int = None, lr_encoder: float = None, lr_projector: float = None, lr_base: float = None,):
        """Get optimizer for the model with configurable learning rates."""
        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()
        if self.model_type == "OpenTSLMSP":
            raise ValueError("TSLMSP architecture not currently supported due to size of seismic data")
        # For Flamingo, use grouped parameters
        params_to_optimize = model.named_parameters()
        params_to_optimize = list(
            filter(lambda x: x[1].requires_grad and not getattr(x[1], "exclude_from_optimizer", False), params_to_optimize,))
        # Group parameters for weight decay
        params_with_wd, params_without_wd = [], []
        for n, p in params_to_optimize:
            if "gated_cross_attn" in n:
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)
        # Use provided base learning rate or default
        base_lr = lr_base if lr_base is not None else 2e-4
        with open(self.log_file_path, "a") as logfile: 
            logfile.write("📊 Learning rate for "+str(self.model_type)+": \n")
            logfile.write("   Base LR: "+f"{base_lr:.2e}"+" \n")
        return torch.optim.AdamW([{"params": params_with_wd, "weight_decay": 0.1}, {"params": params_without_wd, "weight_decay": 0.0},], lr=base_lr,)

    def _save_checkpoint(self, stage: str, epoch: int, val_loss: float, optimizer, scheduler):
        """Save model checkpoint for a specific stage."""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()
        if self.model_type == "OpenTSLMSP":
            raise ValueError("TSLMSP architecture not currently supported due to size of seismic data")
        else:
            # Handle DDP or single GPU case for OpenTSLMFlamingo
            model_state = model.state_dict()
            if hasattr(self.model, "module"):
                # Remove 'module.' prefix for DDP
                model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
            checkpoint = {
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        # Check disk space before saving
        total, used, free = shutil.disk_usage(checkpoint_dir)
        free_gb = free / (1024**3)
        with open(self.log_file_path, "a") as logfile: 
            logfile.write("💾 Disk space: "+f"{free_gb:.2f}"+" GB free in "+str(checkpoint_dir)+" \n")
            # Estimate checkpoint size (rough estimate)
            estimated_size_gb = sum(p.numel() * p.element_size() for p in self._get_model().parameters()) / (1024**3)
            if (free_gb < estimated_size_gb * 2):  # Need at least 2x the size for safe writing
                logfile.write("⚠️  Warning: Low disk space. Need ~"+f"{estimated_size_gb:.2f}"+" GB, have "+f"{free_gb:.2f}"+" GB free \n")
            # Try to save with error handling
            try:
                torch.save(checkpoint, checkpoint_path)
            except Exception as e:
                logfile.write("❌ Failed to save checkpoint: "+str(e)+" \n")
                logfile.write("   Checkpoint path: "+str(checkpoint_path)+" \n")
                logfile.write("   Checkpoint size: "+f"{sum(p.numel() * p.element_size() for p in self._get_model().parameters()) / 1024**3:.2f}"+" GB \n")
                raise RuntimeError(f"Failed to save checkpoint: {e}")
        return checkpoint_dir


    def _save_loss_history(self, stage: str, epoch: int, train_loss: float, val_loss: float):
        """Save loss history to a file for tracking training progress."""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")
        # Ensure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Create the file with header if it doesn't exist
        if not os.path.exists(loss_history_file):
            with open(loss_history_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")
        # Append the current epoch's losses
        with open(loss_history_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")
        pass

    def _display_loss_history(self, stage: str):
        """Display the loss history for a stage if available."""
        checkpoint_dir = os.path.join(self.results_dir, "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")
        with open(self.log_file_path, "a") as logfile: 
            if os.path.exists(loss_history_file):
                try:
                    with open(loss_history_file, "r") as f:
                        lines = f.readlines()
                    if len(lines) > 2:  # More than just header
                        logfile.write("📊 Previous loss history for "+str(stage)+": \n")
                        logfile.write("   Epoch\tTrain_Loss\tVal_Loss \n")
                        logfile.write("   ------------------------------ \n")
                        # Show last 5 epochs (or all if less than 5)
                        start_idx = max(2, len(lines) - 5)  # Skip header lines
                        for line in lines[start_idx:]:
                            if line.strip() and not line.startswith("-"):
                                parts = line.strip().split("\t")
                                if len(parts) == 3:
                                    epoch, train_loss, val_loss = parts
                                    logfile.write("   "+str(epoch)+"\t"+str(train_loss)+"\t"+str(val_loss)+" \n")
                        if len(lines) > 7:  # More than 5 epochs
                            logfile.write("   ... and "+f"{len(lines) - 7}"+" more epochs \n")
                        logfile.write(" \n")
                except Exception as e:
                    logfile.write("⚠️  Could not read loss history: "+str(e)+" \n")
        pass


    def _load_checkpoint(self, stage: str, optimizer, scheduler, eval_only: bool = False):
        """Load model checkpoint for a specific stage."""
        try:
            checkpoint_path = os.path.join(self.results_dir, stage, "checkpoints", "best_model.pt")
        except:
            checkpoint_path = os.path.join(self.results_dir, "checkpoints", "best_model.pt")

        if os.path.exists(checkpoint_path):
            # Always load checkpoint to CPU first to avoid GPU OOM spikes
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()
            
            if self.model_type == "OpenTSLMSP":
                raise ValueError("TSLMSP architecture not currently supported due to size of seismic data")
            else:
                # Handle DDP or single GPU case for OpenTSLMFlamingo
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    # Add 'module.' prefix for DDP
                    model_state = {f"module.{k}": v for k, v in model_state.items()}
                
                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_state, strict=False)
                    with open(self.log_file_path, "a") as logfile: 
                        if missing_keys:
                            logfile.write("⚠️  Warning: Missing keys when loading checkpoint for "+str(stage)+": \n")
                            for key in missing_keys[:10]:  # Show first 10 missing keys
                                logfile.write("   - "+str(key)+" \n")
                            if len(missing_keys) > 10:
                                logfile.write("   ... and "+str(len(missing_keys) - 10)+" more keys \n")
                        if unexpected_keys:
                            logfile.write("⚠️  Warning: Unexpected keys when loading checkpoint for "+str(stage)+": \n")
                            for key in unexpected_keys[:10]:  # Show first 10 unexpected keys
                                logfile.write("   - "+str(key)+" \n")
                            if len(unexpected_keys) > 10:
                                logfile.write("   ... and "+str(len(unexpected_keys) - 10)+" more keys \n")
                except Exception as e:
                    raise RuntimeError(f"Failed to load model state from checkpoint for {stage}: {e}")
                
                # Only load optimizer state when training
                if (not eval_only and optimizer is not None and "optimizer_state" in checkpoint):
                    optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Only load scheduler state when training
            if (not eval_only and scheduler is not None and "scheduler_state" in checkpoint):
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            return checkpoint.get("epoch", "?"), checkpoint.get("val_loss", float("inf"))

        return None, float("inf")


    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _checkpoint_exists(self, stage: str) -> bool:
        """Check if a checkpoint exists for a specific stage."""
        checkpoint_path = os.path.join(self.results_dir, "checkpoints", "best_model.pt")#stage, "checkpoints", "best_model.pt")
        return os.path.exists(checkpoint_path)

    def _enable_lora_if_needed(self, stage_name: str):
        """Enable LoRA for OpenTSLMSP models in stages after stage2."""
        if self.model_type != "OpenTSLMSP":
            open(self.log_file_path, "a").write("ℹ️  LoRA not configured for model_type "+str(self.model_type)+" \n")#print(f"ℹ️  LoRA not configured for model_type {self.model_type}")
            return  # LoRA only for OpenTSLMSP
        raise ValueError("TSLMSP architecture not currently supported due to size of seismic data")
        pass


    def _train_stage(
        self, stage_name: str, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader = None, 
        num_epochs: int = 30, lr_encoder: float = 1e-4, lr_projector: float = 1e-4, lr_base: float = 1e-4, 
        metric_func: Callable = None, batch_size: int = None, eval_only: bool = False, sampler=None, num_epochs_betw_each_val_run=2, 
    ):# -> Dict[str, Any]:
        """Generic training function for any stage."""
        checkpoint_dir = None
        epoch = None
        if batch_size is None: # Use provided batch_size or default to global BATCH_SIZE
            batch_size = BATCH_SIZE
        
        with open(self.log_file_path, "a") as logfile: 
            logfile.write("\n🚀 Starting "+str(stage_name)+" Training with "+str(self.model_type)+" \n")
            if eval_only:
                logfile.write("🔍 EVAL-ONLY MODE: Skipping training, only running evaluation \n")
            logfile.write("============================================================ \n")
            logfile.write("📊 Stage Configuration: \n")
            logfile.write("   Epochs: "+str(num_epochs)+" \n")
            if self.model_type == "OpenTSLMSP":
                raise ValueError("TSLMSP architecture not currently supported due to size of seismic data")
            else:
                logfile.write("   Base LR: "+f"{lr_base:.2e}"+" \n")
            logfile.write("   Batch size per GPU: "+str(batch_size)+" \n")
            logfile.write(" \n")

            # Check if checkpoint exists when in eval_only mode
            if eval_only and not self._checkpoint_exists(stage_name):
                raise RuntimeError(f"Eval-only mode requires a checkpoint for {stage_name}, but none found at {os.path.join(self.results_dir, 'checkpoints', 'best_model.pt')}")

            # Enable LoRA if needed for this stage
            self._enable_lora_if_needed(stage_name)
            # Initialize optimizer, scaler, and scheduler
            optimizer = self._get_optimizer(batch_size, lr_encoder, lr_projector, lr_base)
            scaler = GradScaler()
            total_steps = num_epochs * len(train_loader)
            warmup_steps = int(WARMUP_FRAC * total_steps)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,)
            # report training config details
            logfile.write("📈 Total training steps: "+str(total_steps)+" \n")
            logfile.write("🔥 Warmup steps: "+str(warmup_steps)+" \n")

            # Load previous checkpoint if exists (for resuming current stage)
            best_epoch, best_val_loss = self._load_checkpoint(stage_name, optimizer, scheduler, eval_only=eval_only)
            if best_epoch is not None:
                logfile.write("📂 Resuming "+str(stage_name)+" from epoch "+str(best_epoch)+" (val_loss: "+f"{best_val_loss:.4f}"+") \n")
                # Display previous loss history if available
                self._display_loss_history(stage_name)
            else:
                logfile.write("🆕 Starting fresh training for "+str(stage_name)+" \n")
                best_val_loss = float("inf")  # Ensure proper initialization

            # Skip training loop if eval_only is True
            if eval_only:
                logfile.write("⏭️  Skipping training loop (eval_only mode) \n")
                logfile.write("📂 Using existing checkpoint for evaluation \n") 
                epoch = best_epoch
                epochs_no_improve = 0
            else:
                # Training loop
                num_train_batches = len(train_loader)
                epochs_no_improve = 0
                start_epoch = best_epoch + 1 if best_epoch is not None else 1
                val_loss_from_prior_epochs = None
                for epoch in range(start_epoch, num_epochs + 1):
                    logfile.write("_train_stage."+str(stage_name)+": epoch "+str(epoch)+" of "+str(num_epochs)+" \n")
                    # Set epoch for distributed sampler
                    if hasattr(train_loader.sampler, "set_epoch"):
                        train_loader.sampler.set_epoch(epoch)
                    
                    # Training
                    self.model.train()
                    running_loss = 0.0
                    for i, batch in enumerate(train_loader):#prog):
                        if i % (num_train_batches//50) == 0:
                            logfile.write("_train_stage."+str(stage_name)+": \t train batch "+str(i)+" of "+str(num_train_batches)+" \n")
                        # DEBUG PRINT: Only for the first batch of the first epoch
                        if epoch == start_epoch and i == 0:
                            logfile.write("[DEBUG] Batch "+str(i)+" - batch == "+str(batch)+" \n")
                            logfile.write("[DEBUG] Batch "+str(i)+" - batch size: "+str(len(batch))+" \n")
                            if isinstance(batch, list) and isinstance(batch[0], dict):
                                for k, v in batch[0].items():
                                    if hasattr(v, "shape"):
                                        logfile.write("[DEBUG] Sample key '"+str(k)+"' shape: "+str(v.shape)+" \n")
                                    elif isinstance(v, list):
                                        logfile.write("[DEBUG] Sample key '"+str(k)+"' list length: "+str(len(v))+" \n")
                            logfile.write(torch.cuda.memory_summary()+" \n" if torch.cuda.is_available() else "No CUDA \n")
                        # reset optimizer and compute loss and new gradients
                        optimizer.zero_grad()
                        # Mixed-precision: Forward pass under autocast
                        with autocast(): 
                            loss = self._get_model().compute_loss(batch)
                        # Scale loss before backward pass
                        scaler.scale(loss).backward()
                        # Handle gradient clipping for distributed training
                        scaler.unscale_(optimizer)  # Unscales gradients
                        clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)
                        # Step optimizer and scheduler with scaled gradients
                        scaler.step(optimizer)#optimizer.step()
                        scaler.update()
                        scheduler.step()
                        running_loss += loss.item()
                        if not np.isfinite(running_loss):
                            logfile.write("_train_stage."+str(stage_name)+": \t NON-FINITE running_loss VALUE ENCOUNTERED - KILLING JOB TO SAVE COMPUTE \n")
                            raise ValueError("running_loss == "+str(running_loss))
                        # if i == 10: # FOR DEBUGGING PURPOSES
                        #     logfile.write("_train_stage."+str(stage_name)+": \t BREAKING FOR DEBUGGING PURPOSES \n") # FOR DEBUGGING PURPOSES
                        #     break # FOR DEBUGGING PURPOSES
                    avg_train_loss = running_loss / len(train_loader)
                    logfile.write("_train_stage."+str(stage_name)+": - Epoch "+str(epoch)+" — train loss: "+f"{avg_train_loss:.4f}"+" \n")

                    # Validation
                    if (epoch-1) % num_epochs_betw_each_val_run == 0:
                        num_val_batches = len(val_loader)
                        val_loss = 0.0
                        self.model.eval()
                        with torch.no_grad():
                            for vi, batch in enumerate(val_loader):#tqdm(val_loader, desc=f"Validating {stage_name}", disable=self.rank != 0,):
                                if vi % (num_val_batches//50) == 0:
                                    logfile.write("_train_stage."+str(stage_name)+": \t val batch "+str(batch)+" of "+str(num_val_batches)+" \n")
                                # Mixed precision context
                                with torch.cuda.amp.autocast():
                                    # Forward pass with mixed precision
                                    val_loss += self._get_model().compute_loss(batch).item()
                                    if not np.isfinite(val_loss):
                                        logfile.write("_train_stage."+str(stage_name)+": \t NON-FINITE val_loss VALUE ENCOUNTERED - KILLING JOB TO SAVE COMPUTE \n")
                                        raise ValueError("val_loss == "+str(val_loss))
                                # if vi == 10: # FOR DEBUGGING PURPOSES
                                #     logfile.write("_train_stage."+str(stage_name)+": \t BREAKING FOR DEBUGGING PURPOSES \n") # FOR DEBUGGING PURPOSES
                                #     break # FOR DEBUGGING PURPOSES
                        avg_val_loss = val_loss / len(val_loader)
                        # Synchronize validation loss across all ranks
                        if dist.is_initialized():
                            val_loss_tensor = torch.tensor(avg_val_loss, device=self.device)
                            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                            avg_val_loss = val_loss_tensor.item()# / self.world_size
                        val_loss_from_prior_epochs = avg_val_loss
                    assert val_loss_from_prior_epochs is not None
                    logfile.write("_train_stage."+str(stage_name)+": - Epoch "+str(epoch)+" — val   loss: "+f"{val_loss_from_prior_epochs:.4f}"+" \n")#avg_val_loss:.4f}"+" \n")
                    logfile.write("_train_stage."+str(stage_name)+": - Epoch "+str(epoch)+" — best  loss: "+f"{val_loss_from_prior_epochs:.4f}"+" \n")#best_val_loss:.4f}"+" \n")
                    # Save loss history for this epoch
                    self._save_loss_history(stage_name, epoch, avg_train_loss, val_loss_from_prior_epochs)#avg_val_loss)

                    # Early stopping - all ranks need to make the same decision
                    # logfile.write("_train_stage."+str(stage_name)+": \t SAVING FOR DEBUGGING PURPOSES \n") # FOR DEBUGGING PURPOSES
                    should_save = False # TRUE FOR DEBUGGING PURPOSES
                    if epoch % num_epochs_betw_each_val_run == 0: # COMMENTED OUT FOR DEBUGGING PURPOSES
                        should_save = val_loss_from_prior_epochs + 1e-4 < best_val_loss #avg_val_loss + 1e-4 < best_val_loss # COMMENTED OUT FOR DEBUGGING PURPOSES
                        if dist.is_initialized(): # COMMENTED OUT FOR DEBUGGING PURPOSES
                            save_tensor = torch.tensor(1 if should_save else 0, device=self.device) # COMMENTED OUT FOR DEBUGGING PURPOSES
                            dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM) # COMMENTED OUT FOR DEBUGGING PURPOSES
                            should_save = (save_tensor.item() > 0)  # If any rank thinks we should save, we save # COMMENTED OUT FOR DEBUGGING PURPOSES
                    if should_save:
                        best_val_loss = val_loss_from_prior_epochs#avg_val_loss
                        epochs_no_improve = 0
                        checkpoint_dir = self._save_checkpoint(stage_name, epoch, val_loss_from_prior_epochs, optimizer, scheduler)#avg_val_loss, optimizer, scheduler)
                        logfile.write("✔️  New best model saved.\n \n")
                    else:
                        epochs_no_improve += 1
                        logfile.write("No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n \n")
                        # Synchronize early stopping decision across all ranks
                        if epochs_no_improve >= EARLY_STOP_PAT:
                            logfile.write("\nEarly stopping triggered after {epoch} epochs. \n")
                            logfile.write("Final stats: best_val_loss="+f"{best_val_loss:.4f}"+", epochs_no_improve="+str(epochs_no_improve)+" \n")
                            break

                    # Synchronize best_val_loss and epochs_no_improve across all ranks
                    if dist.is_initialized():
                        best_loss_tensor = torch.tensor(best_val_loss, device=self.device)
                        epochs_tensor = torch.tensor(epochs_no_improve, device=self.device)
                        dist.broadcast(best_loss_tensor, src=0)
                        dist.broadcast(epochs_tensor, src=0)
                        best_val_loss = best_loss_tensor.item()
                        epochs_no_improve = int(epochs_tensor.item())
                    # if epoch > 4:#start_epoch: # FOR DEBUGGING PURPOSES
                    #     logfile.write("_train_stage."+str(stage_name)+": BREAKING FOR DEBUGGING PURPOSES \n") # FOR DEBUGGING PURPOSES
                    #     break # FOR DEBUGGING PURPOSES

            # Load best model and evaluate
            best_epoch, _ = self._load_checkpoint(stage_name, optimizer, scheduler)
            if best_epoch is not None:
                logfile.write("📂 Loaded best checkpoint from epoch "+str(best_epoch)+" for evaluation. \n")
            if epoch is None:
                epoch = best_epoch
                logfile.write("🏁 Training completed for "+str(stage_name)+" \n")
                logfile.write("   Total epochs run: "+str(epoch)+" \n")
            else:
                logfile.write("🏁 Training completed for "+str(stage_name)+" \n")
                logfile.write("   Total epochs run: "+str(epoch)+" \n")
                logfile.write("   Best validation loss: "+f"{best_val_loss:.4f}"+" \n")
                logfile.write("   Epochs without improvement: "+str(epochs_no_improve)+" \n")
            
            logfile.write("_train_stage."+str(stage_name)+": !!! SKIPPING EVALUATE STAGE TO FREE UP COMPUTE RESOURCES FOR NOW !!! \n")

        return checkpoint_dir


    def stageSE_seismic_events(
        self, train_loader, val_loader, test_loader, batch_size: int = None, eval_only: bool = False, 
        num_epochs: int = 20, lr_encoder: float = 2e-4, lr_projector: float = 1e-4,
        lr_base: float = 2e-4, num_epochs_betw_each_val_run: int = 2
    ):# -> Dict[str, Any]:
        """Stage SE: Seismic Data Event-Informed Prompt Completion.
        Configuration:
        - Epochs: 20
        - OpenTSLMSP: encoder_lr=1e-4, projector_lr=5e-5 (lower for fine-tuning)
        - OpenTSLMFlamingo: base_lr=1e-4 (lower for fine-tuning)
        - Metric: Test loss only
        """
        return self._train_stage(
            "stageSE_seismic_events", # stage_name: 
            train_loader, 
            val_loader, 
            test_loader, 
            num_epochs=num_epochs, #20,
            lr_encoder=lr_encoder, #2e-4,
            lr_projector=lr_projector, #1e-4,
            lr_base=lr_base, #2e-4,
            metric_func=None,  # Only test loss for captioning
            batch_size=batch_size,
            eval_only=eval_only,
            sampler=None,
            num_epochs_betw_each_val_run=num_epochs_betw_each_val_run, 
        )


def run_finetuning(
    curr_finetune_save_dir, finetune_data_dir, pretrain_path, patch_size, EOS_TOKEN, device, batch_size, 
    model_type, gradient_checkpointing, llm_id, eval_only, hf_cache_dir=None, num_epochs=20, lr_encoder=2e-4, 
    lr_projector=1e-4, lr_base=2e-4, num_epochs_betw_each_val_run=2
):
    current_file_path = Path(__file__)
    absolute_current_file_path = current_file_path.resolve()
    module_directory = absolute_current_file_path.parent.parent
    sys.path.append(str(module_directory))
    curr_log_file_path = curr_finetune_save_dir + os.sep + 'logs_from_finetuning.txt'

    with open(curr_log_file_path, "a") as logfile: 
        logfile.write("run_finetuning: START \n")
        logfile.write("run_finetuning: curr_finetune_save_dir == "+str(curr_finetune_save_dir)+" \n")
        # define paths to current data
        finetune_train_data_path = finetune_data_dir + os.sep + CURRICULUM_SEISMIC_DATA_TRAIN_FILE_NAME
        finetune_val_data_path = finetune_data_dir + os.sep + CURRICULUM_SEISMIC_DATA_VAL_FILE_NAME
        # define data loaders corresponding to current data
        logfile.write("run_finetuning: Loading Data \n")

        finetune_train_data_loader, _ = get_text_time_series_hdf5_data_loader(
            finetune_train_data_path, curr_log_file_path, EOS_TOKEN, batch_size=batch_size, patch_size=patch_size, 
            time_series_sample_rt=TIME_SERIES_SAMPLE_RT, include_gen_knowledge=INCLUDE_GEN_KNOWLEDGE, 
            condense_ts_descriptions=CONDENSE_TS_DESCRIPTIONS, 
        )
        finetune_val_data_loader, _ = get_text_time_series_hdf5_data_loader(
            finetune_val_data_path, curr_log_file_path, EOS_TOKEN, batch_size=batch_size, patch_size=patch_size, 
            time_series_sample_rt=TIME_SERIES_SAMPLE_RT, include_gen_knowledge=INCLUDE_GEN_KNOWLEDGE, 
            condense_ts_descriptions=CONDENSE_TS_DESCRIPTIONS, 
        )
        # train ...
        logfile.write("run_finetuning: Defining SeismicCurriculumTrainer \n")
        trainer = SeismicCurriculumTrainer(
            curr_log_file_path, 
            model_type, 
            curr_finetune_save_dir, 
            pretrained_model_checkpoint=pretrain_path, 
            hf_cache_dir=hf_cache_dir,
            device=device, 
            gradient_checkpointing=gradient_checkpointing, 
            llm_id=llm_id, 
            patch_size=patch_size, 
        )
        logfile.write("run_finetuning: running stageSE_seismic_events \n")
        finetune_chkpt_dir = trainer.stageSE_seismic_events(
            finetune_train_data_loader, 
            finetune_val_data_loader, 
            None, # test_data_loader
            batch_size=batch_size, 
            eval_only=eval_only, 
            num_epochs=num_epochs, #20, 
            lr_encoder=lr_encoder, #2e-4, 
            lr_projector=lr_projector, #1e-4,
            lr_base=lr_base, #2e-4, 
            num_epochs_betw_each_val_run=num_epochs_betw_each_val_run, #2
        )
        logfile.write("run_finetuning:  - finetune_chkpt_dir == "+str(finetune_chkpt_dir)+" \n")
        logfile.write("run_finetuning: STOP \n")
    pass



if __name__ == "__main__": 
    # read in arguments ##########################################################################################
    parser = argparse.ArgumentParser(description="USGS Seismic Data Curriculum Learning for OpenTSLM Models.")
    # ARGS SPECIFIC TO Seisimc-Data OpenTSLM TRAINING ---
    parser.add_argument("--data_dir", type=str, required=True, help="Base directory of cached pre-processed data")
    parser.add_argument("--save_dir", type=str, default="None", help="Directory where training outputs should be saved")
    parser.add_argument("--log_file_path", type=str, default="None", help="Path to where logs should be saved")
    parser.add_argument("--patch_size", type=str, default=None, help="Patch size to use prior to time series positional encoding")
    parser.add_argument("--max_num_gpus", type=str, default=1, help="Maximum number of gpus to use (esp. for cross-val training)")
    parser.add_argument("--huggingface_model_ckpt_root", type=str, default="OpenTSLM/gemma-3-270m-", help="Beginning of hugging face model checkpoint path")
    parser.add_argument("--num_epochs", type=int, default=20, help="Maximum number of training epochs",)
    parser.add_argument("--lr_encoder", type=float, default=0.0002, help="Encoder learning rate",)
    parser.add_argument("--lr_projector", type=float, default=0.0001, help="Projector learning rate",)
    parser.add_argument("--lr_base", type=float, default=0.0002, help="Base learning rate",)
    parser.add_argument("--num_epochs_betw_each_val_run", type=int, default=2, help="Number of training epochs before each validation run",)
    parser.add_argument("--hf_cache_dir", type=str, default="~/huggingface_cache", help="Local Hugging Face cache root to load models from")
    parser.add_argument("--hf_offline", default=True, action="store_true", help="Force Hugging Face/Transformers to run in offline mode (use local cache only)",)
    # ARGS FROM OpenTSLM TRAINING SCRIPT ----------------
    parser.add_argument("--model", type=str, choices=["OpenTSLMSP", "OpenTSLMFlamingo"], required=True, help="Model type to train",)
    parser.add_argument("--stages", nargs="+", choices=CURRICULUM_STAGES, default=CURRICULUM_STAGES, help="Stages to run (default: all stages)",)
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training (default: use value from model_config.py)",)#4, help="Batch size for training (default: use value from model_config.py)",)
    parser.add_argument("--eval_only", default=False, action="store_true", help="Skip training and only run evaluation (requires existing checkpoint)",)
    parser.add_argument("--llm_id", type=str, default="google/gemma-3-270m", help="LLM model ID for OpenTSLMFlamingo (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')",)
    parser.add_argument("--verbose", default=False, action="store_true", help="Enable verbose logging")
    # Distributed training arguments
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true", help="Enable gradient checkpointing",)
    args = parser.parse_args()
    log_file_path = args.log_file_path

    with open(log_file_path, "a") as logfile: 
        logfile.write("__MAIN__: START \n")
        task_ID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        time.sleep(task_ID*5) # stagger tasks to avoid race conditions
        data_dir = args.data_dir
        save_dir = args.save_dir
        patch_size = int(args.patch_size)
        max_num_gpus = int(args.max_num_gpus)
        huggingface_model_ckpt_root = args.huggingface_model_ckpt_root
        hf_cache_dir = args.hf_cache_dir
        hf_offline = args.hf_offline

        # set up environment #########################################################################################
        logfile.write("__MAIN__: setting up environment \n")
        # logfile.write("__MAIN__: module_directory == "+str(module_directory)+" \n")
        from OpenTSLM.src.logger import set_global_verbose
        
        # Set up global logging
        set_global_verbose(args.verbose)
        if hf_cache_dir:
            os.environ["HF_HOME"] = hf_cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir
            os.environ["HF_HUB_CACHE"] = hf_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
            logfile.write("__MAIN__: HF cache dir set to "+str(hf_cache_dir)+" \n")
        if hf_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logfile.write("__MAIN__: HF/Transformers offline mode enabled \n")
        # # login to huggingface hub
        # huggingface_hub_login_credential_filepath = str(module_directory)+os.sep+"huggingface_hub_login_credential.txt"
        # login_cred = open(huggingface_hub_login_credential_filepath, 'r').read().strip()
        # hfh_login(login_cred)

        # begin training #############################################################################################
        data_folds = sorted([x for x in os.listdir(data_dir) if "fold_" in x])
        pretrained_huggingface_model_paths = sorted([x for x in HUGGINGFACE_OPENTSLM_FLAMINGO_CHKPTS if huggingface_model_ckpt_root in x])
        data_model_pairings = [[(i,x),(j,y)] for i,x in enumerate(data_folds) for j,y in enumerate(pretrained_huggingface_model_paths)]
        task_ID = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        logfile.write("\n\n__MAIN__: NOW KICKING OFF task_ID == "+str(task_ID)+" @^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@^*^@ \n")
        curr_data_model_pair = data_model_pairings[task_ID-1]
        f_id = curr_data_model_pair[0][0]
        fold_name = curr_data_model_pair[0][1]
        m_id = curr_data_model_pair[1][0]
        model_path = curr_data_model_pair[1][1]

        # for f_id, fold_name in enumerate(data_folds):
        logfile.write("\n\n__MAIN__: NOW TRAINING MODELS ON DATA FROM fold_name == "+str(fold_name)+" ########################################################### \n")
        # update args to reflect current fold
        curr_data_dir = data_dir + os.sep + fold_name

        # finetune model on current fold
        # for m_id, model_path in enumerate(pretrained_huggingface_model_paths):
        logfile.write("\n__MAIN__: NOW TRAINING MODEL FROM model_path == "+str(model_path)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
        EOS_TOKEN = HUGGING_FACE_MODEL_TO_EOS_STR_MAP[huggingface_model_ckpt_root]
        
        # update args to reflect current model
        curr_finetune_save_dir = save_dir + os.sep + model_path.replace('/','_')
        if not os.path.exists(curr_finetune_save_dir):
            os.mkdir(curr_finetune_save_dir)
        curr_finetune_save_dir = curr_finetune_save_dir + os.sep + fold_name
        if not os.path.exists(curr_finetune_save_dir):
            os.mkdir(curr_finetune_save_dir)
        
        # ensure memory - esp. cuda - is clear
        assert torch.cuda.is_available()
        torch.cuda.empty_cache()
        gc.collect()
        
        try: # COMMENTED OUT FOR DEBUGGING PURPOSES
            # update the model
            run_finetuning(
                curr_finetune_save_dir, 
                curr_data_dir, 
                model_path, 
                patch_size, 
                EOS_TOKEN, 
                None, #device, 
                args.batch_size, 
                args.model, 
                args.gradient_checkpointing, 
                args.llm_id, 
                args.eval_only, 
                hf_cache_dir=hf_cache_dir,
                num_epochs=args.num_epochs, #20, 
                lr_encoder=args.lr_encoder, #2e-4, 
                lr_projector=args.lr_projector, #1e-4,
                lr_base=args.lr_base, #2e-4, 
                num_epochs_betw_each_val_run=args.num_epochs_betw_each_val_run, #2
            )
        except Exception as e: # COMMENTED OUT FOR DEBUGGING PURPOSES
            logfile.write("__MAIN__: EXCEPTION ENCOUNTERED DURING FINETUNING ATTEMPT FOR model_path == "+str(model_path)+" \n")
            logfile.write("__MAIN__: \t THE EXCEPTION RETURNED READS AS e == "+str(e)+" \n")
            logfile.write("__MAIN__: \t SKIPPING CURRENT ATTEMPT FOR NOW AND MOVING ON TO ADDITIONAL FOLD+MODEL COMBINATIONS \n")
        
        # clear memory (esp. cuda)
        torch.cuda.empty_cache()
        gc.collect()
        
        # kick off process on available device
        logfile.write("__MAIN__: DONE WITH task_ID == "+str(task_ID)+" !!! \n")
    pass
