import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm  # Add tqdm import
import sys
from torchvision.utils import save_image
# Add the root directory to sys.path
from src.flux.cli import main as flux_main  # Adjust path if needed
# from src.flux.cli_kontext import main as flux_main
from src.flux.util import (
    check_onnx_access_for_trt,
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)

# Input files
prompt_file = "path_to_geneval_prompts.txt"
metadata_prompts = "path_to_geneval_evaluation_metadata.jsonl"

# Load prompt
with open(prompt_file, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]
with open(metadata_prompts, "r") as f:
    metadata = [json.loads(line.strip()) for line in f if line.strip()]

print("The total prompts are", len(prompts))
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
images_per_prompt = 4
output_root = "output_path"
torch_device = torch.device(device)


# Parameters to match your flux_main()
model_name = "flux-dev-kontext"
image_width = 1360
image_height = 768
num_steps = 50
guidance = 2.5
guidance_rescale = 0.0
use_fp16 = True
add_sampling_metadata = True
track_usage = False
offload = False
negative_prompt = ""
t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
clip = load_clip(torch_device)
model = load_flow_model(model_name, device="cpu" if offload else torch_device)
ae = load_ae(model_name, device="cpu" if offload else torch_device)

# Select specific prompt indices to generate (update if needed)
selected_indices = list(range(len(prompts)))  # Or set to something like [0, 5, 10]
accept_list = []

# Iterate through selected prompts
for selected_index in tqdm(selected_indices, desc="Prompts"):
    prompt = prompts[selected_index]
    meta = metadata[selected_index]

    # Prepare output directories
    base_dir = Path(output_root) / f"{selected_index:05d}" / "samples"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = Path(output_root) / f"{selected_index:05d}" / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        json.dump(meta, f)

    # Check if all images already exist
    if all((base_dir / f"{j:05d}.jpg").exists() for j in range(images_per_prompt)):
        print(f"Skipping {selected_index}: All images already exist.")
        continue

    print("The step size is", num_steps)

    # Generate images
    for j in tqdm(range(images_per_prompt), desc=f"Prompt {selected_index:05d}", leave=False):
        # Prepare directories
        base_dir = Path(output_root) / f"{selected_index:05d}" / "samples"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Check if image already exists
        # img_path = base_dir / f"{j:05d}.png"

        # Set deterministic seed
        seed = selected_index * 1000 + j
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Call the image generation function
        image, accept = flux_main(
            name=model_name,
            width=image_width,
            height=image_height,
            seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
            prompt=prompt,
            loop=False,
            num_steps=num_steps,
            guidance=guidance,
            output_dir=base_dir,
            model = model,
            ae = ae,
            j=j
        )


        accept_list.append(accept)


print(sum(accept_list)/len(accept_list))
