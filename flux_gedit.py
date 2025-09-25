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
import pickle
# Add the root directory to sys.path
# from src.flux.cli import main as flux_main  # Adjust path if needed
from src.flux.cli_kontext import main as flux_main
from src.flux.util import (
    check_onnx_access_for_trt,
    configs,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
)


from datasets import load_dataset
dataset = load_dataset("stepfun-ai/GEdit-Bench")

filtered_data = [
    entry for entry in dataset["train"]
    if entry.get("instruction_language") == "en"
]


print("Dataset loaded. The length of the dataset is", len(filtered_data))

device = "cuda" if torch.cuda.is_available() else "cpu"
# output_root = "output_path"
torch_device = torch.device(device)

method_name = "flux" 

# Parameters to match your flux_main()
model_name = "flux-dev-kontext"
# image_width = 1360
# image_height = 768
seed = 51
num_steps = 25
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

accept_list = []

# Iterate through selected prompts
for i in range(len(filtered_data)):
    data = filtered_data[i]
    print("The data being processed is", i)

    # Extract info
    image = data["input_image"]
    edit_prompt = data["instruction"]
    key = data["key"]
    task_type = data["task_type"]
    language = data["instruction_language"]
    # Build output path early to check if it already exists
    save_dir = Path("gedit_results_bandit_25_true") / method_name / "fullset" / task_type / "en"
    save_dir.mkdir(parents=True, exist_ok=True)
    img_path = save_dir / f"{key}.png"

    if img_path.exists():
        print(f"Skipping {key} — already exists at {img_path}")
        continue  # Skip already processed image

    if language != "en":
        continue 

    print(edit_prompt)
    print('-'*10)

    image, accept = flux_main(
        name = "flux-dev-kontext",
            aspect_ratio =  "16:9",
            seed = seed,
            prompt = edit_prompt,
            device = "cuda" if torch.cuda.is_available() else "cpu",
            num_steps = num_steps,
            loop = False,
            guidance = 2.5,
            offload = False,
            output_dir = save_dir,
            img_cond_path = temp_img_path,
            track_usage = False,
            model=model,              # ✅ Preloaded model
            ae=ae,
            data = data,                # ✅ Preloaded autoencoder
            )
    # accept = output_dict['outputs']['accepted']
    accept_list.append(accept)

   
    # if image is not None:
    #     save_image(image, img_path)

        


    accept_list.append(accept)

    # print("Image getting saved to", img_path)

    # Save the returned image (if it's a PIL Image)
    # if image is not None:
    #     save_image(image, img_path)

print(sum(accept_list)/len(accept_list))

