import math
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .util import PREFERED_KONTEXT_RESOLUTIONS
import torch.nn.functional as F



import math
import numpy as np
# import matplotlib.pyplot as plt


import numpy as np

class UCB1Bandit:
    def __init__(self, actions: list[int], c: float = 2.0):
        """
        UCB1 Bandit with a custom action set.
        
        Args:
            actions: List of possible actions (e.g., [1,2,3,4] for skip lengths).
            c: Exploration constant (higher = more exploration).
        """
        self.actions = np.array(actions)
        self.n_actions = len(actions)
        self.c = c
        self.q_values = np.zeros(self.n_actions, dtype=np.float32)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int32)
        self.total_steps = 0

    def choose_action(self) -> int:
        self.total_steps += 1

        # Try each action at least once
        untried_actions = np.where(self.action_counts == 0)[0]
        if len(untried_actions) > 0:
            return self.actions[untried_actions[0]]

        # UCB formula
        exploration_bonus = self.c * np.sqrt(
            np.log(self.total_steps) / self.action_counts
        )
        ucb_scores = self.q_values + exploration_bonus
        
        # Return the actual action (e.g., skip length), not the index
        return self.actions[np.argmax(ucb_scores)]

    def update(self, action: int, reward: float):
        idx = np.where(self.actions == action)[0][0]  # map action → index
        self.action_counts[idx] += 1
        n = self.action_counts[idx]
        self.q_values[idx] += (1 / n) * (reward - self.q_values[idx])


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(device)


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)
        img_cond = ae.encode(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict


def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict


def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_kontext(
    t5: HFEmbedder,
    clip: HFEmbedder,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    seed: int,
    device: torch.device,
    target_width: int | None = None,
    target_height: int | None = None,
    bs: int = 1,
) -> tuple[dict[str, Tensor], int, int]:
    # load and encode the conditioning image
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    width, height = img_cond.size
    aspect_ratio = width / height
    # Kontext is trained on specific resolutions, using one of them is recommended
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    width = 2 * int(width / 16)
    height = 2 * int(height / 16)

    img_cond = img_cond.resize((8 * width, 8 * height), Image.Resampling.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")
    img_cond_orig = img_cond.clone()

    with torch.no_grad():
        img_cond = ae.encode(img_cond.to(device))

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    # image ids are the same as base image with the first dimension set to 1
    # instead of 0
    img_cond_ids = torch.zeros(height // 2, width // 2, 3)
    img_cond_ids[..., 0] = 1
    img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(height // 2)[:, None]
    img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(width // 2)[None, :]
    img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

    if target_width is None:
        target_width = 8 * width
    if target_height is None:
        target_height = 8 * height

    img = get_noise(
        1,
        target_height,
        target_width,
        device=device,
        dtype=torch.bfloat16,
        seed=seed,
    )

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond_seq"] = img_cond
    return_dict["img_cond_seq_ids"] = img_cond_ids.to(device)
    return_dict["img_cond_orig"] = img_cond_orig
    return return_dict, target_height, target_width


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()

def verifier(v1: torch.Tensor, v2: torch.Tensor) -> float:
        if v1.shape != v2.shape:
            raise ValueError(f"Shape mismatch: {v1.shape} vs {v2.shape}")
        
        mse = F.mse_loss(v1, v2, reduction='mean')
        return mse.item()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens (channel-wise)
    img_cond: Tensor | None = None,
    # extra img tokens (sequence-wise)
    img_cond_seq: Tensor | None = None,
    img_cond_seq_ids: Tensor | None = None,
):
    accept = 0
    prev_pred = None
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        img_input = img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert (
                img_cond_seq_ids is not None
            ), "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        pred = model(
            img=img_input,
            img_ids=img_input_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        if img_input_ids is not None:
            pred = pred[:, : img.shape[1]]

        # if prev_pred is not None: 
        #         pred_new = prev_pred + velocity_old_diff[t_curr]
        #         confidence = verifier(pred_new, pred)
        #         print(f"the confidence at {t_curr} is {confidence}")
        #         threshold = 0.1
        #         if confidence<threshold:
        #             # print("The threshold is", threshold)
        #             pred = pred_new
        #             print(f"Accepted at timestep {t_curr}")
        #             accept+=1
        # prev_pred = pred

        # if prev_pred is not None:
        #     confidence = verifier(prev_pred, pred)
        #     # print(f"the confidence at {t_curr} is {confidence}")
        #     threshold = 0.015
        #     if confidence<threshold:
        #         # print("The threshold is", threshold)
        #         pred = prev_pred
        #         # print(f"Accepted at timestep {t_curr}")
        #         accept+=1
        # prev_pred = pred

        
        img = img + (t_prev - t_curr) * pred
    print(f"{accept}/50 acceptances.")
        # velocity_old[t_curr] = pred
    # print("The used threshold is", threshold)
    # with open("velocity_old.pkl", "wb") as f:
    #     pickle.dump(velocity_old, f)

    return img, accept

bandits = {t: UCB1Bandit(actions=[0, 2, 4, 6], c=2.0) for t in range(50)}


def denoise_with_ucb(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    max_steps_to_skip: int = 7,
    ucb_c: float = 2.0,
):
    # print("This runs.")
    accept = 0
    # --- Initialization ---
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    
#     bandits = {
#     t: UCB1Bandit(actions=[0, 2, 4, 6], c=ucb_c)
#     for t in timesteps[:-1]
# }
    
    last_pred: Optional[Tensor] = None
    prev_pred: Optional[Tensor] = None
    
    steps_to_skip = 0
    decision_info: Optional[dict] = None

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        def compute_prediction():
            img_input = img
            img_input_ids = img_ids
            pred = model(
                img=img_input, img_ids=img_input_ids, txt=txt,
                txt_ids=txt_ids, y=vec, timesteps=t_vec, guidance=guidance_vec,
            )
            if img_input_ids is not None:
                pred = pred[:, : img.shape[1]]
            return pred

        is_reward_step = decision_info and steps_to_skip == 0
        
        if is_reward_step:
            true_pred = compute_prediction()
            num_skips = decision_info['action']
            # print(num_skips)
            approximated_pred = (
                decision_info['pred_at_decision'] + 
                0.005 * (decision_info['pred_at_decision'] - decision_info['prev_pred_at_decision'])
            )
            mse = F.mse_loss(true_pred, approximated_pred)
            reward = (0.001 * num_skips) - mse.item()
            # print(reward)
            bandit_to_update = bandits[i]
            bandit_to_update.update(decision_info['action'], reward)
            pred = true_pred
            decision_info = None

        elif steps_to_skip > 0:
            pred = last_pred + 0.005*(last_pred - prev_pred)
            steps_to_skip -= 1
            accept+=1
            # print("accept")
        else:
            pred = compute_prediction()
            if last_pred is not None and prev_pred is not None:
                bandit = bandits[i]
                action_skips = bandit.choose_action()
                approximated_pred = last_pred
                mse = F.mse_loss(pred, approximated_pred)
                reward = (0.005 * action_skips) - mse.item()
                bandit.update(action_skips, reward)
                # print(action_skips)
                if action_skips > 0:
                    steps_to_skip = action_skips
                    decision_info = {
                        "timestep": t_curr, "action": action_skips,
                        "pred_at_decision": pred, "prev_pred_at_decision": last_pred,
                    }
        
        img = img + (t_prev - t_curr) * pred
        prev_pred = last_pred
        last_pred = pred
    print(f"{accept}/50 acceptances.")
            
    return img, accept


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
