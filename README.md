# FastFlow with Bandit Inference (Flux-dev Extension)

This repository contains an implementation of ideas from the paper **["FastFlow: Accelerating the Generative Flow Matching Models with Bandit Inference"]**, adapted for the **Flux-dev** model.  

## Key Updates  
The main modifications are in:  
- `src/flux/cli.py`  
- `src/flux/sampling.py`  

## New Features  
- **`UC1Bandit` class**  
  - Implements a multi-armed bandit at every timestep.  
  - Determines which timesteps can be skipped to reduce computation while maintaining generation quality.  

- **`denoise_with_ucb` function**  
  - Performs denoising using the multi-armed bandit strategy.  
  - Dynamically adapts inference to accelerate sampling.  

## Summary  
This fork integrates **bandit-based timestep selection** into the Flux-dev generative process, enabling **faster inference** with minimal loss in output fidelity.
