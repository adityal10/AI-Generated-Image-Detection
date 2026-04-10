
import torch
import glob
import numpy as np

from training import training_loop
from sparse_sampling import select_sparse_samples
from data_generation import generate_vae_samples, generate_vq_samples, generate_diffusion_samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# step 1 - constructing datasets using algorithms 1,2 and 3
REAL_DIR = 'data/real/real_images'

generate_vae_samples(REAL_DIR=REAL_DIR,FAKE_DIR='output/vae',DEVICE=DEVICE)
generate_vq_samples(model_path="deepseek-ai/Janus-Pro-1B", REAL_DIR=REAL_DIR, FAKE_DIR='output/vq', DEVICE=DEVICE)
generate_diffusion_samples(model_path='models/pixelflow_model.pt', REAL_DIR=REAL_DIR, FAKE_DIR='output/diffusion', DEVICE=DEVICE)

# step 2 - selecting sparsed samples
_, real_features, selected_data_dict = select_sparse_samples(
    REAL_DIR=REAL_DIR, VAE_DIR='output/vae', 
    VQ_DIR='output/vq', DIFFUSION_DIR='output/diffusion', 
    REPO_DIR='dinov3', WEIGHTS_PATH='models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', DEVICE=DEVICE)

# step 3 - training the model
set_names = ["vae", "vq", "diffusion"]

# get all fake image paths
fake_image_paths = []
for _set in set_names:
    fake_image_paths.append(selected_data_dict[_set]['selected_paths'])

final_fake_image_paths = []
for sublist in fake_image_paths:
    final_fake_image_paths.extend(sublist)

# get all real image paths
real_image_paths = sorted(glob.glob(f"{REAL_DIR}/*.jpg"))[:300]

# final training loop
training_loop(
    REPO_DIR='dinov3', WEIGHT_PATH='models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', DEVICE=DEVICE,
    real_image_paths=real_image_paths, final_fake_image_paths=final_fake_image_paths)