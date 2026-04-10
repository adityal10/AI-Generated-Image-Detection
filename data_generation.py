# imports
import os
from PIL import Image
from tqdm import tqdm
import glob

# pytorch
import torch
import torchvision.transforms as T

from image_construction import reconstruct_image_vq, construct_image_diffusion
from models import load_vae, load_vq, load_diffusion
from processing import vae_image_processing, vae_postprocess, vq_image_processing, diffusion_image_processing

def generate_vae_samples(REAL_DIR, FAKE_DIR, DEVICE):
    model = load_vae()

    real_images = sorted([f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')])

    print(f"Processing {len(real_images)} images through VAE...")

    with torch.no_grad():
        for filename in tqdm(real_images):
            img_path = os.path.join(REAL_DIR, filename)
            save_path = os.path.join(FAKE_DIR, filename)
            
            if os.path.exists(save_path): 
                continue
                
            # Load and convert
            raw_img = Image.open(img_path).convert("RGB")
            input_tensor = vae_image_processing(raw_img).unsqueeze(0).to(DEVICE)
            
            if DEVICE == "cuda":
                input_tensor = input_tensor.half()

            # .sample(): adding mathematical noise
            latents = model.encode(input_tensor).latent_dist.sample()
            reconstructed = model.decode(latents).sample
            
            # Save back as image
            vae_postprocess(reconstructed).save(save_path)

    print(f"Done! VAE samples are ready in {FAKE_DIR}")

@torch.no_grad()
def generate_vq_samples(model_path, REAL_DIR, FAKE_DIR, DEVICE):
    model = load_vq(model_path, DEVICE)

    real_images = sorted([f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')])

    print(f"Constructing {len(real_images)} images...")

    for filename in tqdm(real_images):
        try:
            # load image
            img_path = os.path.join(REAL_DIR, filename)
            # real_img = Image.open(img_path).convert("RGB")

            # pre-process the image
            pixel_values = vq_image_processing(img_path)

            # construct image
            vq_image = reconstruct_image_vq(model, pixel_values)

            # Save
            vq_image.save(os.path.join(FAKE_DIR, f"vq_{filename}"))

        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    print(f"Done! VQ samples are ready in {FAKE_DIR}")

def generate_diffusion_samples(model_path, REAL_DIR, FAKE_DIR, DEVICE):

    model = load_diffusion(model_path, DEVICE)

    # loading images and creating directories
    input_folder = REAL_DIR
    image_paths = sorted(glob.glob(f"{input_folder}/*.jpg"))[:300]
    output_dir = FAKE_DIR
    os.makedirs(output_dir, exist_ok=True)

    BATCH_SIZE = 8

    pbar = tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Processing Batches")

    for i in pbar:
        batch_paths = image_paths[i : i + BATCH_SIZE]
        
        results = construct_image_diffusion(batch_paths, model)
        
        for img, path in zip(results, batch_paths):
            filename = os.path.basename(path)
            save_path = os.path.join(output_dir, filename)
            img.save(save_path)

        pbar.set_postfix({
            "batch_size": len(batch_paths),
            "last_file": os.path.basename(batch_paths[-1])
        })
