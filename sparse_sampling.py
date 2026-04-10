from sklearn.metrics.pairwise import euclidean_distances
from kmedoids import KMedoids
from scipy.spatial.distance import cdist
import torch
from PIL import Image
import numpy as np

from processing import sparse_image_processing
from models import load_dinvo3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import glob


def extract_features(image_paths, model, DEVICE):

    set_features = []

    with torch.no_grad():
        for path in image_paths:
            # load and transform image
            img = Image.open(path).convert('RGB')
            img_t = sparse_image_processing(img).unsqueeze(0).to(DEVICE)
            
            # Extract the global average pool
            feat = model(img_t) 
            set_features.append(feat.squeeze().cpu())
        
        return torch.stack(set_features).numpy()

def select_sparse_samples(REAL_DIR, VAE_DIR, VQ_DIR, DIFFUSION_DIR, REPO_DIR, WEIGHTS_PATH, DEVICE):

    real_image_paths = sorted(glob.glob(f"{REAL_DIR}/*.jpg"))[:300]
    vae_image_paths = sorted(glob.glob(f"{VAE_DIR}/*.jpg"))[:300]
    vq_image_paths = sorted(glob.glob(f"{VQ_DIR}/*.jpg"))[:300]
    diffusion_image_paths = sorted(glob.glob(f"{DIFFUSION_DIR}/*.jpg"))

    # load model
    model = load_dinvo3(REPO_DIR, WEIGHTS_PATH)

    # initialize the main dictionary
    selected_data_dict = {}
    all_features_list = [] 

    set_names = ["vae", "vq", "diffusion"]
    image_sets = [vae_image_paths, vq_image_paths, diffusion_image_paths]

    k = 100

    for i, img_set in enumerate(image_sets):
        name = set_names[i]
        print(f"Extracting {name} features for set with {len(img_set)} images...")

        # extract features    
        set_feat = extract_features(img_set, model, DEVICE)
        all_features_list.append(set_feat)

        # run k-medoids
        km = KMedoids(n_clusters=k, metric='euclidean', method='fasterpam')
        km.fit(set_feat)

        # get indicies
        indices = km.medoid_indices_

        # create sub-directory
        selected_data_dict[name] = {
            "selected_paths": [img_set[idx] for idx in indices],
            "selected_features": set_feat[indices]
        }
        
        # selected_features = set_feat[km.medoid_indices_]
        # selected_features_list.append(selected_features)

    all_original_features = np.concatenate(all_features_list, axis=0)

    # extract real features
    real_image_features = extract_features(real_image_paths, model, DEVICE=DEVICE)

    return all_original_features, real_image_features, selected_data_dict