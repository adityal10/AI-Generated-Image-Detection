import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class SimpleTrainingDataset(Dataset):
    def __init__(self, real_image_paths, fake_image_paths, image_size=224, seed=42):
        self.real_paths = list(real_image_paths)
        self.fake_paths = list(fake_image_paths)
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # random seed
        rng = np.random.RandomState(seed)
        
        num_real = len(self.real_paths)
        num_fake = len(self.fake_paths)
        max_len = max(num_real, num_fake)
        
        # Sample with replacement to match lengths
        real_indices = rng.choice(num_real, size=max_len, replace=True)
        fake_indices = rng.choice(num_fake, size=max_len, replace=True)
        
        # create image with real:1 and fake:0
        self.images = []
        for i in range(max_len):
            self.images.append((self.real_paths[real_indices[i]], 1))
            self.images.append((self.fake_paths[fake_indices[i]], 0))
        
        # Shuffle images (non-pairwise training)
        rng.shuffle(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)