from PIL import Image
import torch
import torchvision.transforms as T

def vae_image_processing():
    # image processing
    # why normalize, vae takes image input with a size of 512 and input pixels ranging between [-1,1]
    preprocess = T.Compose([
        T.CenterCrop(512), # center crop to size 512x512
        T.ToTensor(),
        T.Normalize([0.5], [0.5]) # transform input pixels from [0,1] to [-1,1]
    ])
    return preprocess

# we transform the input pixels from [-1,1] to [0,1] for standard .jpg format
def vae_postprocess(tensor):
    # Move back to [0, 1] range for PIL
    tensor = (tensor / 2 + 0.5).clamp(0, 1)
    return T.ToPILImage()(tensor.cpu().squeeze())

def vq_image_processing(image_path):
    img = Image.open(image_path).convert("RGB")

    # center crop and resize to 378x378 as per Janus
    transform = T.Compose([
        T.Resize(378, interpolation=T.InterpolationMode.LANCZOS),
        T.CenterCrop(378),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(torch.bfloat16).cuda()
    
def diffusion_image_processing():
    # load image and transform image
    RES = 256
    transform = T.Compose([
        T.CenterCrop(RES),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Efficiently maps to [-1, 1]
    ])

    return transform

def sparse_image_processing():
    # Transform for DINOv3 input
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform