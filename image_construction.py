from PIL import Image
import torch
import torchvision.transforms as T
from processing import diffusion_image_processing
from models import load_diffusion

def reconstruct_image_vq(model, pixel_values):

    # encode: extracting the latent 'h'
    # return output type tuple in Janus codebase
    encoded_output = model.gen_vision_model.encode(pixel_values)
    h = encoded_output[0] if isinstance(encoded_output, tuple) else encoded_output

    # quantize: converting 'h' to discrete tokens 'z'
    # next-token auto-regression, where tokens are predicted sequentially
    quant_output = model.gen_vision_model.quantize(h)
    z = quant_output[0] if isinstance(quant_output, tuple) else quant_output

    # decode: converting discrete tokens back to pixels
    reconstructed = model.gen_vision_model.decode(z)

    # post-process for saving
    out = reconstructed.squeeze(0).cpu().float().clamp(0, 1)
    return T.ToPILImage()(out)

def construct_image_diffusion(model_path, image_paths, DEVICE):
    model = load_diffusion(model_path=model_path, DEVICE=DEVICE)

    # Load all images in the current batch list
    imgs = [diffusion_image_processing(Image.open(p).convert("RGB")) for p in image_paths]
    
    x_t_real = torch.stack(imgs).to(DEVICE)
    curr_batch_size = x_t_real.shape[0]

    # hyperparameters from the paper (Algorithm 3)
    T_STEPS = 25     # no. of denoising iterations
    STRENGTH = 0.2      # noise strength

    RES = 256
    PATCH_SIZE = 4
    HEAD_DIM = model.attention_head_dim # 72
    SEQ_LEN = (RES // PATCH_SIZE) ** 2     # 4096

    # Create the dummy embedding once and keep it on the GPU
    DUMMY_POS = torch.zeros((SEQ_LEN, HEAD_DIM, 2), device=DEVICE).half()
    CLASS_LABELS = torch.zeros((1,), device=DEVICE, dtype=torch.long)
    CURR_LATENT_SIZE = torch.tensor([float(RES)], device=DEVICE).half()

    # adding noise
    epsilon = torch.randn_like(x_t_real)
    x_t = (1 - STRENGTH) * x_t_real + STRENGTH * epsilon
    
    batch_class_labels = torch.zeros((curr_batch_size,), device=DEVICE, dtype=torch.long)
    dt = STRENGTH / T_STEPS

    
    # denoising
    for i in range(T_STEPS):
        t_val = STRENGTH - (i * dt)
        curr_t = torch.full((curr_batch_size,), t_val * 1000, device=DEVICE,dtype=torch.float32)

        with torch.amp.autocast('cuda'):
            v_pred = model(hidden_states=x_t, encoder_hidden_states=None, class_labels=batch_class_labels,
                timestep=curr_t, # Most models scale t by 1000
                latent_size=CURR_LATENT_SIZE, pos_embed=DUMMY_POS
            )
        x_t = x_t - v_pred * dt 
    
    # post process
    x_hat = (x_t + 1) / 2
    x_hat = x_hat.clamp(0, 1).cpu().float()
    return [T.ToPILImage()(img) for img in x_hat]