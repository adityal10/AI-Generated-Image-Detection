import torch
from diffusers import AutoencoderKL

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

from PixelFlow.pixelflow.model import PixelFlowModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DinoDetector_V3(torch.nn.Module):
    """
    Normal architecture with dinvo3 backbone
    Linear classfication layer
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # self.dropout = nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, 2) # binary classification

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
    

def load_vae():
    vae = AutoencoderKL.from_pretrained('sd2-community/stable-diffusion-2-1', subfolder="vae",torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
    return vae

def load_vq(model_path, DEVICE):    
    model_path = model_path

    # load the processor
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # load the model with device_map
    # Using device_map="cuda" to automatically handle cuda transfer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE
    )

    vl_gpt.eval()

    return vl_gpt

def load_diffusion(model_path, DEVICE):
    denoiser = PixelFlowModel(
        num_attention_heads=16, attention_head_dim=72, in_channels=3, out_channels=3, depth=28, num_classes=1000, patch_size=4
    )
 
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if "state_dict" in checkpoint:
        message = denoiser.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        message = denoiser.load_state_dict(checkpoint, strict=False)
    print(f"Loaded weights with partial match: {message}")

    denoiser.to(DEVICE)
    denoiser.eval()

    return denoiser

def load_dinvo3(REPO_DIR, WEIGHTS_PATH, DEVICE):

    dinov3_vitb16 = torch.hub.load(REPO_DIR, 'dinov3_vitb16', source='local', weights=WEIGHTS_PATH)

    dinov3_vitb16.to(DEVICE)
    dinov3_vitb16.eval()

    return dinov3_vitb16