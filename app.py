import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models import DinoDetector_V3

# device
DEVICE = torch.device("cpu")

# load model
REPO_DIR = "dinov3"

dinov3_vitb16 = torch.hub.load(
    REPO_DIR, 'dinov3_vitb16', 
    source='local', 
    weights='models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    
model = DinoDetector_V3(dinov3_vitb16)
state_dict = torch.load("models/final_model.pth",map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# main function
def predict_single_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    prob_fake = probabilities[0, 0].item()
    prob_real = probabilities[0, 1].item()
    pred_label = "Real" if prediction.item() == 1 else "Fake"

    return pred_label, prob_real, prob_fake


st.title("AI Image Detector")

url = 'https://arxiv.org/pdf/2601.20461'
st.write("Replicated from - [Exploiting the Final Component of Generator Architectures for AI-Generated Image Detection](%s)" % url)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    if st.button("Predict"):
        label, prob_real, prob_fake = predict_single_image(image)

        st.subheader(f"Prediction: {label}")
        st.write(f"Real: {prob_real:.4f} | Fake: {prob_fake:.4f}", )

    st.image(image, caption="Uploaded Image")
