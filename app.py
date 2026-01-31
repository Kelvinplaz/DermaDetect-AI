import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import timm
import numpy as np
import matplotlib.cm as cm
import io

# --- Page Configuration ---
st.set_page_config(page_title="DermaDetectAI", layout="centered", page_icon="🧬")

# === Constants ===
LABELS = ['AKIEC', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'VASC']
MALIGNANT_CLASSES = {'AKIEC', 'BCC', 'MEL'}

FULL_LABELS = {
    'AKIEC': 'Actinic Keratoses and Intraepithelial Carcinoma',
    'BCC': 'Basal Cell Carcinoma',
    'BKL': 'Benign Keratosis-like Lesions',
    'DF': 'Dermatofibroma',
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevi',
    'VASC': 'Vascular Lesions'
}

DESCRIPTIONS = {
    'AKIEC': "Early signs of sun-damage that may progress to squamous cell carcinoma.",
    'BCC': "Common, slow-growing skin cancer. Rarely spreads but needs treatment.",
    'BKL': "Non-cancerous age/sun-related skin growths. Includes seborrheic keratoses.",
    'DF': "Benign nodule from minor trauma. Firm and brown/pink in appearance.",
    'MEL': "Aggressive skin cancer from melanocytes. Urgent diagnosis crucial.",
    'NV': "Common harmless moles formed by melanocytes.",
    'VASC': "Benign blood vessel growths like hemangiomas. Usually red/purple."
}

# === Load Model ===
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("convnextv2_base.fcmae_ft_in1k", pretrained=True, num_classes=len(LABELS))
    model.load_state_dict(torch.load("best_modelcvn.pth", map_location=device))
    model.eval().to(device)
except FileNotFoundError:
    st.error("Model file 'best_modelcvn.pth' not found.")
    st.stop()

# === Grad-CAM Function ===
def generate_gradcam(input_tensor, class_idx):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.stages[-1].blocks[-1]
    fwd = target_layer.register_forward_hook(forward_hook)
    bwd = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    pred = output[0, class_idx]
    pred.backward()

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    cam = torch.mean(acts, dim=1).squeeze()
    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam_np = cam.detach().cpu().numpy()

    heatmap = cm.jet(cam_np)[..., :3]
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    original = transforms.ToPILImage()(input_tensor.squeeze().cpu()).resize((224, 224))
    blended = Image.blend(original, heatmap, alpha=0.5)

    fwd.remove()
    bwd.remove()
    return blended

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

st.markdown("""
<style>
    .reportview-container, .main {
        background-color: #fce4ec;
    }

    .title-tagline {
        text-align: center;
        font-family: 'Inter', serif;
        margin-top: 1.5rem;
    }

    .title-text {
        font-size: 44px;
        font-weight: 700;
        color: #6a1b9a;
        margin-bottom: 0.2rem;
    }

    .tagline {
        font-size: 20px;
        color: #880e4f;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed #f28ab2;
        border-radius: 10px;
        padding: 20px;
        background-color: #fffafa;
        text-align: center;
        margin-top: 1rem;
    }

    div[data-testid="stFileUploader"] p {
        font-size: 1.1em;
        font-weight: bold;
        color: #880e4f;
    }

    .stButton>button {
        font-size: 18px;
        padding: 0.6em 1.5em;
        border-radius: 8px;
        background-color: #d81b60;
        color: white;
        border: none;
    }

    .stButton>button:hover {
        background-color: #ad1457;
    }

    .disclaimer {
        font-size: 0.85rem;
        color: #5a5a5a;
        margin-top: 1rem;
    }
</style>

<!-- Load Playfair Display font -->
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)



# === Title & Tagline ===
st.markdown("""
<div class="title-tagline">
    <div class="title-text">DermaDetectAI</div>
    <div class="tagline">Your Skin's Safety, Our Priority</div>
</div>
""", unsafe_allow_html=True)

# === Upload Section ===
uploaded_file = st.file_uploader("Upload a Skin Lesion Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

# === Process Image ===
if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB")

    if st.button("Analyze"):
        padded_img = ImageOps.fit(input_img, (224, 224), method=Image.Resampling.BILINEAR)

        st.markdown("### Preview and Grad-CAM")
        col1, col2 = st.columns(2)

        with st.spinner("Analyzing..."):
            input_tensor = transform(padded_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                pred_label = LABELS[pred_idx]
                confidence = float(probs[pred_idx].cpu().numpy()) * 100

            gradcam_img = generate_gradcam(input_tensor, pred_idx)

        with col1:
            st.image(padded_img, caption="Original Image", use_container_width=True)
            buf1 = io.BytesIO()
            padded_img.save(buf1, format="PNG")
            st.download_button("Download Original Image", buf1.getvalue(), "original.png", "image/png")

        with col2:
            st.image(gradcam_img, caption="Grad-CAM", use_container_width=True)
            buf2 = io.BytesIO()
            gradcam_img.save(buf2, format="PNG")
            st.download_button("Download Grad-CAM Image", buf2.getvalue(), "gradcam.png", "image/png")

        st.markdown("---")
        st.markdown(f"### Prediction: **{pred_label}** — *{FULL_LABELS[pred_label]}*")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown(f"**Details:** {DESCRIPTIONS[pred_label]}")

        if pred_label in MALIGNANT_CLASSES:
            st.error("⚠️ This condition may be **malignant**. Please consult a dermatologist.")
        else:
            st.success("✅ This appears to be **benign**. Clinical confirmation is still advised.")

        st.markdown('<p class="disclaimer"> Disclaimer: This AI tool is for educational and preliminary screening purposes only. It is not a substitute for professional medical advice.</p>', unsafe_allow_html=True)
else:
    st.info("Please upload a skin lesion image to begin analysis.")
