import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import requests

# =============================
# APP CONFIG
# =============================
st.set_page_config(page_title="Dental AI App", layout="centered")
st.title("🦷 Dental Disease Detection & Help App")

# =============================
# MODEL DOWNLOAD
# =============================
MODEL_URL = "https://huggingface.co/spaces/Sanjai-2005/Dental-App/resolve/main/model.pt"
MODEL_PATH = "model.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model (first time only)..."):
            r = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

# =============================
# IMAGE TRANSFORM
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ["Healthy", "Calculus", "Gingivitis"]
disease = None

# =============================
# IMAGE UPLOAD
# =============================
st.subheader("📸 Upload Dental Image")
uploaded_file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    disease = class_names[pred.item()]
    st.success(f"🧾 Detected: **{disease}**")

# =============================
# CURE SUGGESTIONS
# =============================
st.subheader("💊 Patient Care Advice")

if disease == "Calculus":
    st.write("""
    • Professional scaling required  
    • Brush twice daily  
    • Use anti-plaque mouthwash  
    • Avoid tobacco  
    """)
elif disease == "Gingivitis":
    st.write("""
    • Maintain oral hygiene  
    • Use medicated mouthwash  
    • Avoid sugary food  
    • Visit dentist if bleeding continues  
    """)
elif disease == "Healthy":
    st.write("""
    • Teeth look healthy  
    • Continue brushing twice daily  
    • Regular dental checkups  
    """)
else:
    st.info("Upload image to see care advice")

# =============================
# OFFLINE Q&A SYSTEM
# =============================
st.subheader("❓ Ask Dental Questions (Offline Support)")

OFFLINE_QA = {
    "calculus": "Dental calculus is hardened plaque caused by poor oral hygiene. It requires professional scaling.",
    "gingivitis": "Gingivitis is early-stage gum disease causing redness and bleeding. It is reversible with good care.",
    "bleeding gums": "Bleeding gums are usually caused by gingivitis, poor brushing, or vitamin deficiency.",
    "tooth decay": "Tooth decay happens when bacteria damage enamel due to sugary food and poor brushing.",
    "bad breath": "Bad breath may be due to plaque, tongue bacteria, or gum disease.",
    "scaling": "Scaling is a dental procedure to remove hardened plaque from teeth.",
    "brushing": "Brush twice daily using fluoride toothpaste for healthy teeth.",
    "mouthwash": "Mouthwash helps reduce bacteria and improve gum health.",
    "sugar": "Excess sugar increases tooth decay and gum disease risk.",
    "healthy teeth": "Healthy teeth are pink gums, no pain, no bleeding, and fresh breath."
}

user_question = st.text_input("Type your dental question:")

if user_question:
    found = False
    q = user_question.lower()

    for key in OFFLINE_QA:
        if key in q:
            st.success(OFFLINE_QA[key])
            found = True
            break

    if not found:
        st.warning("❌ Question not available offline. Please consult a dentist.")
