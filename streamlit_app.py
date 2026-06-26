import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# Import modules from our project layout
import data_loader
import model

# --- PAGE SETUP ---
st.set_page_config(page_title="EuroSAT AI Classifier", page_icon="🌍", layout="centered")

st.title("🌍 EuroSAT Land Cover Predictor")
st.markdown("This AI model processes satellite imagery to classify land usage.")

# --- AUTO-DOWNLOAD DATASET / ASSETS ---
@st.cache_resource
def initialize_app():
    """Downloads dataset and prepares the AI network layout."""
    # Run kagglehub auto-download
    dataset_path = data_loader.download_dataset()
    
    # Initialize our PyTorch neural network model
    ai_model = model.get_eurosat_model(num_classes=10)
    return ai_model, dataset_path

# Display loading indicator while downloading/setting up components
with st.spinner("Downloading EuroSAT dataset assets via kagglehub... Please wait..."):
    ai_model, dataset_path = initialize_app()

# --- EXPLAINING ML CONCEPTS ---
with st.sidebar:
    st.header("🧠 Machine Learning Specs")
    st.markdown("**User Input (Features / $X$):**")
    st.caption("A single $64 \\times 64$ pixel satellite image patch. To the computer, this is interpreted as a grid matrix of color-intensity values.")
    
    st.markdown("**Model Output (Target / $\\hat{y}$):**")
    st.caption("A specific land cover category label paired with a calculated confidence percentage score.")

# --- FILE UPLOADER (USER INPUT) ---
st.subheader("1. Upload User Input ($X$)")
uploaded_file = st.file_uploader("Upload a satellite image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and display the user image input
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Uploaded Image", use_container_width=True)
    
    with col2:
        # Standardize User Input into a 64x64 pixel image grid as required by EuroSAT
        resized_image = image.resize((64, 64))
        st.image(resized_image, caption="Resized Image Input ($64 \\times 64$ Matrix)", width=150)
        
    # --- PRE-PROCESSING THE IMAGE FOR THE MODEL ---
    # Convert image to numerical tensor grids and scale intensities
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add a batch dimension -> transform shape from [3, 64, 64] to [1, 3, 64, 64]
    input_tensor = preprocess(image).unsqueeze(0)

    # --- MODEL INFERENCE (PREDICTION) ---
    st.subheader("2. Model Output Prediction ($\\hat{y}$)")
    
    with torch.no_grad():
        # Pass input matrix X into the neural network to calculate logits
        outputs = ai_model(input_tensor)
        # Apply Softmax to translate raw network output numbers into probability percentages
        probabilities = F.softmax(outputs, dim=1)[0]
        
    # Find the position index of the maximum calculated confidence score
    confidence, class_idx = torch.max(probabilities, 0)
    predicted_label = data_loader.CLASSES[class_idx.item()]
    confidence_percentage = confidence.item() * 100

    # Display results prominently to the end-user
    st.success(f"### Predicted Class: **{predicted_label}**")
    st.metric(label="Prediction Confidence Score", value=f"{confidence_percentage:.2f}%")

    # --- BREAKDOWN OF PROBABILITIES ---
    with st.expander("View Full Confidence Metrics Across All 10 Classes"):
        for i, class_name in enumerate(data_loader.CLASSES):
            prob = probabilities[i].item() * 100
            st.write(f"**{class_name}**: {prob:.2f}%")
            st.progress(int(prob))
else:
    st.info("ℹ️ Upload a sample image above to generate an AI model prediction output.")