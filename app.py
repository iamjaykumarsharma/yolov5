import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

# Load the model
# Make sure to update the path to where your best.pt is located
#MODEL_PATH = 'best.pt'
#model = torch.hub.load('yolov5','custom', path='best.pt',force_reload=True,source='local', pretrained =True)
model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')  # local repo

st.title('YOLOv5 Custom Model Deployment')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert image to numpy array
    image_np = np.array(image)

    # Perform prediction
    results = model(image_np)

    # Render results
    results.render()  # updates results.imgs with boxes and labels
    st.image(results.imgs[0], caption='Processed Image', use_column_width=True)
    
    # Optional: Display results as dataframe
    st.write(results.pandas().xyxy[0])

# Ensure your Streamlit app runs
if __name__ == '__main__':
    st.run()