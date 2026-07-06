# 🦠 Malaria AI Detector Dashboard

A state-of-the-art, hyper-modern web application built with Streamlit and TensorFlow that instantly detects the presence of malaria parasites in microscopic blood cell images. 

This project goes beyond standard Streamlit applications by implementing a bespoke "Aurora/Web3" dashboard aesthetic using deep custom CSS and HTML injections.

## 🚀 Features

- **Deep Learning Inference**: Seamlessly loads a pre-trained Convolutional Neural Network (`malaria_model.h5`) to analyze cellular structures with high accuracy.
- **Batch Processing**: Supports multi-file uploads, allowing users to analyze several blood cell images simultaneously.
- **Next-Gen Aesthetic**:
  - **Aurora Background**: Beautiful, deep dark theme featuring glowing radial gradients.
  - **Glassmorphism Design**: Frosted glass containers and tightly bound prediction cards.
  - **Perfectly Symmetrical Layouts**: Uses strict flexbox containers and CSS `object-fit` constraints to ensure all images and result cards render perfectly, regardless of the uploaded image dimensions.
- **Real-Time Preprocessing**: Automatically resizes (64x64) and normalizes uploaded images before passing them to the AI model.

## 🛠️ Technology Stack

- **Frontend/UI**: Streamlit (with extensive Custom HTML/CSS)
- **Machine Learning**: TensorFlow / Keras
- **Image Processing**: Pillow (PIL), NumPy
- **Encoding**: Base64 for inline HTML image rendering

## 💡 How It Works

1. **Upload**: Users upload microscopic images of blood cells (`.png`, `.jpg`, `.jpeg`).
2. **Process**: The app converts the images to Base64 for the custom UI rendering, while simultaneously resizing and normalizing the raw arrays for the AI model.
3. **Predict**: The CNN model predicts the probability of infection.
4. **Display**: The UI dynamically generates a custom-styled result card (Teal for Negative/Healthy, Red for Positive/Infected) alongside the original image, displaying the AI's confidence level.

## 💻 Running Locally

1. Ensure you have the required dependencies installed:
   ```bash
   pip install streamlit tensorflow numpy pillow
   ```
2. Place your trained model (`malaria_model.h5`) in the root directory.
3. Start the application:
   ```bash
   streamlit run app.py
   ```
