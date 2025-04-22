import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import os

st.set_page_config(page_title="AI-Powered Air-Writing Calculator", layout="wide")

# --- Navbar --- #
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "CNN Model Approach", "Gemini API Approach"],
        icons=["house", "cpu", "stars"],
        menu_icon="cast",
        default_index=0
    )

# --- Home Page --- #
if selected == "Home":
    st.title("✍️ AI-Powered Air-Writing Calculator")
    st.markdown("""
    Welcome to the **Air-Writing Calculator** – an innovative way to perform math operations by simply writing in the air with your index finger!
    
    ### 🔍 Project Description
    This tool allows users to draw mathematical expressions in the air using hand gestures, which are then recognized and evaluated using either a CNN-based model or Google's Gemini API.
    
    ### 🚀 Features
    - Real-time hand tracking using webcam  
    - Expression recognition using trained CNN model  
    - Solve any math equation using Gemini API  
    - Clear or erase canvas with intuitive hand gestures  
    """)

    ### 🖐️ Gesture Manual
    st.subheader("🖐️ Gesture Manual")
    
    gesture_data = [
        ("Draw", "D:/Sem - 6/Minor 2/Handwritten-Feed-for-Calculation/Streamlit/assets/draw.png", "Draw with index finger ([0,1,0,0,0])"),
        ("Pause", "D:/Sem - 6/Minor 2/Handwritten-Feed-for-Calculation/Streamlit/assets/pause.png", "Stop Drawing ([0,1,1,0,0])"),
        ("Clear Canvas", "D:/Sem - 6/Minor 2/Handwritten-Feed-for-Calculation/Streamlit/assets/clear.png", "Reset entire canvas ([1,1,1,1,1])"),
        ("Erase (spot)", "D:/Sem - 6/Minor 2/Handwritten-Feed-for-Calculation/Streamlit/assets/erase.png", "Erase using index finger ([0,1,1,1,1])"),
        ("Trigger Gemini", "D:/Sem - 6/Minor 2/Handwritten-Feed-for-Calculation/Streamlit/assets/evaluate.png", "Evaluate the expression ([0,1,0,0,1])"),
        ("Backspace", "D:/Sem - 6/Minor 2/Handwritten-Feed-for-Calculation/Streamlit/assets/backspace.png", "Backspace last character in case of Gemini ([0,0,0,0,1])")
    ]
    
    for title, img_path, action in gesture_data:
        # Check if the image exists
        if os.path.exists(img_path):
            cols = st.columns([1, 2, 6])
            with cols[0]:
                st.image(Image.open(img_path), width=60)
            with cols[1]:
                st.markdown(f"**{title}**")
            with cols[2]:
                st.markdown(action)
        else:
            st.error(f"Image for **{title}** not found at {img_path}. Please check the path.")
    
    st.markdown("""
    ### 🧠 Available Approaches
    **1. CNN-Based Model:**  
    - Trained to recognize: digits 0-9, +, -, ×, ÷, !, √  
    - Fast and offline-capable  

    **2. Gemini API-Based Model:**  
    - Uses Google Gemini for expression recognition  
    - Handles **all kinds of mathematical expressions** accurately  
    - Requires internet connection  

    ---

    ### 🤝 Contributors
    - Sajal Korde  
    - Shruti Bhargava  
    - Snehil Sharma

    ### 📬 Contact
    For any queries or contributions, feel free to reach out to us via email.
    """)

# --- CNN Model Page --- #
elif selected == "CNN Model Approach":
    st.header("🧠 CNN Model Based Air-Writing Recognition")
    st.markdown("""
    This approach uses a **Convolutional Neural Network (CNN)** trained on digits and math operators drawn in the air.

    - Works offline using your webcam  
    - Supports digits (0-9), +, -, ×, ÷, !, √  
    - Designed to segment and classify each symbol before evaluating  
    """)
    if st.button("Get Started ✍️"):
        try:
            import cnn_model_runner  # Make sure this exists and handles camera
            cnn_model_runner.run()
        except ImportError:
            st.error("Error: CNN model runner module not found!")

# --- Gemini API Page --- #
elif selected == "Gemini API Approach":
    st.header("🌐 Gemini API Based Air-Writing Recognition")
    st.markdown("""
    This method uses **Google's Gemini API** to interpret your air-written math expressions.

    - Handles complex and diverse mathematical expressions  
    - Automatically returns final answer with high accuracy  
    - Requires internet access  
    """)
    if st.button("Get Started 🌟"):
        try:
            import gemini_runner  # Make sure this exists and handles camera
            gemini_runner.run_gemini_calculator()
        except ImportError:
            st.error("Error: Gemini runner module not found!")
