import streamlit as st
from streamlit_option_menu import option_menu

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
    
    ### 🖐️ Gesture Manual
    | Gesture            | Fingers Up        | Action                  |
    |--------------------|-------------------|--------------------------|
    | Draw               | [0,1,0,0,0]        | Draw with index finger   |
    | Clear Canvas       | [1,1,1,1,1]        | Reset entire canvas      |
    | Erase (spot)       | [0,1,1,1,1]        | Erase with index         |
    | Trigger Gemini     | [0,1,0,0,1]        | Solve using Gemini API   |

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
    - Shruti Bhargava – [shruti.cse2021@gmail.com](mailto:shruti.cse2021@gmail.com)  
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
        import cnn_model_runner  # will run the camera

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
        import gemini_runner  # will run the camera