import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import tempfile

st.title("Product Return Agent")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "stage" not in st.session_state:
    st.session_state["stage"] = "awaiting_image"
if "image_path" not in st.session_state:
    st.session_state["image_path"] = None
if "image_classification" not in st.session_state:
    st.session_state["image_classification"] = None
if "product_title" not in st.session_state:
    st.session_state["product_title"] = ""
if "return_reason" not in st.session_state:
    st.session_state["return_reason"] = ""

# --- Display Chat History ---
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chatbot Logic ---
def bot_say(content):
    st.session_state["chat_history"].append({"role": "assistant", "content": content})

def user_say(content):
    st.session_state["chat_history"].append({"role": "user", "content": content})

# --- Main Chatbot Flow ---
if st.session_state["stage"] == "awaiting_image":
    # Only prompt if not already prompted
    if not st.session_state["chat_history"] or st.session_state["chat_history"][-1]["content"] != "Welcome! Please upload a product image or type 'skip' to continue without an image.":
        bot_say("Welcome! Please upload a product image or type 'skip' to continue without an image.")
        st.rerun()

    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"], key="chat_image_upload")
    user_input = st.chat_input("Type 'skip' to continue without an image...")

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state["image_path"] = tmp_file.name

        from image_agent import ImageAgent
        image_analyzer = ImageAgent()
        classification = image_analyzer.classify_image(st.session_state["image_path"])
        st.session_state["image_classification"] = classification

        if classification == "real":
            bot_say("Image is valid. What is the product title?")
            st.session_state["stage"] = "awaiting_product_title"
        else:
            from main import IMAGE_CLASSIFICATION_MESSAGES
            error_msg = IMAGE_CLASSIFICATION_MESSAGES.get(classification, "Invalid picture. Please re-upload a valid picture.")
            bot_say(error_msg)
        st.rerun()

    elif user_input:
        user_say(user_input)
        if user_input.strip().lower() == "skip":
            st.session_state["image_path"] = "dummy.jpg"
            st.session_state["image_classification"] = "real"
            bot_say("Test mode enabled. What is the product title?")
            st.session_state["stage"] = "awaiting_product_title"
            st.rerun()
        else:
            bot_say("Please upload an image or type 'skip' to continue.")
            st.rerun()

elif st.session_state["stage"] == "awaiting_product_title":
    user_input = st.chat_input("Enter the product title")
    if user_input:
        user_say(user_input)
        st.session_state["product_title"] = user_input.strip()
        bot_say("What is the reason for return?")
        st.session_state["stage"] = "awaiting_return_reason"
        st.rerun()

elif st.session_state["stage"] == "awaiting_return_reason":
    user_input = st.chat_input("Enter the reason for return")
    if user_input:
        user_say(user_input)
        st.session_state["return_reason"] = user_input.strip()
        # --- Run the agent and append result ---
        try:
            from main import run_agent_streamlit
            result = run_agent_streamlit(
                st.session_state["image_path"],
                st.session_state["product_title"],
                st.session_state["return_reason"],
                image_classification_override=st.session_state["image_classification"]
            )
            bot_say(result)
            st.session_state["stage"] = "showing_recommendation"
        except Exception as e:
            bot_say(f"Error running agent: {e}")
        st.rerun()

elif st.session_state["stage"] == "showing_recommendation":
    user_input = st.chat_input("Type 'restart' to start over or ask another question.")
    if user_input:
        user_say(user_input)
        if user_input.strip().lower() == "restart":
            for key in ["chat_history", "stage", "image_path", "image_classification", "product_title", "return_reason"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        else:
            bot_say("Thank you for using the Product Return Agent. Type 'restart' to start over.")
            st.rerun()