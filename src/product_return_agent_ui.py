import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import tempfile

# has a bug, when the image is valid, cannot show the generated recommendation 

st.title("Product Return Agent")
st.subheader("Upload a product image or skip to test mode. Then enter product details to get a recommendation.")

# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []
if "validation_result" not in st.session_state:
    st.session_state['validation_result'] = None
if "skip_test" not in st.session_state:
    st.session_state['skip_test'] = False
if "image_path" not in st.session_state:
    st.session_state['image_path'] = None
if "product_title" not in st.session_state:
    st.session_state['product_title'] = ""
if "return_reason" not in st.session_state:
    st.session_state['return_reason'] = ""
if "awaiting_product_info" not in st.session_state:
    st.session_state['awaiting_product_info'] = False
if "test_label" not in st.session_state:
    st.session_state['test_label'] = 'valid'

# --- Chat History Display ---
chat_container = st.container()
with chat_container:
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- Step 1: Image Upload or Skip ---
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Skip to Test"):
        st.session_state['skip_test'] = True
        st.session_state['validation_result'] = None
        st.session_state['image_path'] = "dummy.jpg"
        st.session_state['chat_history'].append({
            "role": "user",
            "content": "Skipped image upload (test mode enabled)."
        })
        st.session_state['awaiting_product_info'] = False
        st.rerun()

with col2:
    if not st.session_state['skip_test']:
        uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state['image_path'] = tmp_file.name
            try:
                from main import validate_image
                result = validate_image(st.session_state['image_path'])
                st.session_state['validation_result'] = result
                st.session_state['skip_test'] = False
                st.session_state['chat_history'].append({
                    "role": "user",
                    "content": "Uploaded an image."
                })
                if result == "valid":
                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": "Image is valid. Please provide product title and reason for return."
                    })
                    st.session_state['awaiting_product_info'] = True
                else:
                    st.session_state['chat_history'].append({
                        "role": "assistant",
                        "content": f"Image validation result: {result}. Please re-upload a valid picture or skip to test mode."
                    })
                    st.session_state['awaiting_product_info'] = False
                st.rerun()
            except Exception as e:
                st.error(f"Image validation error: {e}")

# --- Step 2: Test Mode: Select Label ---
if st.session_state['skip_test']:
    st.info("Test mode enabled. Select an image label to simulate.")
    label = st.radio(
        "Select image label:",
        ("valid", "ai-generated", "photoshopped", "invalid"),
        index=["valid", "ai-generated", "photoshopped", "invalid"].index(st.session_state['test_label']),
        key="test_label_radio",
        horizontal=True
    )
    if label != st.session_state['test_label']:
        st.session_state['test_label'] = label
        st.session_state['validation_result'] = label
        st.session_state['awaiting_product_info'] = (label == 'valid')
        # Add to chat history
        st.session_state['chat_history'].append({
            "role": "user",
            "content": f"Test label selected: {label}"
        })
        if label == 'valid':
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": "Image is valid. Please provide product title and reason for return."
            })
        else:
            # Show error message as in backend
            if label == 'ai-generated':
                msg = 'Invalid picture: AI-generated. Please re-upload a valid picture.'
            elif label == 'photoshopped':
                msg = 'Invalid picture: Photoshopped. Please re-upload a valid picture.'
            else:
                msg = 'Invalid picture. Please re-upload a valid picture.'
            st.session_state['chat_history'].append({
                "role": "assistant",
                "content": msg
            })
        st.rerun()

    # If not valid, show error and do not show product info form
    if st.session_state['test_label'] != 'valid':
        st.warning("Please re-upload a valid picture or select 'valid' to proceed.")

# --- Step 3: Product Info Input (if ready) ---
if (st.session_state['skip_test'] and st.session_state['test_label'] == 'valid' and st.session_state.get('awaiting_product_info', False)) or (not st.session_state['skip_test'] and st.session_state.get('awaiting_product_info', False)):
    with st.form("product_info_form", clear_on_submit=True):
        product_title = st.text_input("Product Title")
        return_reason = st.text_area("Reason for Return")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.session_state['product_title'] = product_title
            st.session_state['return_reason'] = return_reason
            st.session_state['chat_history'].append({
                "role": "user",
                "content": f"Product Title: {product_title}  \nReason for Return: {return_reason}"
            })
            # --- Run the agent and append result ---
            try:
                from main import run_agent_streamlit
                override = st.session_state['test_label'] if st.session_state['skip_test'] else st.session_state['validation_result']
                result = run_agent_streamlit(
                    st.session_state['image_path'],
                    product_title,
                    return_reason,
                    image_validation_override=override
                )
                st.session_state['chat_history'].append({
                    "role": "assistant",
                    "content": f"Recommendation: {result}"
                })
            except Exception as e:
                st.session_state['chat_history'].append({
                    "role": "assistant",
                    "content": f"Error running agent: {e}"
                })
            st.session_state['awaiting_product_info'] = False
            st.rerun()

# --- Clear Chat Button ---
if st.button("Clear Chat"):
    for key in ["chat_history", "validation_result", "skip_test", "image_path", "product_title", "return_reason", "awaiting_product_info", "test_label"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- End Chat Button ---
if st.button("End Chat"):
    st.success("Thank you for using the Product Return Agent. Goodbye!")
    st.stop()
