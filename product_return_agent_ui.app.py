import streamlit as st
import tempfile

# has a bug, when the image is valid, cannot show the generated recommendation 

st.title("Product Return Agent")
st.write("Upload an image or skip to a test case.")

# --- Initialize Session State ---
if "validation_result" not in st.session_state:
    st.session_state.validation_result = None
if "skip_test" not in st.session_state:
    st.session_state.skip_test = False
if "image_path" not in st.session_state:
    st.session_state.image_path = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "product_title" not in st.session_state:
    st.session_state.product_title = ""
if "return_reason" not in st.session_state:
    st.session_state.return_reason = ""

# --- Callback function to run the agent ---
def run_recommendation_flow():
    """
    This function is called on form submission.
    It runs the agent and stores the result in session state.
    """
    st.session_state.submitted = True
    try:
        from main import run_agent_streamlit
        override = st.session_state.validation_result if st.session_state.skip_test else None
        result = run_agent_streamlit(
            st.session_state.image_path,
            st.session_state.product_title,
            st.session_state.return_reason,
            image_validation_override=override
        )
        st.session_state.recommendation = result
    except Exception as e:
        st.error(f"Error running agent: {e}")
        st.session_state.recommendation = None

# --- Step 1: Image Upload or Skip ---
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Skip to Test"):
        # Reset state for a new run
        st.session_state.skip_test = True
        st.session_state.validation_result = 'valid'
        st.session_state.recommendation = None
        st.session_state.submitted = False
        st.session_state.product_title = ""
        st.session_state.return_reason = ""
        st.rerun()

with col2:
    uploaded_file = st.file_uploader(
        "Upload a product image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.image_path = tmp_file.name
        try:
            from main import validate_image
            st.session_state.validation_result = validate_image(st.session_state.image_path)
            st.session_state.skip_test = False
            st.session_state.recommendation = None
            st.session_state.submitted = False
            st.rerun()
        except Exception as e:
            st.error(f"Image validation error: {e}")

# --- Step 2: Handle Test Case Selection ---
if st.session_state.skip_test:
    st.info("Test mode enabled.")
    test_case = st.radio(
        "Select validation result:",
        ("valid", "ai-generated", "photoshopped", "invalid"),
        index=0, # Default to 'valid'
        key="test_case_radio",
        horizontal=True
    )
    if st.session_state.validation_result != test_case:
        st.session_state.validation_result = test_case
        st.session_state.recommendation = None
        st.session_state.submitted = False

    st.session_state.image_path = "dummy.jpg"

# --- Step 3: Show Form if Ready ---
is_ready_for_form = st.session_state.validation_result == "valid"

if is_ready_for_form:
    # Only create the form if the condition is met.
    st.success("Image is valid. Please provide more details below.")
    
    with st.form("product_info_form"):
        st.text_input("Product Title", key="product_title") # No need for disabled here
        st.text_area("Reason for Return", key="return_reason") # No need for disabled here
        st.form_submit_button(
            "Submit",
            on_click=run_recommendation_flow
        )

# --- Step 4: Display Results or Errors ---
# This part of your code is already correct.
# It will now be reached correctly after the form is submitted.
if st.session_state.submitted:
    if st.session_state.recommendation:
        st.success("Recommendation:")
        st.markdown(st.session_state.recommendation)
    else:
        st.info("No suitable recommendation was found based on your input.")
elif st.session_state.validation_result: # This handles the initial error display
    if st.session_state.validation_result == "ai-generated":
        st.error("AI-generated picture. Please re-upload a valid picture.")
    elif st.session_state.validation_result == "photoshopped":
        st.error("Photoshopped picture. Please re-upload a valid picture.")
    elif st.session_state.validation_result != "valid":
        st.error("Invalid picture. Please re-upload a valid picture.")
