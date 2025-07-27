import streamlit as st
import tempfile
import os
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import numpy as np
import pandas as pd
from image_agent import ImageAgent

# Set page config
st.set_page_config(
    page_title="Image Classification Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .feedback-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .comparison-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔍 Image Classification Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Two-stage AI-powered image classification: Real vs AI-Generated vs Photoshopped</p>', unsafe_allow_html=True)

# Initialize the agent
@st.cache_resource
def load_agent():
    try:
        agent = ImageAgent()
        return agent
    except Exception as e:
        st.error(f"Error loading agent: {e}")
        return None

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = []

# Load the agent
agent = load_agent()

if agent is None:
    st.error("❌ Failed to load Image Agent. Please check the model files and try again.")
    st.stop()
else:
    st.success("✅ Image Agent loaded successfully!")

def create_confidence_chart(probabilities, stage_name):
    """Create a confidence chart using Plotly"""
    if not probabilities:
        return None
    
    # Extract class names and values
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
        )
    ])
    
    fig.update_layout(
        title=f'{stage_name} Classification Probabilities',
        xaxis_title='Classes',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )
    
    return fig

def create_pie_chart(probabilities, stage_name):
    """Create a pie chart for probabilities"""
    if not probabilities:
        return None
    
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    fig = px.pie(
        values=values,
        names=classes,
        title=f'{stage_name} Probability Distribution'
    )
    
    fig.update_layout(height=400)
    return fig

def classify_image_with_ui(uploaded_file):
    """Classify uploaded image and display results"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Classify the image with detailed analysis
        classification, confidence, probabilities, analysis = agent.classify_image_with_analysis(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return classification, confidence, probabilities, analysis
        
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None, None, None, None

# Main app
def main():
    # Sidebar for feedback history
    with st.sidebar:
        st.header("📊 Session History")
        
        # Feedback history
        if st.session_state.feedback_data:
            st.subheader("💬 Feedback History")
            for i, feedback in enumerate(st.session_state.feedback_data[-5:]):  # Show last 5
                with st.expander(f"Feedback {i+1} - {feedback['classification'].title()}"):
                    st.write(f"**Classification:** {feedback['classification'].title()}")
                    st.write(f"**Confidence:** {feedback['confidence']:.3f}")
                    st.write(f"**Reasoning:** {feedback['reasoning']}")
                    st.write(f"**Time:** {feedback['timestamp']}")
        
        # Comparison history
        if st.session_state.comparison_data:
            st.subheader("🔄 Comparison History")
            for i, comparison in enumerate(st.session_state.comparison_data[-5:]):  # Show last 5
                with st.expander(f"Comparison {i+1} - {comparison['comparison_type'].title()}"):
                    st.write(f"**Original:** {comparison['original_class'].title()} ({comparison['original_conf']:.3f})")
                    st.write(f"**Comparison:** {comparison['comparison_class'].title()} ({comparison['comparison_conf']:.3f})")
                    st.write(f"**Type:** {comparison['comparison_type'].title()}")
                    st.write(f"**Time:** {comparison['timestamp']}")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.feedback_data = []
            st.session_state.comparison_data = []
            st.success("History cleared!")
            st.rerun()
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image to classify it as Real, AI-Generated, or Photoshopped"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Classification button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Classifying image..."):
                    classification, confidence, probabilities, analysis = classify_image_with_ui(uploaded_file)
                
                if classification:
                    # Display results
                    st.header("📊 Classification Results")
                    
                    # Classification result
                    classification_emoji = {
                        "real": "👤",
                        "ai_generated": "🤖", 
                        "photoshopped": "✂️"
                    }
                    
                    emoji = classification_emoji.get(classification, "❓")
                    
                    # Determine confidence color
                    if confidence >= 0.8:
                        conf_class = "confidence-high"
                    elif confidence >= 0.6:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <h3>{emoji} Classification: {classification.replace('_', ' ').title()}</h3>
                        <p><span class="{conf_class}">Confidence: {confidence:.3f} ({confidence*100:.1f}%)</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interactive prompts based on classification
                    st.header("💬 Follow-up Questions")
                    
                    if classification == "real":
                        st.info("🎯 **Real Image Detected**")
                        reason = st.text_area(
                            "Why do you think this image is real?",
                            placeholder="Please provide your reasoning for why you believe this is a real image...",
                            height=100
                        )
                        if st.button("Submit Reasoning", key="real_reason"):
                            if reason.strip():
                                # Store feedback in session state
                                feedback_entry = {
                                    'classification': classification,
                                    'confidence': confidence,
                                    'reasoning': reason,
                                    'timestamp': str(pd.Timestamp.now())
                                }
                                st.session_state.feedback_data.append(feedback_entry)
                                
                                st.success("✅ Thank you for your feedback! Your reasoning has been recorded.")
                                st.markdown(f"""
                                <div class="feedback-box">
                                    <h4>📝 Your Feedback Recorded</h4>
                                    <p><strong>Classification:</strong> {classification.title()}</p>
                                    <p><strong>Confidence:</strong> {confidence:.3f}</p>
                                    <p><strong>Your Reasoning:</strong> {reason}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.warning("Can you tell me the reason to return?")
                    
                    elif classification == "photoshopped":
                        st.warning("✂️ **Edited Image Detected**")
                        st.write("This image appears to have been edited or photoshopped.")
                        
                        # Option to upload unedited version
                        st.subheader("📤 Upload Unedited Version")
                        st.write("If you have the original, unedited version of this image, please upload it:")
                        
                        unedited_file = st.file_uploader(
                            "Upload unedited image",
                            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                            key="unedited_upload"
                        )
                        
                        if unedited_file is not None:
                            unedited_image = Image.open(unedited_file)
                            st.image(unedited_image, caption="Uploaded Unedited Image", width=300)
                            
                            if st.button("Compare Images", key="compare_edited"):
                                with st.spinner("Analyzing unedited image..."):
                                    try:
                                        # Create temporary file for unedited image
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                            tmp_file.write(unedited_file.getvalue())
                                            tmp_path = tmp_file.name
                                        
                                        # Classify the unedited image
                                        unedited_class, unedited_conf, unedited_probs, unedited_analysis = agent.classify_image_with_analysis(tmp_path)
                                        
                                        # Clean up
                                        os.unlink(tmp_path)
                                        
                                        # Display comparison
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.subheader("Original Image")
                                            st.write(f"Classification: {classification}")
                                            st.write(f"Confidence: {confidence:.3f}")
                                        
                                        with col2:
                                            st.subheader("Unedited Image")
                                            st.write(f"Classification: {unedited_class}")
                                            st.write(f"Confidence: {unedited_conf:.3f}")
                                        
                                        # Store comparison data
                                        comparison_entry = {
                                            'original_class': classification,
                                            'original_conf': confidence,
                                            'comparison_class': unedited_class,
                                            'comparison_conf': unedited_conf,
                                            'comparison_type': 'unedited',
                                            'timestamp': str(pd.Timestamp.now())
                                        }
                                        st.session_state.comparison_data.append(comparison_entry)
                                        
                                        # Show differences
                                        st.subheader("📊 Comparison Analysis")
                                        st.markdown(f"""
                                        <div class="comparison-box">
                                            <h4>🔄 Image Comparison Results</h4>
                                            <p><strong>Original Image:</strong> {classification.title()} (Confidence: {confidence:.3f})</p>
                                            <p><strong>Unedited Image:</strong> {unedited_class.title()} (Confidence: {unedited_conf:.3f})</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        if classification != unedited_class:
                                            st.success("✅ The unedited image has a different classification!")
                                        else:
                                            st.info("ℹ️ Both images have the same classification.")
                                        
                                    except Exception as e:
                                        st.error(f"Error analyzing unedited image: {e}")
                    
                    else:  # ai_generated
                        st.error("🤖 **AI-Generated Image Detected**")
                        st.write("This image appears to be generated by artificial intelligence.")
                        
                        # Option to upload real image
                        st.subheader("📤 Upload Real Image")
                        st.write("If you have a real, non-AI-generated image, please upload it:")
                        
                        real_file = st.file_uploader(
                            "Upload real image",
                            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
                            key="real_upload"
                        )
                        
                        if real_file is not None:
                            real_image = Image.open(real_file)
                            st.image(real_image, caption="Uploaded Real Image", width=300)
                            
                            if st.button("Compare Images", key="compare_ai"):
                                with st.spinner("Analyzing real image..."):
                                    try:
                                        # Create temporary file for real image
                                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                                            tmp_file.write(real_file.getvalue())
                                            tmp_path = tmp_file.name
                                        
                                        # Classify the real image
                                        real_class, real_conf, real_probs, real_analysis = agent.classify_image_with_analysis(tmp_path)
                                        
                                        # Clean up
                                        os.unlink(tmp_path)
                                        
                                        # Display comparison
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.subheader("AI-Generated Image")
                                            st.write(f"Classification: {classification}")
                                            st.write(f"Confidence: {confidence:.3f}")
                                        
                                        with col2:
                                            st.subheader("Real Image")
                                            st.write(f"Classification: {real_class}")
                                            st.write(f"Confidence: {real_conf:.3f}")
                                        
                                        # Store comparison data
                                        comparison_entry = {
                                            'original_class': classification,
                                            'original_conf': confidence,
                                            'comparison_class': real_class,
                                            'comparison_conf': real_conf,
                                            'comparison_type': 'real',
                                            'timestamp': str(pd.Timestamp.now())
                                        }
                                        st.session_state.comparison_data.append(comparison_entry)
                                        
                                        # Show differences
                                        st.subheader("📊 Comparison Analysis")
                                        st.markdown(f"""
                                        <div class="comparison-box">
                                            <h4>🔄 Image Comparison Results</h4>
                                            <p><strong>AI-Generated Image:</strong> {classification.title()} (Confidence: {confidence:.3f})</p>
                                            <p><strong>Real Image:</strong> {real_class.title()} (Confidence: {real_conf:.3f})</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        if classification != real_class:
                                            st.success("✅ The real image has a different classification!")
                                        else:
                                            st.info("ℹ️ Both images have the same classification.")
                                        
                                    except Exception as e:
                                        st.error(f"Error analyzing real image: {e}")
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["📈 Bar Charts", "🥧 Pie Charts", "🔍 Details"])
                    
                    with tab1:
                        st.subheader("Probability Charts")
                        
                        # Stage 1 chart
                        if "stage1" in probabilities:
                            fig1 = create_confidence_chart(probabilities["stage1"], "Stage 1")
                            if fig1:
                                st.plotly_chart(fig1, use_container_width=True)
                        
                        # Stage 2 chart (if applicable)
                        if "stage2" in probabilities:
                            fig2 = create_confidence_chart(probabilities["stage2"], "Stage 2")
                            if fig2:
                                st.plotly_chart(fig2, use_container_width=True)
                    
                    with tab2:
                        st.subheader("Probability Distribution")
                        
                        # Stage 1 pie chart
                        if "stage1" in probabilities:
                            pie1 = create_pie_chart(probabilities["stage1"], "Stage 1")
                            if pie1:
                                st.plotly_chart(pie1, use_container_width=True)
                        
                        # Stage 2 pie chart (if applicable)
                        if "stage2" in probabilities:
                            pie2 = create_pie_chart(probabilities["stage2"], "Stage 2")
                            if pie2:
                                st.plotly_chart(pie2, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Detailed Information")
                        
                        # Raw probabilities
                        with st.expander("🔍 Raw Probability Data"):
                            st.json(probabilities)
                        
                        # Threshold analysis
                        with st.expander("🔍 Threshold Analysis"):
                            st.json(analysis)
    
    # Add some sample images for testing
    with col2:
        st.header("📋 Sample Images")
        st.markdown("""
        **Test with these sample images:**
        
        - **Real images**: Photos taken with cameras
        - **AI-generated**: Images created by AI models
        - **Photoshopped**: Edited/retouched images
        
        Upload any image to test the classification system!
        """)
        
        # Add some helpful information
        st.info("""
        **How it works:**
        
        1. **Stage 1**: Classifies as Real vs Not Real
        2. **Stage 2**: If Not Real, classifies as AI-Generated vs Photoshopped
        
        The system uses both image features and metadata for accurate classification.
        """)

if __name__ == "__main__":
    main()
