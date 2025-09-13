"""
Streamlit web app for CNN sentiment analysis.
Author: Mahtab (Project Lead)
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Dict, List

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page setup
st.set_page_config(
    page_title="CNN Sentiment Analysis",
    page_icon="🎭",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2e86ab;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if our API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def predict_text(text: str):
    """Send text to API for prediction."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}

def predict_batch(texts: List[str]):
    """Send multiple texts to API for prediction."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"texts": texts},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code} - {response.text}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 CNN Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check API status
    api_healthy, health_info = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ API is not running!")
        st.info("Please start the API server first:")
        st.code("python app/backend.py", language="bash")
        st.stop()
    
    # Show API status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("✅ API is running")
    with col2:
        if health_info and health_info.get('model_loaded'):
            st.success("✅ Model loaded")
        else:
            st.warning("⚠️ Model not loaded")
    with col3:
        if health_info and health_info.get('models_available'):
            st.success("✅ Modules available")
        else:
            st.warning("⚠️ Waiting for Shiv's modules")
    
    st.markdown("---")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Single Text", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("🔍 Single Text Analysis")
        
        # Text input
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your text here... (e.g., 'I love this product!')",
            height=100
        )
        
        # Predict button
        if st.button("🎯 Analyze Sentiment", type="primary"):
            if user_text.strip():
                with st.spinner("Analyzing sentiment..."):
                    result = predict_text(user_text)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Display results
                        sentiment = result['predicted_sentiment']
                        confidence = result['confidence']
                        
                        # Emoji mapping
                        emoji_map = {
                            'positive': '😊',
                            'neutral': '😐',
                            'negative': '😞'
                        }
                        
                        # Color mapping
                        color_map = {
                            'positive': '#28a745',
                            'neutral': '#ffc107',
                            'negative': '#dc3545'
                        }
                        
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>{emoji_map.get(sentiment, '🤖')} Predicted Sentiment: 
                            <span style="color: {color_map.get(sentiment, '#000')}">{sentiment.title()}</span></h3>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show probability distribution
                        if 'probability_distribution' in result:
                            st.subheader("📊 Probability Distribution")
                            prob_dist = result['probability_distribution']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Positive", f"{prob_dist.get('positive', 0):.1%}")
                            with col2:
                                st.metric("Neutral", f"{prob_dist.get('neutral', 0):.1%}")
                            with col3:
                                st.metric("Negative", f"{prob_dist.get('negative', 0):.1%}")
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("📊 Batch Text Analysis")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Area", "Upload File"],
            horizontal=True
        )
        
        texts_to_analyze = []
        
        if input_method == "Text Area":
            batch_text = st.text_area(
                "Enter multiple texts (one per line):",
                placeholder="Enter each text on a new line...\nExample: I love this!\nExample: This is terrible.\nExample: It's okay.",
                height=150
            )
            
            if batch_text.strip():
                texts_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt'],
                help="Upload a .txt file with one text per line"
            )
            
            if uploaded_file:
                content = uploaded_file.read().decode('utf-8')
                texts_to_analyze = [line.strip() for line in content.split('\n') if line.strip()]
        
        if texts_to_analyze:
            st.info(f"📝 Ready to analyze {len(texts_to_analyze)} texts")
            
            if st.button("🎯 Analyze All", type="primary"):
                if len(texts_to_analyze) > 100:
                    st.warning("Too many texts! Limiting to first 100.")
                    texts_to_analyze = texts_to_analyze[:100]
                
                with st.spinner(f"Analyzing {len(texts_to_analyze)} texts..."):
                    result = predict_batch(texts_to_analyze)
                    
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Display summary
                        st.subheader("📈 Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Texts", result['total_count'])
                        
                        summary = result.get('sentiment_summary', {})
                        with col2:
                            st.metric("Positive", summary.get('positive', 0))
                        with col3:
                            st.metric("Neutral", summary.get('neutral', 0))
                        with col4:
                            st.metric("Negative", summary.get('negative', 0))
                        
                        # Display detailed results
                        st.subheader("📋 Detailed Results")
                        predictions = result.get('predictions', [])
                        
                        if predictions:
                            # Create DataFrame for better display
                            df_results = []
                            for pred in predictions:
                                df_results.append({
                                    'Text': pred['text'][:100] + '...' if len(pred['text']) > 100 else pred['text'],
                                    'Sentiment': pred['predicted_sentiment'],
                                    'Confidence': f"{pred['confidence']:.1%}"
                                })
                            
                            df = pd.DataFrame(df_results)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv"
                            )
    
    with tab3:
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 CNN Sentiment Analysis Project
        
        This is a machine learning project that analyzes the sentiment of text using a Convolutional Neural Network (CNN).
        
        **Team Members:**
        - **Mahtab** - Project Lead & API Development 
        - **Sachin** - Data Processing & EDA
        - **Shiv** - Model Training & Inference
        
        **Features:**
        - ✅ Single text sentiment analysis
        - ✅ Batch text processing
        - ✅ Real-time predictions via API
        - ✅ Interactive web interface
        
        **Tech Stack:**
        - **Backend:** FastAPI
        - **Frontend:** Streamlit  
        - **ML Framework:** PyTorch
        - **Data Processing:** pandas, NLTK
        """)
        
        # API testing section
        st.subheader("🔧 API Testing")
        if st.button("Test API Connection"):
            try:
                response = requests.get(f"{API_BASE_URL}/test")
                if response.status_code == 200:
                    st.success("✅ API connection successful!")
                    st.json(response.json())
                else:
                    st.error(f"❌ API test failed: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    main()
