"""
Sentiment Analysis Dashboard - Home Page
Complete ML Pipeline with CNN Model
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from data.processor import DataPreprocessor, load_data_from_url
    from models.trainer import (
        Vocabulary, SentimentDataset, CNNSentimentClassifier,
        ModelTrainer, device
    )
    from models.inference import predict_sentiment, evaluate_model_comprehensive
except Exception as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎭 Sentiment Analysis Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        margin-top: -80px;
    }
    .css-1d391kg {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_data
def load_model_and_vocab():
    """Load trained model and vocabulary"""
    try:
        # Get current working directory and project root
        current_dir = os.getcwd()
        
        # Try different possible paths for the model files
        possible_paths = [
            # If running from ml-project/app/
            ('../models/vocabulary.pkl', '../models/best_cnn_model.pth'),
            # If running from ml-project/
            ('models/vocabulary.pkl', 'models/best_cnn_model.pth'),
            # If files are in root directory
            ('vocabulary.pkl', 'best_cnn_model.pth'),
            # If files are in current directory
            ('./vocabulary.pkl', './best_cnn_model.pth')
        ]
        
        vocab = None
        model = None
        
        # Try each path combination
        for vocab_path, model_path in possible_paths:
            try:
                     
                if os.path.exists(vocab_path) and os.path.exists(model_path):
                    # Load vocabulary
                    with open(vocab_path, 'rb') as f:
                        vocab = pickle.load(f)
                    
                    # Load model
                    model = CNNSentimentClassifier(
                             vocab_size=len(vocab),
                              embed_dim=64,      # ← Changed from 128 to 64
                             output_dim=3,
                             filter_sizes=[3, 4, 5],
                             num_filters=50,    # ← Changed from 100 to 50
                             dropout=0.3
                          )
                    
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model.eval()
                    
                    st.success(f"✅ Model loaded from: {model_path}")
                    st.success(f"✅ Vocabulary loaded from: {vocab_path}")
                    return model, vocab, True
                    
            except Exception as e:
                   
                continue
        
        # If we get here, no paths worked
        st.error("❌ Could not find model files in any expected location")
        
        # Show what files actually exist
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(('.pth', '.pkl', '.json')):
                    st.write(f"  {os.path.join(root, file)}")
        
        return None, None, False
        
    except Exception as e:
        st.error(f"Error in load_model_and_vocab: {e}")
        return None, None, False

@st.cache_data
def load_sample_data():
    """Load sample data for testing"""
    try:
        df = load_data_from_url()
        return df.head(100), True
    except:
        # Fallback sample data
        sample_texts = [
            "I love this product! It's amazing!",
            "This is okay, nothing special.",
            "I hate this. Worst experience ever.",
            "The service was fantastic and the staff was very helpful.",
            "Not bad, could be better though.",
            "Absolutely terrible quality. Would not recommend.",
            "Great value for money. Highly satisfied!",
            "It's alright, meets basic expectations.",
            "Disappointing product. Poor quality control.",
            "Excellent customer service and fast delivery!"
        ]
        return pd.DataFrame({'text': sample_texts}), True

def predict_sentiment_advanced(text, model, vocab):
    """Advanced sentiment prediction with confidence"""
    try:
        # Basic prediction
        sentiment, probabilities = predict_sentiment(model, text, vocab, max_length=50, device=device)
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        # Create detailed results
        results = {
            'sentiment': sentiment,
            'confidence': confidence,
            'probabilities': {
                'negative': probabilities[0],
                'neutral': probabilities[1], 
                'positive': probabilities[2]
            }
        }
        
        return results
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None
def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predict Sentiment", "📈 Model Analytics", "🧪 Batch Analysis"]
    )
    
    # Load model and vocab
    with st.spinner("Loading model and vocabulary..."):
        model, vocab, model_loaded = load_model_and_vocab()
    
    if not model_loaded:
        st.error("❌ Could not load model. Please check if model files exist.")
        st.info("💡 Run the training pipeline first: `python src/main.py`")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "🔮 Predict Sentiment":
        show_prediction_page(model, vocab)
    elif page == "📈 Model Analytics":
        show_analytics_page()
    elif page == "🧪 Batch Analysis":
        show_batch_analysis_page(model, vocab)

def show_home_page():
    """Home page content"""
    st.subheader("Welcome to the Sentiment Analysis Dashboard! 🎉")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 What This App Does:
        - **Real-time sentiment analysis** using a trained CNN model
        - **Batch processing** for multiple texts
        - **Model performance analytics** and visualizations
        - **Interactive predictions** with confidence scores
        
        ### 🧠 Model Details:
        - **Architecture:** Convolutional Neural Network (CNN)
        - **Classes:** Positive, Neutral, Negative
        - **Features:** Word embeddings, multiple filter sizes
        - **Training Data:** Social media comments and posts
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 How to Use:
        1. **Navigate** using the sidebar menu
        2. **Single Prediction:** Use "🔮 Predict Sentiment" page
        3. **Batch Analysis:** Upload CSV or paste multiple texts
        4. **Analytics:** View model performance metrics
        
        ### 📊 Features:
        - ✅ Real-time predictions
        - ✅ Confidence scores
        - ✅ Probability distributions
        - ✅ Batch processing
        - ✅ Data visualization
        """)
    
    # Quick test section
    st.subheader("🎯 Quick Test")
    test_text = st.text_input("Enter text to analyze:", "I love this application!")
    
    if st.button("🔍 Analyze") and test_text:
        with st.spinner("Analyzing..."):
            model, vocab, _ = load_model_and_vocab()
            result = predict_sentiment_advanced(test_text, model, vocab)
            
            if result:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sentiment", result['sentiment'].title())
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']:.2%}")
                
                with col3:
                    sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😢"}
                    st.metric("Result", sentiment_emoji.get(result['sentiment'], "🤔"))
def show_prediction_page(model, vocab):
    """Sentiment prediction page"""
    st.subheader("🔮 Real-time Sentiment Prediction")
    
    # Input methods
    input_method = st.radio("Choose input method:", ["✏️ Type Text", "📁 Upload File"])
    
    if input_method == "✏️ Type Text":
        # Text input
        user_text = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type your text here... (e.g., 'I love this product!')"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
        
        if analyze_button and user_text:
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment_advanced(user_text, model, vocab)
                
                if result:
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        sentiment_colors = {"positive": "green", "neutral": "orange", "negative": "red"}
                        st.metric(
                            "Predicted Sentiment", 
                            result['sentiment'].title(),
                            delta_color=sentiment_colors.get(result['sentiment'], "gray")
                        )
                    
                    with col2:
                        st.metric("Confidence Score", f"{result['confidence']:.2%}")
                    
                    with col3:
                        st.metric("Text Length", f"{len(user_text.split())} words")
                    
                    with col4:
                        sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😢"}
                        st.metric("Mood", sentiment_emoji.get(result['sentiment'], "🤔"))
                    
                    # Probability distribution
                    st.subheader("📈 Probability Distribution")
                    
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Neutral', 'Positive'],
                        'Probability': [
                            result['probabilities']['negative'],
                            result['probabilities']['neutral'],
                            result['probabilities']['positive']
                        ]
                    })
                    
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
                    bars = ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=colors, alpha=0.8)
                    
                    # Customize chart
                    ax.set_ylabel('Probability')
                    ax.set_title('Sentiment Probability Distribution')
                    ax.set_ylim(0, 1)
                    
                    # Add value labels on bars
                    for bar, prob in zip(bars, prob_df['Probability']):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.3f}', ha='center', va='bottom')
                    
                    st.pyplot(fig)
                    
                    # Confidence interpretation
                    st.subheader("🎯 Confidence Interpretation")
                    confidence = result['confidence']
                    
                    if confidence >= 0.8:
                        st.success(f"🎯 **High Confidence** ({confidence:.2%}) - The model is very confident in this prediction.")
                    elif confidence >= 0.6:
                        st.warning(f"⚖️ **Medium Confidence** ({confidence:.2%}) - The model is moderately confident.")
                    else:
                        st.info(f"🤔 **Low Confidence** ({confidence:.2%}) - The model is uncertain. Consider the context.")
    
    else:  # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                text_column = st.selectbox("Select text column:", df.columns)
                
                if st.button("🔍 Analyze All"):
                    with st.spinner("Analyzing all texts..."):
                        results = []
                        for text in df[text_column]:
                            result = predict_sentiment_advanced(str(text), model, vocab)
                            if result:
                                results.append({
                                    'text': text,
                                    'sentiment': result['sentiment'],
                                    'confidence': result['confidence']
                                })
                        
                        results_df = pd.DataFrame(results)
                        st.subheader("📊 Batch Analysis Results")
                        st.dataframe(results_df)
                        
                        # Summary statistics
                        st.subheader("📈 Summary Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sentiment_counts = results_df['sentiment'].value_counts()
                            fig, ax = plt.subplots()
                            sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
                            ax.set_title('Sentiment Distribution')
                            st.pyplot(fig)
                        
                        with col2:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Average Confidence", f"{avg_confidence:.2%}")
                            
                            st.write("**Confidence Distribution:**")
                            fig, ax = plt.subplots()
                            ax.hist(results_df['confidence'], bins=20, alpha=0.7)
                            ax.set_xlabel('Confidence Score')
                            ax.set_ylabel('Frequency')
                            ax.set_title('Confidence Score Distribution')
                            st.pyplot(fig)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")
def show_analytics_page():
    """Model analytics and performance page"""
    st.subheader("📈 Model Performance Analytics")
    
    try:
        # Load results if available
        with open('models/pipeline_results.json', 'r') as f:
            results = json.load(f)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Test Accuracy", f"{results.get('test_accuracy', 0):.2%}")
        
        with col2:
            st.metric("F1-Score (Weighted)", f"{results.get('test_f1_weighted', 0):.3f}")
        
        with col3:
            st.metric("F1-Score (Macro)", f"{results.get('test_f1_macro', 0):.3f}")
        
        with col4:
            st.metric("Training Epochs", results.get('epochs_trained', 'N/A'))
        
        # Model architecture info
        st.subheader("🏗️ Model Architecture")
        st.code("""
CNN Sentiment Classifier:
├── Embedding Layer (vocab_size → 128)
├── Convolutional Layers
│   ├── Conv1D (filter_size=3, filters=100)
│   ├── Conv1D (filter_size=4, filters=100)
│   └── Conv1D (filter_size=5, filters=100)
├── Max Pooling
├── Dropout (0.3)
└── Dense Layer (300 → 3)

Total Parameters: ~1.2M
Device: CPU/GPU
        """)
        
        # Training history if available
        if 'training_history' in results:
            st.subheader("📊 Training History")
            
            history = results['training_history']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss plot
                epochs = list(range(1, len(history['train_losses']) + 1))
                fig, ax = plt.subplots()
                ax.plot(epochs, history['train_losses'], label='Training Loss', color='blue')
                ax.plot(epochs, history['val_losses'], label='Validation Loss', color='red')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
            
            with col2:
                # Accuracy plot
                fig, ax = plt.subplots()
                ax.plot(epochs, history['val_accuracies'], label='Validation Accuracy', color='green')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('Validation Accuracy')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)
    
    except FileNotFoundError:
        st.warning("📋 No training results found. Train the model first using `python src/main.py`")
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

def show_batch_analysis_page(model, vocab):
    """Batch analysis page"""
    st.subheader("🧪 Batch Sentiment Analysis")
    
    # Sample data option
    if st.button("📋 Load Sample Data"):
        sample_df, success = load_sample_data()
        if success:
            st.session_state['batch_data'] = sample_df
            st.success("✅ Sample data loaded!")
    
    # Manual text input
    st.subheader("✏️ Manual Input")
    texts_input = st.text_area(
        "Enter multiple texts (one per line):",
        height=200,
        placeholder="Enter each text on a new line...\nExample:\nI love this!\nThis is okay.\nI don't like it."
    )
    
    if st.button("🔍 Analyze Texts") and texts_input:
        texts = [text.strip() for text in texts_input.split('\n') if text.strip()]
        
        with st.spinner(f"Analyzing {len(texts)} texts..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, text in enumerate(texts):
                result = predict_sentiment_advanced(text, model, vocab)
                if result:
                    results.append({
                        'Text': text[:100] + '...' if len(text) > 100 else text,
                        'Sentiment': result['sentiment'],
                        'Confidence': result['confidence']
                    })
                
                progress_bar.progress((i + 1) / len(texts))
            
            if results:
                results_df = pd.DataFrame(results)
                
                st.subheader("📊 Batch Results")
                st.dataframe(results_df)
                
                # Summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    sentiment_counts = results_df['Sentiment'].value_counts()
                    st.write("**Sentiment Distribution:**")
                    for sentiment, count in sentiment_counts.items():
                        percentage = (count / len(results)) * 100
                        st.write(f"- {sentiment.title()}: {count} ({percentage:.1f}%)")
                
                with col2:
                    avg_confidence = results_df['Confidence'].mean()
                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                    
                    high_conf = (results_df['Confidence'] >= 0.8).sum()
                    st.write(f"**High Confidence (≥80%):** {high_conf}/{len(results)}")
                
                with col3:
                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sentiment_counts.plot(kind='bar', ax=ax, color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
                    ax.set_title('Sentiment Distribution')
                    ax.set_xlabel('Sentiment')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
