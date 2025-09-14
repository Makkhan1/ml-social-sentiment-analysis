"""
Data processing and preprocessing module for sentiment analysis
"""
import pandas as pd
import numpy as np
import re
import ssl
import warnings
warnings.filterwarnings('ignore')

# Handle SSL issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Import NLTK and related packages
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import requests

class DataPreprocessor:
    """Main data preprocessing class with robust NLTK handling"""
    
    def __init__(self):
        print("Initializing DataPreprocessor...")
        
        # Download and setup NLTK data first
        self._setup_nltk()
        
        # Initialize components after NLTK setup
        self._initialize_components()
        
        print("✅ DataPreprocessor initialized successfully!")
        
    def _setup_nltk(self):
        """Download and setup all required NLTK data"""
        print("Setting up NLTK data...")
        
        # Required NLTK datasets
        required_datasets = {
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet',
            'punkt': 'tokenizers/punkt',
            'vader_lexicon': 'vader_lexicon',
            'omw-1.4': 'corpora/omw-1.4'
        }
        
        for name, path in required_datasets.items():
            try:
                # Check if already downloaded
                nltk.data.find(path)
                print(f"✅ {name} already available")
            except LookupError:
                print(f"📥 Downloading {name}...")
                try:
                    nltk.download(name, quiet=False)
                    print(f"✅ {name} downloaded successfully")
                except Exception as e:
                    print(f"⚠️  Warning: Could not download {name}: {e}")
    
    def _initialize_components(self):
        """Initialize NLTK components after data is downloaded"""
        try:
            # Import after downloading
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            # Initialize lemmatizer
            self.lemmatizer = WordNetLemmatizer()
            
            # Initialize stopwords
            try:
                self.stop_words = set(stopwords.words('english'))
                print(f"✅ Loaded {len(self.stop_words)} English stopwords")
            except LookupError:
                print("⚠️  Using default stopwords list")
                # Fallback stopwords list
                self.stop_words = {
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
                    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                    'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                    'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                    'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before',
                    'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
                    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                    'don', 'should', 'now'
                }
            
            # Initialize VADER analyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text is None:
            return ""
        
        try:
            # Convert to string and lowercase
            text = str(text).lower()
            
            # Remove URLs, mentions, hashtags
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters and digits but keep basic punctuation
            text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenize and remove stopwords
            tokens = text.split()
            
            # Filter tokens
            filtered_tokens = []
            for token in tokens:
                if len(token) > 2 and token not in self.stop_words:
                    try:
                        # Try to lemmatize
                        lemmatized = self.lemmatizer.lemmatize(token)
                        filtered_tokens.append(lemmatized)
                    except:
                        # If lemmatization fails, use original token
                        filtered_tokens.append(token)
            
            return ' '.join(filtered_tokens)
            
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return str(text) if text else ""
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            if not text or str(text).strip() == "":
                return 'unknown'
            return detect(str(text))
        except (LangDetectException, Exception):
            return 'unknown'
    
    def get_vader_sentiment(self, text):
        """Get sentiment using VADER"""
        try:
            if not text or str(text).strip() == "":
                return 'neutral', 0.0
            
            scores = self.vader_analyzer.polarity_scores(str(text))
            compound = scores['compound']
            
            if compound >= 0.05:
                return 'positive', compound
            elif compound <= -0.05:
                return 'negative', compound
            else:
                return 'neutral', compound
                
        except Exception as e:
            print(f"Error in VADER sentiment analysis: {e}")
            return 'neutral', 0.0
    
    def get_textblob_sentiment(self, text):
        """Get sentiment using TextBlob"""
        try:
            if not text or str(text).strip() == "":
                return 'neutral', 0.0
            
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive', polarity
            elif polarity < -0.1:
                return 'negative', polarity
            else:
                return 'neutral', polarity
                
        except Exception as e:
            print(f"Error in TextBlob sentiment analysis: {e}")
            return 'neutral', 0.0

def load_data_from_url(url):
    """Load data from Google Cloud Storage URL"""
    try:
        print(f"Loading data from: {url}")
        
        # Try to load with different encodings
        try:
            df = pd.read_csv(url, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(url, encoding='latin-1')
            except:
                df = pd.read_csv(url, encoding='cp1252')
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def extract_comments(df, existing_comment_cols, preprocessor):
    """Extract all comments from the dataframe and apply preprocessing"""
    print(f"Processing comments from {len(existing_comment_cols)} comment columns...")
    
    comment_data = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:  # Progress indicator
            print(f"Processing row {idx}/{len(df)}")
            
        for col in existing_comment_cols:
            comment_text = row[col]
            
            if pd.notna(comment_text) and str(comment_text).strip() != "":
                try:
                    original_comment = str(comment_text)
                    cleaned_comment = preprocessor.clean_text(original_comment)
                    
                    if cleaned_comment and len(cleaned_comment.split()) >= 2:
                        language = preprocessor.detect_language(original_comment)
                        vader_sentiment, vader_score = preprocessor.get_vader_sentiment(original_comment)
                        textblob_sentiment, textblob_score = preprocessor.get_textblob_sentiment(original_comment)
                        
                        comment_data.append({
                            'post_id': row.get('Post_ID', f'post_{idx}'),
                            'original_comment': original_comment,
                            'cleaned_comment': cleaned_comment,
                            'comment_length': len(original_comment.split()),
                            'cleaned_length': len(cleaned_comment.split()),
                            'language': language,
                            'vader_sentiment': vader_sentiment,
                            'vader_score': vader_score,
                            'textblob_sentiment': textblob_sentiment,
                            'textblob_score': textblob_score,
                        })
                        
                except Exception as e:
                    print(f"Error processing comment in row {idx}: {e}")
                    continue
    
    print(f"✅ Extracted {len(comment_data)} valid comments")
    return pd.DataFrame(comment_data)

def create_ensemble_sentiment(row):
    """Create ensemble sentiment labels"""
    try:
        vader_sent = row['vader_sentiment']
        textblob_sent = row['textblob_sentiment']
        
        # If both agree, use that sentiment
        if vader_sent == textblob_sent:
            return vader_sent
        
        # If they disagree, use the one with stronger absolute score
        if abs(row['vader_score']) > abs(row['textblob_score']):
            return vader_sent
        else:
            return textblob_sent
            
    except Exception as e:
        print(f"Error creating ensemble sentiment: {e}")
        return 'neutral'
