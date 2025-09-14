"""
Exploratory Data Analysis utilities
"""
"""
Exploratory Data Analysis utilities for sentiment analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup_plotting():
    """Setup plotting configuration"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_initial_data_exploration(df, existing_comment_cols):
    """Create initial data exploration plots"""
    setup_plotting()
    
    # Calculate total comments per post
    df['total_comments'] = df[existing_comment_cols].notna().sum(axis=1)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Initial Data Distributions', fontsize=16)

    # Platform Distribution
    df['Platform'].value_counts().plot(kind='bar', ax=axes[0,0], 
                                      color=sns.color_palette("viridis", len(df['Platform'].unique())))
    axes[0,0].set_title('Platform Distribution')
    axes[0,0].set_xlabel('Platform')
    axes[0,0].set_ylabel('Number of Posts')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Original Sentiment Score Distribution
    df['Sentiment_Score'].value_counts().plot(kind='bar', ax=axes[0,1], 
                                             color=sns.color_palette("plasma", len(df['Sentiment_Score'].unique())))
    axes[0,1].set_title('Original Sentiment Score Distribution')
    axes[0,1].set_xlabel('Sentiment Score')
    axes[0,1].set_ylabel('Number of Posts')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Comments per Post Distribution
    df['total_comments'].value_counts().sort_index().plot(kind='bar', ax=axes[0,2], 
                                                          color=sns.color_palette("magma", len(df['total_comments'].unique())))
    axes[0,2].set_title('Number of Comments per Post')
    axes[0,2].set_xlabel('Number of Comments')
    axes[0,2].set_ylabel('Number of Posts')

    # Engagement Rate Distribution
    sns.histplot(df['Engagement_Rate'], bins=30, kde=True, ax=axes[1,0], color='skyblue')
    axes[1,0].set_title('Engagement Rate Distribution')
    axes[1,0].set_xlabel('Engagement Rate')
    axes[1,0].set_ylabel('Frequency')

    # Number of Likes Distribution
    sns.histplot(df['Number_of_Likes'], bins=30, kde=True, ax=axes[1,1], color='lightcoral')
    axes[1,1].set_title('Number of Likes Distribution')
    axes[1,1].set_xlabel('Number of Likes')
    axes[1,1].set_ylabel('Frequency')

    # Engagement Rate vs. Sentiment Score
    sns.boxplot(x='Sentiment_Score', y='Engagement_Rate', data=df, ax=axes[1,2], palette='viridis')
    axes[1,2].set_title('Engagement Rate by Original Sentiment Score')
    axes[1,2].set_xlabel('Sentiment Score')
    axes[1,2].set_ylabel('Engagement Rate')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_preprocessing_insights(comment_df, english_comments):
    """Plot preprocessing insights"""
    setup_plotting()
    
    # Language Distribution
    plt.figure(figsize=(10, 6))
    comment_df['language'].value_counts().plot(kind='bar', color=sns.color_palette("tab10"))
    plt.title('Language Distribution of All Comments')
    plt.xlabel('Language')
    plt.ylabel('Number of Comments')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Sentiment Distribution Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('Sentiment Distribution Comparison', fontsize=16)

    comment_df['vader_sentiment'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('VADER Sentiment')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Number of Comments')
    axes[0].tick_params(axis='x', rotation=45)

    comment_df['textblob_sentiment'].value_counts().plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('TextBlob Sentiment')
    axes[1].set_xlabel('Sentiment')
    axes[1].tick_params(axis='x', rotation=45)

    english_comments['final_sentiment'].value_counts().plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Ensemble Final Sentiment (English Comments)')
    axes[2].set_xlabel('Sentiment')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_data_splits(y_train, y_val, y_test):
    """Plot data split verification"""
    setup_plotting()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle('Sentiment Distribution Across Data Splits', fontsize=16)

    # Train set distribution
    pd.Series(y_train).value_counts(normalize=True).plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Train Set')
    axes[0].set_xlabel('Sentiment')
    axes[0].set_ylabel('Proportion')
    axes[0].tick_params(axis='x', rotation=45)

    # Validation set distribution
    pd.Series(y_val).value_counts(normalize=True).plot(kind='bar', ax=axes[1], color='lightcoral')
    axes[1].set_title('Validation Set')
    axes[1].set_xlabel('Sentiment')
    axes[1].tick_params(axis='x', rotation=45)

    # Test set distribution
    pd.Series(y_test).value_counts(normalize=True).plot(kind='bar', ax=axes[2], color='lightgreen')
    axes[2].set_title('Test Set')
    axes[2].set_xlabel('Sentiment')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_comment_length_distribution(english_comments):
    """Plot comment length distributions"""
    setup_plotting()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(english_comments['comment_length'], bins=50, color='blue', 
                 label='Original Length', kde=True, stat='density', alpha=0.5)
    sns.histplot(english_comments['cleaned_length'], bins=50, color='red', 
                 label='Cleaned Length', kde=True, stat='density', alpha=0.5)
    plt.title('Distribution of Comment Lengths (English Comments)')
    plt.xlabel('Number of Words')
    plt.ylabel('Density')
    plt.legend()
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()
