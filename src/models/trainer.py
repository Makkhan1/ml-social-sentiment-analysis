"""
CNN Model and Training utilities for sentiment analysis
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Vocabulary:
    """Vocabulary class for text encoding"""
    def __init__(self, texts, min_freq=2, max_vocab_size=None):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.build_vocab(texts)
    
    def build_vocab(self, texts):
        word_freq = {}
        for text in texts:
            for word in str(text).split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        idx = 2
        for word, freq in sorted_words:
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                if self.max_vocab_size and idx >= self.max_vocab_size:
                    break
    
    def encode(self, text):
        return [self.word2idx.get(word, 1) for word in str(text).split()]
    
    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices if idx != 0])
    
    def __len__(self):
        return len(self.word2idx)

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis"""
    def __init__(self, texts, labels, vocab, max_length):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self.vocab.encode(text)
        
        # Truncate or pad to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded.extend([0] * (self.max_length - len(encoded)))
        
        return {
            'text': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(self.label_map[label], dtype=torch.long),
            'length': torch.tensor(min(len(encoded), self.max_length), dtype=torch.long)
        }

class CNNSentimentClassifier(nn.Module):
    """CNN Model for sentiment classification"""
    def __init__(self, vocab_size, embed_dim, output_dim, filter_sizes=[3, 4, 5], num_filters=100, dropout=0.3):
        super(CNNSentimentClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Convolutional layers with different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=size) 
            for size in filter_sizes
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, x, lengths):
        # x: [batch_size, seq_len]
        
        # Embedding
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Transpose for Conv1d: [batch_size, embed_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution + ReLU activation
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, conv_seq_len]
            
            # Max pooling over the sequence dimension
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate all max-pooled outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Apply dropout
        concatenated = self.dropout(concatenated)
        
        # Final classification layer
        output = self.fc(concatenated)
        
        return output

class ModelTrainer:
    """Model training utilities"""
    def __init__(self, model, device, train_loader, val_loader):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in self.train_loader:
            texts = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            lengths = batch['length']
            
            optimizer.zero_grad()
            outputs = self.model(texts, lengths)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * texts.size(0)
            total_samples += texts.size(0)
        
        return total_loss / total_samples
    
    def validate_epoch(self, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                texts = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                lengths = batch['length']
                
                outputs = self.model(texts, lengths)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                
                total_loss += loss.item() * texts.size(0)
                total_samples += texts.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples * 100
        
        return avg_loss, accuracy
    
    def train(self, optimizer, criterion, num_epochs, patience=5):
        """Train the model with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\nStarting training...")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(optimizer, criterion)
            val_loss, val_accuracy = self.validate_epoch(criterion)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_cnn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        self.model.load_state_dict(torch.load('models/best_cnn_model.pth'))
        print("Training finished. Best model loaded.")
        return self.train_losses, self.val_losses, self.val_accuracies

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    texts = torch.stack([item['text'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    
    return {
        'text': texts,
        'label': labels,
        'length': lengths
    }
