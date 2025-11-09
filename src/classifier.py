"""
Multinomial Naive Bayes Classifier
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import warnings
warnings.filterwarnings('ignore')


class FakeNewsClassifier:
    """Multinomial Naive Bayes classifier"""
    
    def __init__(self, vectorizer_type='tfidf', max_features=10000, ngram_range=(1, 3), alpha=0.1):
        """
        Initialize the classifier
        
        Args:
            vectorizer_type: 'tfidf' or 'count' for text vectorization
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to extract
            alpha: Smoothing parameter for Multinomial NB
        """
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.alpha = alpha
        
        # Initialize vectorizer with parameters
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=ngram_range,
                min_df=3,
                max_df=0.85,
                sublinear_tf=True,  # Use sublinear tf scaling
                use_idf=True,
                smooth_idf=True
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=ngram_range,
                min_df=3,
                max_df=0.85
            )
        
        # Initialize Multinomial Naive Bayes with tuned alpha
        self.model = MultinomialNB(alpha=alpha)
        
    def preprocess_text(self, text):
        """
        Text preprocessing
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove numbers but keep words with numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_additional_features(self, df):
        """
        Extract additional features from the dataset
        
        Args:
            df: DataFrame with text data
            
        Returns:
            DataFrame with additional features
        """
        # Text length features
        df['text_length'] = df['text'].fillna('').str.len()
        df['title_length'] = df['title'].fillna('').str.len()
        df['word_count'] = df['text'].fillna('').str.split().str.len()
        
        # Punctuation features
        df['exclamation_count'] = df['text'].fillna('').str.count('!')
        df['question_count'] = df['text'].fillna('').str.count('\?')
        df['capital_ratio'] = df['text'].fillna('').apply(
            lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
        )
        
        return df
    
    def load_and_prepare_data(self, filepath):
        """
        Load and prepare the dataset with enhanced preprocessing
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Prepared DataFrame
        """
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nLabel distribution:\n{df['label'].value_counts()}")
        print(f"Label distribution (%):\n{df['label'].value_counts(normalize=True) * 100}")
        
        # Check for class imbalance
        label_counts = df['label'].value_counts()
        imbalance_ratio = label_counts.max() / label_counts.min()
        if imbalance_ratio > 1.5:
            print(f"\n⚠️  Warning: Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        
        # Combine title and text
        print("\nPreprocessing text data...")
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        df['cleaned_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Extract additional features
        df = self.extract_additional_features(df)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 10]
        
        print(f"Dataset shape after cleaning: {df.shape}")
        
        # Data quality checks
        print(f"\nAverage text length: {df['text_length'].mean():.0f} characters")
        print(f"Average word count: {df['word_count'].mean():.0f} words")
        
        return df
    
    def train(self, X_train, y_train):
        """
        Train the model
        
        Args:
            X_train: Training text data
            y_train: Training labels
        """
        print("\nVectorizing text data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        print(f"\nTraining Multinomial Naive Bayes (alpha={self.alpha})...")
        self.model.fit(X_train_vec, y_train)
        
        print("Model training completed!")
        
    def predict(self, X_test):
        """Make predictions"""
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities"""
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X_test_vec)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Text data
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        # Vectorize all data
        X_vec = self.vectorizer.fit_transform(X)
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(self.model, X_vec, y, cv=skf, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def evaluate(self, X_test, y_test, output_dir='outputs'):
        """
        Comprehensive evaluation
        
        Args:
            X_test: Test text data
            y_test: True labels
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        # Determine the positive label (fake/FAKE)
        classes = self.model.classes_
        fake_label = 'FAKE' if 'FAKE' in classes else 'fake'
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=fake_label, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=fake_label, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=fake_label, average='binary')
        
        # ROC AUC
        if len(self.model.classes_) == 2:
            fake_idx = list(self.model.classes_).index(fake_label)
            roc_auc = roc_auc_score(y_test, y_proba[:, fake_idx])
        else:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        # Print results
        print("\n" + "="*70)
        print("MODEL EVALUATION RESULTS")
        print("="*70)
        print(f"Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision:  {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:     {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:   {f1:.4f} ({f1*100:.2f}%)")
        if roc_auc:
            print(f"ROC AUC:    {roc_auc:.4f}")
        print("="*70)
        
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, ['Real', 'Fake'], output_dir)
        
        # ROC curve
        if roc_auc:
            self.plot_roc_curve(y_test, y_proba[:, fake_idx], output_dir)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, classes, output_dir='outputs'):
        """Plot confusion matrix"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Multinomial Naive Bayes', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentages
        total = cm.sum()
        for i in range(len(classes)):
            for j in range(len(classes)):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved as '{output_path}'")
        plt.close()
    
    def plot_roc_curve(self, y_test, y_proba, output_dir='outputs'):
        """Plot ROC curve"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert labels to binary
        y_test_binary = (y_test == 'fake').astype(int)
        
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_proba)
        roc_auc = roc_auc_score(y_test_binary, y_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Multinomial Naive Bayes', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved as '{output_path}'")
        plt.close()
    
    def get_top_features(self, n=20):
        """
        Get top features for each class
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary with top features for each class
        """
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        top_features = {}
        
        for idx, label in enumerate(self.model.classes_):
            log_prob = self.model.feature_log_prob_[idx]
            top_indices = np.argsort(log_prob)[-n:][::-1]
            top_features[label] = list(feature_names[top_indices])
        
        return top_features
    
    def save_model(self, filepath='models/mnb_model.pkl'):
        """Save the trained model"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'vectorizer_type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'alpha': self.alpha
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to '{filepath}'")
    
    def load_model(self, filepath='models/mnb_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.vectorizer_type = model_data['vectorizer_type']
        self.max_features = model_data['max_features']
        self.ngram_range = model_data.get('ngram_range', (1, 2))
        self.alpha = model_data.get('alpha', 1.0)
        
        print(f"Model loaded from '{filepath}'")
    
    def predict_single(self, text):
        """Predict a single news article"""
        cleaned = self.preprocess_text(text)
        prediction = self.predict([cleaned])[0]
        probabilities = self.predict_proba([cleaned])[0]
        
        # Handle both uppercase and lowercase labels
        classes = self.model.classes_
        fake_label = 'FAKE' if 'FAKE' in classes else 'fake'
        real_label = 'REAL' if 'REAL' in classes else 'real'
        
        fake_idx = list(classes).index(fake_label)
        real_idx = list(classes).index(real_label)
        
        return {
            'prediction': prediction,
            'confidence': max(probabilities),
            'probabilities': {
                'fake': probabilities[fake_idx],
                'real': probabilities[real_idx]
            }
        }
