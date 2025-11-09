"""
Model Comparison Script
Compare different classifiers for fake news detection
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import FakeNewsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def compare_algorithms(X_train, X_test, y_train, y_test, output_dir='outputs'):
    """
    Compare different machine learning algorithms
    
    Args:
        X_train, X_test: Training and test text data
        y_train, y_test: Training and test labels
        output_dir: Directory to save outputs
    """
    print("="*70)
    print("ALGORITHM COMPARISON")
    print("="*70)
    
    # Vectorize data
    print("\nVectorizing data with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        stop_words='english',
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_vec.shape}")
    
    # Define models to compare
    models = {
        'Multinomial NB (α=0.1)': MultinomialNB(alpha=0.1),
        'Multinomial NB (α=0.5)': MultinomialNB(alpha=0.5),
        'Multinomial NB (α=1.0)': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Linear SVM': LinearSVC(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = []
    
    print("\nTraining and evaluating models...")
    print("-" * 70)
    
    for name, model in models.items():
        print(f"\n{name}:")
        
        # Train
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Determine the positive label (FAKE or fake)
        unique_labels = set(y_test.unique())
        fake_label = 'FAKE' if 'FAKE' in unique_labels else 'fake'
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=fake_label, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=fake_label, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=fake_label, average='binary')
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Plot comparison
    plot_model_comparison(results_df, output_dir)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"\nResults saved to '{output_dir}/model_comparison.csv'")
    
    return results_df


def plot_model_comparison(results_df, output_dir='outputs'):
    """Plot model comparison"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Comparison - Fake News Classification', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        sorted_df = results_df.sort_values(metric, ascending=True)
        
        bars = ax.barh(sorted_df['Model'], sorted_df[metric], color=color, alpha=0.7)
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as '{output_path}'")
    plt.close()


def compare_hyperparameters(X_train, X_test, y_train, y_test, output_dir='outputs'):
    """
    Compare different hyperparameter settings for Multinomial NB
    
    Args:
        X_train, X_test: Training and test text data
        y_train, y_test: Training and test labels
        output_dir: Directory to save outputs
    """
    print("\n" + "="*70)
    print("HYPERPARAMETER COMPARISON (Multinomial Naive Bayes)")
    print("="*70)
    
    results = []
    
    # Test different configurations
    configs = [
        {'max_features': 5000, 'ngram_range': (1, 1), 'alpha': 1.0},
        {'max_features': 5000, 'ngram_range': (1, 2), 'alpha': 1.0},
        {'max_features': 10000, 'ngram_range': (1, 2), 'alpha': 1.0},
        {'max_features': 10000, 'ngram_range': (1, 3), 'alpha': 1.0},
        {'max_features': 10000, 'ngram_range': (1, 3), 'alpha': 0.5},
        {'max_features': 10000, 'ngram_range': (1, 3), 'alpha': 0.1},
    ]
    
    for config in configs:
        print(f"\nTesting: max_features={config['max_features']}, "
              f"ngram_range={config['ngram_range']}, alpha={config['alpha']}")
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            stop_words='english',
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train
        model = MultinomialNB(alpha=config['alpha'])
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        
        # Determine the positive label (FAKE or fake)
        unique_labels = set(y_test.unique())
        fake_label = 'FAKE' if 'FAKE' in unique_labels else 'fake'
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=fake_label, average='binary')
        
        config_str = f"feat={config['max_features']}, n-gram={config['ngram_range']}, α={config['alpha']}"
        
        results.append({
            'Configuration': config_str,
            'Accuracy': accuracy,
            'F1-Score': f1
        })
        
        print(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\n" + "="*70)
    print("HYPERPARAMETER COMPARISON SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'hyperparameter_comparison.csv'), index=False)
    
    return results_df


def main():
    """Main execution function"""
    
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    
    # Configuration
    DATA_PATH = r'data\cleaned\news.csv'
    OUTPUT_DIR = r'outputs'
    
    # Load data
    print("\nLoading and preprocessing data...")
    classifier = FakeNewsClassifier()
    df = classifier.load_and_prepare_data(DATA_PATH)
    
    # Prepare data
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train):,} samples")
    print(f"  Testing:  {len(X_test):,} samples")
    
    # Compare algorithms
    algo_results = compare_algorithms(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
    # Compare hyperparameters
    hyper_results = compare_hyperparameters(X_train, X_test, y_train, y_test, OUTPUT_DIR)
    
    # Best configuration
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    best_model = algo_results.iloc[0]
    print(f"\nBest Algorithm: {best_model['Model']}")
    print(f"  Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
    print(f"  F1-Score: {best_model['F1-Score']:.4f}")
    
    best_config = hyper_results.iloc[0]
    print(f"\nBest Hyperparameters: {best_config['Configuration']}")
    print(f"  Accuracy: {best_config['Accuracy']:.4f} ({best_config['Accuracy']*100:.2f}%)")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
