"""
Training Script
Train the Multinomial Naive Bayes model with tuned parameters
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.classifier import FakeNewsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main execution function"""
    
    print("="*70)
    print("MULTINOMIAL NAIVE BAYES FAKE NEWS CLASSIFIER")
    print("="*70)
    
    # Configuration
    DATA_PATH = r'data\cleaned\news.csv'
    MODEL_PATH = r'models\mnb_model.pkl'
    OUTPUT_DIR = r'outputs'
    
    # Hyperparameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    VECTORIZER_TYPE = 'tfidf'
    MAX_FEATURES = 10000      # Increased from 5000
    NGRAM_RANGE = (1, 3)      # Trigrams included
    ALPHA = 0.1               # Reduced smoothing
    
    print(f"\nConfiguration:")
    print(f"  Vectorizer: {VECTORIZER_TYPE.upper()}")
    print(f"  Max Features: {MAX_FEATURES:,}")
    print(f"  N-gram Range: {NGRAM_RANGE}")
    print(f"  Alpha (smoothing): {ALPHA}")
    print(f"  Test Size: {TEST_SIZE*100:.0f}%")
    
    # Initialize classifier
    classifier = FakeNewsClassifier(
        vectorizer_type=VECTORIZER_TYPE,
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        alpha=ALPHA
    )
    
    # Load and prepare data
    df = classifier.load_and_prepare_data(DATA_PATH)
    
    # Split data
    print(f"\nSplitting data (stratified split)...")
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"Training set size: {len(X_train):,}")
    print(f"Test set size: {len(X_test):,}")
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Cross-validation on training data
    print("\n" + "="*70)
    print("CROSS-VALIDATION")
    print("="*70)
    cv_scores = classifier.cross_validate(X_train, y_train, cv=5)
    
    # Evaluate on test set
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    metrics = classifier.evaluate(X_test, y_test, output_dir=OUTPUT_DIR)
    
    # Feature importance
    print("\n" + "="*70)
    print("TOP PREDICTIVE FEATURES")
    print("="*70)
    top_features = classifier.get_top_features(n=15)
    
    for label, features in top_features.items():
        print(f"\nTop 15 features for '{label.upper()}' news:")
        for i, feature in enumerate(features, 1):
            print(f"  {i:2d}. {feature}")
    
    # Save model
    classifier.save_model(MODEL_PATH)
    
    # Sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    # Test on a few examples
    sample_indices = X_test.index[:3]
    for idx in sample_indices:
        sample_text = df.loc[idx, 'combined_text'][:150] + "..."
        actual_label = y_test.loc[idx]
        
        result = classifier.predict_single(df.loc[idx, 'combined_text'])
        
        print(f"\nSample: {sample_text}")
        print(f"Actual: {actual_label.upper()} | Predicted: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.2%} | Fake: {result['probabilities']['fake']:.2%}, Real: {result['probabilities']['real']:.2%}")
        print("-" * 70)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"+ Model trained with {len(X_train):,} samples")
    print(f"+ Test accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"+ F1-Score: {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"+ ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"+ Model saved to: {MODEL_PATH}")
    print(f"+ Visualizations saved to: {OUTPUT_DIR}/")
    print("="*70)
    
    # Performance assessment
    if metrics['accuracy'] >= 0.85:
        print("\n*** EXCELLENT! Model performance is publication-ready (>=85% accuracy) ***")
    elif metrics['accuracy'] >= 0.75:
        print("\n+ Good! Model performance is acceptable (>=75% accuracy)")
        print("   Consider further optimization for better results")
    else:
        print("\n! Model performance needs improvement (<75% accuracy)")
        print("   Recommendations:")
        print("   1. Run: python scripts/optimize.py")
        print("   2. Try different preprocessing techniques")
        print("   3. Consider ensemble methods")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    

if __name__ == "__main__":
    main()
