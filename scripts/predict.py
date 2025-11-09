"""
Prediction Script
Use this script to predict whether a news article is real or fake using the trained model
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.improved_classifier import ImprovedFakeNewsClassifier


def predict_news(text, model_path='models/improved_mnb_model.pkl'):
    """
    Predict if a news article is real or fake
    
    Args:
        text: News article text (title + content)
        model_path: Path to the saved model
        
    Returns:
        Prediction result dictionary
    """
    # Load the trained model
    classifier = ImprovedFakeNewsClassifier()
    
    try:
        classifier.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running: python scripts/train.py")
        return None
    
    # Make prediction
    result = classifier.predict_single(text)
    
    return result


def main():
    """Main execution function for interactive predictions"""
    
    print("="*60)
    print("FAKE NEWS PREDICTOR")
    print("="*60)
    print("\nThis tool uses Multinomial Naive Bayes to classify news articles")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        print("-" * 60)
        text = input("\nEnter news article text (or 'quit' to exit): ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not text:
            print("Please enter some text!")
            continue
        
        # Make prediction
        result = predict_news(text)
        
        if result:
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nProbabilities:")
            print(f"  - Fake News: {result['probabilities']['fake']:.2%}")
            print(f"  - Real News: {result['probabilities']['real']:.2%}")
            print("="*60)


if __name__ == "__main__":
    # Check if text is provided as command line argument
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        result = predict_news(text)
        
        if result:
            print("\n" + "="*60)
            print("PREDICTION RESULT")
            print("="*60)
            print(f"Text: {text[:200]}...")
            print(f"\nPrediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"\nProbabilities:")
            print(f"  - Fake News: {result['probabilities']['fake']:.2%}")
            print(f"  - Real News: {result['probabilities']['real']:.2%}")
            print("="*60)
    else:
        # Interactive mode
        main()
