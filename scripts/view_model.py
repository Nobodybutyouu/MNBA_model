"""
Model Viewer Script
View the contents and details of a saved model file
"""

import pickle
import sys
import os


def view_model(file_path='models/mnb_fake_news_model.pkl'):
    """
    View the contents of a saved model file
    
    Args:
        file_path: Path to the PKL file
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        print("="*60)
        print("MODEL FILE CONTENTS")
        print("="*60)
        
        if isinstance(data, dict):
            print("\nModel Dictionary Keys:")
            for key in data.keys():
                print(f"  - {key}")
            
            print("\n" + "-"*60)
            
            if 'model' in data:
                print("\nModel Type:", type(data['model']).__name__)
                print("Model Parameters:", data['model'].get_params())
            
            if 'vectorizer' in data:
                print("\nVectorizer Type:", type(data['vectorizer']).__name__)
                print("Vectorizer Parameters:", data['vectorizer'].get_params())
            
            if 'vectorizer_type' in data:
                print("\nVectorizer Type Setting:", data['vectorizer_type'])
            
            if 'max_features' in data:
                print("Max Features:", data['max_features'])
        else:
            print("\nModel Type:", type(data).__name__)
            print("\nModel Details:")
            print(data)
        
        print("\n" + "="*60)
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        print("Please ensure the model has been trained and saved.")
    except Exception as e:
        print(f"Error loading PKL file: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'models/mnb_fake_news_model.pkl'
    
    view_model(file_path)
