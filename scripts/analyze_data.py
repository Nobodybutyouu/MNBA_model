"""
Data Analysis Script
Analyze the dataset to understand why model performance is low
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re


def analyze_text_quality(df):
    """Analyze text quality and patterns"""
    
    print("="*70)
    print("TEXT QUALITY ANALYSIS")
    print("="*70)
    
    # Sample texts
    print("\nSample REAL news texts:")
    real_samples = df[df['label'] == 'real']['text'].head(3)
    for i, text in enumerate(real_samples, 1):
        print(f"\n{i}. {text[:200]}...")
    
    print("\n" + "-"*70)
    print("\nSample FAKE news texts:")
    fake_samples = df[df['label'] == 'fake']['text'].head(3)
    for i, text in enumerate(fake_samples, 1):
        print(f"\n{i}. {text[:200]}...")
    
    # Text statistics
    print("\n" + "="*70)
    print("TEXT STATISTICS")
    print("="*70)
    
    df['text_length'] = df['text'].fillna('').str.len()
    df['word_count'] = df['text'].fillna('').str.split().str.len()
    df['avg_word_length'] = df['text'].fillna('').apply(
        lambda x: np.mean([len(word) for word in x.split()]) if x else 0
    )
    df['unique_words'] = df['text'].fillna('').apply(
        lambda x: len(set(x.lower().split()))
    )
    df['lexical_diversity'] = df['unique_words'] / (df['word_count'] + 1)
    
    print("\nBy Label:")
    stats = df.groupby('label')[['text_length', 'word_count', 'avg_word_length', 
                                   'unique_words', 'lexical_diversity']].mean()
    print(stats)
    
    # Check for patterns
    print("\n" + "="*70)
    print("PATTERN ANALYSIS")
    print("="*70)
    
    # Check if text looks randomly generated
    sample_text = df['text'].iloc[0]
    words = sample_text.split()[:50]
    
    print(f"\nFirst 50 words of first article:")
    print(' '.join(words))
    
    # Check word frequency
    all_words = ' '.join(df['text'].fillna('')).lower().split()
    word_freq = Counter(all_words)
    
    print(f"\nTotal unique words in dataset: {len(word_freq):,}")
    print(f"\nMost common words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count:,}")
    
    # Check for news-specific words
    news_words = ['president', 'government', 'election', 'trump', 'biden', 
                  'congress', 'senate', 'minister', 'official', 'statement']
    
    print(f"\nNews-specific word frequency:")
    for word in news_words:
        count = word_freq.get(word, 0)
        print(f"  {word}: {count}")
    
    return df


def check_data_authenticity(df):
    """Check if data looks authentic or synthetic"""
    
    print("\n" + "="*70)
    print("DATA AUTHENTICITY CHECK")
    print("="*70)
    
    issues = []
    
    # Check 1: Are there proper sentences?
    sample_text = df['text'].iloc[0]
    sentences = sample_text.split('.')
    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
    
    print(f"\nAverage sentence length: {avg_sentence_length:.1f} words")
    if avg_sentence_length < 5:
        issues.append("Very short sentences - may indicate synthetic data")
    
    # Check 2: Capitalization patterns
    sample_texts = df['text'].head(10)
    capital_starts = sum(1 for text in sample_texts if text and text[0].isupper())
    print(f"Texts starting with capital letter: {capital_starts}/10")
    if capital_starts < 5:
        issues.append("Low capitalization - may indicate preprocessed data")
    
    # Check 3: Punctuation patterns
    avg_periods = df['text'].fillna('').str.count('\.').mean()
    avg_commas = df['text'].fillna('').str.count(',').mean()
    print(f"Average periods per text: {avg_periods:.1f}")
    print(f"Average commas per text: {avg_commas:.1f}")
    
    # Check 4: Check for URLs, emails (should be in news)
    has_urls = df['text'].fillna('').str.contains('http|www', case=False).sum()
    has_emails = df['text'].fillna('').str.contains('@', case=False).sum()
    print(f"Texts with URLs: {has_urls}")
    print(f"Texts with emails: {has_emails}")
    
    # Check 5: Named entities (rough check)
    capitalized_words = []
    for text in df['text'].head(100):
        if pd.notna(text):
            words = text.split()
            capitalized_words.extend([w for w in words if w and w[0].isupper() and len(w) > 1])
    
    unique_caps = len(set(capitalized_words))
    print(f"Unique capitalized words (first 100 texts): {unique_caps}")
    if unique_caps < 50:
        issues.append("Few capitalized words - may lack proper nouns/names")
    
    # Summary
    print("\n" + "="*70)
    print("AUTHENTICITY ASSESSMENT")
    print("="*70)
    
    if issues:
        print("\nPotential Issues Detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        print("\nConclusion: Data may be synthetic, heavily preprocessed, or anonymized")
        print("This explains the ~50% accuracy (random guessing)")
    else:
        print("\nNo major issues detected - data appears authentic")
    
    return issues


def plot_distributions(df, output_dir='outputs'):
    """Plot data distributions"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Text length distribution
    for label in df['label'].unique():
        subset = df[df['label'] == label]['text_length']
        axes[0, 0].hist(subset, bins=50, alpha=0.6, label=label)
    axes[0, 0].set_xlabel('Text Length (characters)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].legend()
    
    # Word count distribution
    for label in df['label'].unique():
        subset = df[df['label'] == label]['word_count']
        axes[0, 1].hist(subset, bins=50, alpha=0.6, label=label)
    axes[0, 1].set_xlabel('Word Count')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Word Count Distribution')
    axes[0, 1].legend()
    
    # Lexical diversity
    for label in df['label'].unique():
        subset = df[df['label'] == label]['lexical_diversity']
        axes[1, 0].hist(subset, bins=50, alpha=0.6, label=label)
    axes[1, 0].set_xlabel('Lexical Diversity (unique words / total words)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Lexical Diversity Distribution')
    axes[1, 0].legend()
    
    # Category distribution (fallback to label distribution if missing)
    ax = axes[1, 1]
    if 'category' in df.columns and not df['category'].isna().all():
        category_counts = df.groupby(['category', 'label']).size().unstack(fill_value=0)
        category_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Category')
        ax.set_title('Category Distribution by Label')
        ax.legend(title='Label')
        ax.tick_params(axis='x', rotation=45)
    else:
        label_counts = df['label'].value_counts().sort_index()
        label_counts.plot(kind='bar', ax=ax, color=sns.color_palette("deep", len(label_counts)))
        ax.set_xlabel('Label')
        ax.set_title('Label Distribution')
        ax.tick_params(axis='x', rotation=0)

    ax.set_ylabel('Count')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'data_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDistribution plots saved to '{output_path}'")
    plt.close()


def main():
    """Main execution"""
    
    print("="*70)
    print("DATASET ANALYSIS FOR LOW MODEL PERFORMANCE")
    print("="*70)
    
    # Load data
    DATA_PATH = r'data\cleaned\news.csv'
    
    print(f"\nLoading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    
    # Analyze text quality
    df = analyze_text_quality(df)
    
    # Check authenticity
    issues = check_data_authenticity(df)
    
    # Plot distributions
    plot_distributions(df)
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if issues:
        print("\nThe low model accuracy (~50%) is likely due to data quality issues.")
        print("\nOptions:")
        print("  1. Obtain a different dataset with real news text")
        print("  2. If this is anonymized data, request the original text")
        print("  3. Use this as a demonstration of the methodology only")
        print("  4. For manuscript: Focus on the methodology rather than results")
        print("\nSuggested datasets:")
        print("  - LIAR dataset")
        print("  - FakeNewsNet")
        print("  - ISOT Fake News Dataset")
        print("  - Kaggle Fake News Detection datasets")
    else:
        print("\nData appears authentic. Model performance may be improved by:")
        print("  1. Better feature engineering")
        print("  2. Ensemble methods")
        print("  3. Deep learning approaches")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
