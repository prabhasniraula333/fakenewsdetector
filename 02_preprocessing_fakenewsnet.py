import pandas as pd
import re

print("Loading dataset...")
df = pd.read_csv('data/fake_or_real_news.csv')

print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# The 'text' column was already created by fakenewsnet_loader.py
# from the 'title' column
df['content'] = df['text']

def clean_text(text):
    """Clean and normalize text"""
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', " ", text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z\s]', " ", text)
    # Remove extra whitespace
    text = re.sub(r'\s+', " ", text).strip()
    return text

print("\nCleaning text...")
df['content'] = df['content'].apply(clean_text)

# Remove any empty content
df = df[df['content'].str.strip() != '']
df = df[df['content'].str.len() > 10]  # At least 10 characters

print(f"Samples after cleaning: {len(df)}")

# Show samples
print("\n" + "="*60)
print("SAMPLE CLEANED DATA:")
print("="*60)
print(df[['content', 'label']].head(10))

# Save cleaned dataset
df.to_csv('data/fake_or_real_news_cleaned.csv', index=False)
print("\n Cleaned dataset saved as: data/fake_or_real_news_cleaned.csv")

# Show label distribution
print("\n" + "="*60)
print("LABEL DISTRIBUTION:")
print("="*60)
print(df['label'].value_counts())
print(f"\nFake: {(df['label']=='fake').sum()} ({(df['label']=='fake').sum()/len(df)*100:.1f}%)")
print(f"Real: {(df['label']=='real').sum()} ({(df['label']=='real').sum()/len(df)*100:.1f}%)")