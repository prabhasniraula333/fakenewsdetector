import pandas as pd
import os

print("="*60)
print("LOADING FAKENEWSNET DATASET")
print("="*60)

# Load PolitiFact data
print("\nLoading PolitiFact dataset...")
try:
    politifact_fake = pd.read_csv('data/politifact_fake.csv')
    politifact_real = pd.read_csv('data/politifact_real.csv')
    
    politifact_fake['label'] = 'fake'
    politifact_real['label'] = 'real'
    
    print(f"✓ PolitiFact Fake: {len(politifact_fake)} articles")
    print(f"✓ PolitiFact Real: {len(politifact_real)} articles")
    
    politifact = pd.concat([politifact_fake, politifact_real], ignore_index=True)
except FileNotFoundError as e:
    print(f"  PolitiFact files not found: {e}")
    politifact = None

# Load GossipCop data (optional - can be large)
print("\nLoading GossipCop dataset...")
try:
    gossipcop_fake = pd.read_csv('data/gossipcop_fake.csv')
    gossipcop_real = pd.read_csv('data/gossipcop_real.csv')
    
    gossipcop_fake['label'] = 'fake'
    gossipcop_real['label'] = 'real'
    
    print(f"✓ GossipCop Fake: {len(gossipcop_fake)} articles")
    print(f"✓ GossipCop Real: {len(gossipcop_real)} articles")
    
    gossipcop = pd.concat([gossipcop_fake, gossipcop_real], ignore_index=True)
except FileNotFoundError as e:
    print(f"  GossipCop files not found: {e}")
    print("   (GossipCop is optional - you can use just PolitiFact)")
    gossipcop = None

# Combine datasets
print("\n" + "="*60)
print("COMBINING DATASETS")
print("="*60)

datasets_to_combine = []
if politifact is not None:
    datasets_to_combine.append(politifact)
if gossipcop is not None:
    datasets_to_combine.append(gossipcop)

if len(datasets_to_combine) == 0:
    print(" ERROR: No datasets found!")
    print("\nPlease download FakeNewsNet from:")
    print("https://www.kaggle.com/datasets/mdepak/fakenewsnet")
    exit(1)

df = pd.concat(datasets_to_combine, ignore_index=True)

# Check what columns exist
print(f"\nColumns in dataset: {df.columns.tolist()}")

# FakeNewsNet CSVs have: id, url, title, tweet_ids
# We use 'title' as our text since full articles aren't included
if 'title' in df.columns:
    print("\n✓ Using 'title' column as text")
    df['text'] = df['title']
elif 'news_url' in df.columns and 'title' not in df.columns:
    print("\n  No 'title' column found!")
    print(f"Available columns: {df.columns.tolist()}")
    print("This dataset may need additional processing.")
else:
    print(f"\n Available columns: {df.columns.tolist()}")
    print("Please check which column contains the news text")

# Remove rows with missing titles
df = df[df['text'].notna()]
df = df[df['text'] != '']
print(f"\nArticles with valid titles: {len(df)}")

# Display info
print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)
print(f"Total articles: {len(df)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())

# Remove rows with missing text
df = df[df['text'].notna()]
df = df[df['text'] != '']
print(f"\nArticles after removing empty: {len(df)}")

# Sample articles
print("\n" + "="*60)
print("SAMPLE ARTICLES")
print("="*60)
print("\nSample FAKE article:")
fake_sample = df[df['label'] == 'fake'].iloc[0]['text']
print(fake_sample[:300])

print("\nSample REAL article:")
real_sample = df[df['label'] == 'real'].iloc[0]['text']
print(real_sample[:300])

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Keep only necessary columns
df = df[['text', 'label']]

df.to_csv('data/fake_or_real_news.csv', index=False)
print("\n" + "="*60)
print(" DATASET SAVED!")
print("="*60)
print("Saved as: data/fake_or_real_news.csv")
print("\n NOTE: This dataset uses news TITLES only")
print("Expected accuracy: 70-85% ")