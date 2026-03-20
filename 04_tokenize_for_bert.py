import pandas as pd
from transformers import BertTokenizer
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/fake_or_real_news_encoded.csv')

# Check for any empty or NaN values
print(f"Total samples: {len(df)}")
print(f"NaN in content: {df['content'].isna().sum()}")
print(f"Empty strings: {(df['content'] == '').sum()}")

# Remove any rows with empty or NaN content
df = df[df['content'].notna()]
df = df[df['content'] != '']
df = df[df['content'].str.strip() != '']

print(f"Samples after cleaning: {len(df)}")

x = df['content']
y = df['label_num']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to list and ensure all are strings
x_train_list = [str(text).strip() for text in x_train.tolist()]
x_test_list = [str(text).strip() for text in x_test.tolist()]

# Check samples
print(f"\nFirst training sample preview: {x_train_list[0][:100]}...")
print(f"Training samples: {len(x_train_list)}")
print(f"Testing samples: {len(x_test_list)}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print("\nTokenizing training data... This may take a while")
train_encodings = tokenizer(
    x_train_list,
    truncation=True,
    padding='max_length',
    max_length=256,
    return_tensors=None  # Return lists, not tensors
)

print("Tokenizing test data...")
test_encodings = tokenizer(
    x_test_list,
    truncation=True,
    padding='max_length',
    max_length=256,
    return_tensors=None
)

# Save the tokenized data
print("\nSaving tokenized data...")
np.save('data/x_train_input_ids.npy', train_encodings['input_ids'])
np.save('data/x_train_attention_mask.npy', train_encodings['attention_mask'])
np.save('data/y_train.npy', y_train.to_numpy())

np.save('data/x_test_input_ids.npy', test_encodings['input_ids'])
np.save('data/x_test_attention_mask.npy', test_encodings['attention_mask'])
np.save('data/y_test.npy', y_test.to_numpy())

print("Tokenization complete!")
print("Saved tokenized data inside /data/")