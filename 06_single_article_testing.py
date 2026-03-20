import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

# LOAD TRAINED MODEL
print("Loading trained model...")
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

print(f"Model loaded successfully on {device}!")
print("="*60)

# TEXT CLEANING FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', " ", text)
    text = re.sub(r'[^a-z\s]', " ", text)
    text = re.sub(r'\s+', " ", text).strip()
    return text

# PREDICTION FUNCTION
def predict_news(article_text):
    """
    Predict if news article is FAKE or REAL
    Returns: prediction label, confidence score
    """
    # Clean the text
    cleaned_text = clean_text(article_text)
    
    # Tokenize
    encoding = tokenizer(
        cleaned_text,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    # Map to label
    label = "FAKE" if prediction == 0 else "REAL"
    
    return label, confidence

# INTERACTIVE MODE
print("\n🔍 FAKE NEWS DETECTOR")
print("="*60)
print("Enter a news article to check if it's FAKE or REAL")
print("Type 'quit' to exit\n")

while True:
    print("-"*60)
    user_input = input("\nEnter news article (or 'quit' to exit):\n> ")
    
    if user_input.lower() == 'quit':
        print("\nExiting... Thanks for using the detector!")
        break
    
    if len(user_input.strip()) < 20:
        print("  Please enter a longer article (at least 20 characters)")
        continue
    
    # Make prediction
    label, confidence = predict_news(user_input)
    
    # Display result
    print("\n" + "="*60)
    print(f" PREDICTION: {label}")
    print(f" CONFIDENCE: {confidence*100:.2f}%")
    print("="*60)
    
    if confidence > 0.95:
        print(" High confidence prediction")
    elif confidence > 0.80:
        print("  Moderate confidence - review carefully")
    else:
        print(" Low confidence - uncertain prediction")