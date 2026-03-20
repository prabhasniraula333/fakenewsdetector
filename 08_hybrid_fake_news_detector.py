import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification

# LOAD MODEL
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f"Model loaded on {device}!")

# CLEANING FUNCTION
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', " ", text)
    text = re.sub(r'[^a-z\s]', " ", text)
    text = re.sub(r'\s+', " ", text).strip()
    return text

# RULE-BASED INDICATORS
def check_clickbait_patterns(text):
    """Detect clickbait/fake news linguistic patterns"""
    text_lower = text.lower()
    
    clickbait_phrases = [
        "you won't believe", "doctors hate", "one weird trick",
        "this one simple", "what happens next", "shocking",
        "they don't want you to know", "big pharma", "mainstream media won't",
        "wake up", "the truth about", "exposed", "conspiracy"
    ]
    
    score = 0
    found_patterns = []
    
    for phrase in clickbait_phrases:
        if phrase in text_lower:
            score += 1
            found_patterns.append(phrase)
    
    return score, found_patterns

def check_emotional_manipulation(text):
    """Check for excessive emotional language"""
    text_upper = text.upper()
    
    # Count ALL CAPS words
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 3]
    caps_ratio = len(caps_words) / len(words) if words else 0
    
    # Count exclamation marks
    exclamation_count = text.count('!')
    
    # Emotional words
    emotional_words = [
        'outrage', 'shocking', 'bombshell', 'explosive', 'devastating',
        'horrifying', 'terrifying', 'unbelievable', 'insane', 'crazy'
    ]
    
    emotion_score = sum(1 for word in emotional_words if word in text.lower())
    
    return {
        'caps_ratio': caps_ratio,
        'exclamations': exclamation_count,
        'emotional_words': emotion_score
    }

def check_source_credibility_indicators(text):
    """Look for vague sourcing"""
    text_lower = text.lower()
    
    vague_sources = [
        "sources say", "according to sources", "insiders claim",
        "anonymous sources", "experts say", "studies show",
        "reports suggest", "it is believed"
    ]
    
    vague_count = sum(1 for phrase in vague_sources if phrase in text_lower)
    
    return vague_count

# BERT PREDICTION
def get_bert_prediction(text):
    """Get BERT model prediction"""
    cleaned = clean_text(text)
    
    encoding = tokenizer(
        cleaned,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

# HYBRID PREDICTION
def hybrid_predict(article_text):
    """
    Combine BERT prediction with rule-based checks
    Returns: final prediction, confidence, explanation
    """
    
    # Get BERT prediction
    bert_pred, bert_conf = get_bert_prediction(article_text)
    bert_label = "FAKE" if bert_pred == 0 else "REAL"
    
    # Get rule-based scores
    clickbait_score, clickbait_patterns = check_clickbait_patterns(article_text)
    emotion_metrics = check_emotional_manipulation(article_text)
    vague_sources = check_source_credibility_indicators(article_text)
    
    # Calculate suspicion score (0-10)
    suspicion_score = 0
    reasons = []
    
    # Clickbait patterns (0-3 points)
    if clickbait_score > 0:
        suspicion_score += min(clickbait_score, 3)
        reasons.append(f"Contains {clickbait_score} clickbait phrase(s): {', '.join(clickbait_patterns[:2])}")
    
    # Emotional manipulation (0-3 points)
    if emotion_metrics['caps_ratio'] > 0.3:
        suspicion_score += 2
        reasons.append(f"Excessive ALL CAPS usage ({emotion_metrics['caps_ratio']*100:.0f}%)")
    if emotion_metrics['exclamations'] > 2:
        suspicion_score += 1
        reasons.append(f"Multiple exclamation marks ({emotion_metrics['exclamations']})")
    if emotion_metrics['emotional_words'] > 2:
        suspicion_score += 1
        reasons.append(f"High emotional language")
    
    # Vague sourcing (0-2 points)
    if vague_sources > 0:
        suspicion_score += min(vague_sources, 2)
        reasons.append(f"Vague source attribution ({vague_sources} instances)")
    
    # Combine with BERT
    if bert_label == "FAKE":
        suspicion_score += 3
        reasons.append(f"BERT model predicts FAKE ({bert_conf*100:.1f}% confidence)")
    else:
        suspicion_score -= 2
        reasons.append(f"BERT model predicts REAL ({bert_conf*100:.1f}% confidence)")
    
    # Final decision
    suspicion_score = max(0, min(10, suspicion_score))  # Clamp 0-10
    
    if suspicion_score >= 6:
        final_label = "LIKELY FAKE"
        confidence = suspicion_score / 10
    elif suspicion_score >= 4:
        final_label = "UNCERTAIN"
        confidence = 0.5
    else:
        final_label = "LIKELY REAL"
        confidence = 1 - (suspicion_score / 10)
    
    return {
        'label': final_label,
        'confidence': confidence,
        'suspicion_score': suspicion_score,
        'bert_prediction': bert_label,
        'bert_confidence': bert_conf,
        'reasons': reasons,
        'clickbait_patterns': clickbait_patterns,
        'emotion_metrics': emotion_metrics
    }

# INTERACTIVE MODE
print("\n" + "="*60)
print(" HYBRID FAKE NEWS DETECTOR")
print("="*60)
print("Combines BERT AI + Rule-based pattern detection")
print("Type 'quit' to exit\n")

while True:
    print("-"*60)
    user_input = input("\nEnter news headline/article:\n> ")
    
    if user_input.lower() == 'quit':
        print("\nExiting...")
        break
    
    if len(user_input.strip()) < 10:
        print("  Please enter a longer text")
        continue
    
    # Make prediction
    result = hybrid_predict(user_input)
    
    # Display results
    print("\n" + "="*60)
    print(f" PREDICTION: {result['label']}")
    print(f" CONFIDENCE: {result['confidence']*100:.1f}%")
    print(f"  SUSPICION SCORE: {result['suspicion_score']}/10")
    print("="*60)
    
    print(f"\n BERT Model: {result['bert_prediction']} ({result['bert_confidence']*100:.1f}%)")
    
    if result['reasons']:
        print(f"\n📋 Warning Signs Detected:")
        for i, reason in enumerate(result['reasons'], 1):
            print(f"   {i}. {reason}")
    else:
        print(f"\n No obvious warning signs detected")
    
    # Confidence interpretation
    if result['confidence'] > 0.8:
        print("\n High confidence in prediction")
    elif result['confidence'] > 0.6:
        print("\n  Moderate confidence - review carefully")
    else:
        print("\n Low confidence - manual fact-checking recommended")