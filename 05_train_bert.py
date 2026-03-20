import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# LOAD TOKENIZED DATA
print("Loading tokenized data...")

x_train_ids = np.load('data/x_train_input_ids.npy')
x_train_mask = np.load('data/x_train_attention_mask.npy')
y_train = np.load('data/y_train.npy')

x_test_ids = np.load('data/x_test_input_ids.npy')
x_test_mask = np.load('data/x_test_attention_mask.npy')
y_test = np.load('data/y_test.npy')

print(f"Training samples: {len(y_train)}")
print(f"Testing samples: {len(y_test)}")

# CONVERT TO TENSORS
train_inputs = torch.tensor(x_train_ids)
train_masks = torch.tensor(x_train_mask)
train_labels = torch.tensor(y_train, dtype=torch.long)

test_inputs = torch.tensor(x_test_ids)
test_masks = torch.tensor(x_test_mask)
test_labels = torch.tensor(y_test, dtype=torch.long)

# CREATE DATA LOADERS
# If OOM errors, reduce back to 8 for the batch size
batch_size = 18

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_loader = DataLoader(test_data, batch_size=batch_size)

# LOAD BERT MODEL
print("\nLoading BERT model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("="*60)
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("="*60)

# OPTIMIZER with weight decay
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# LEARNING RATE SCHEDULER with warmup
epochs = 4
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
    num_training_steps=total_steps
)

# TRAINING LOOP
print(f"\nStarting training for {epochs} epochs...")
print(f"Batch size: {batch_size}")
print(f"Total batches per epoch: {len(train_loader)}")

for epoch in range(epochs):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{epochs}")
    print('='*60)
    
    # TRAINING
    model.train()
    total_train_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        b_input_ids = batch[0].to(device)
        b_attention_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        optimizer.zero_grad()

        output = model(
            b_input_ids,
            attention_mask=b_attention_mask,
            labels=b_labels
        )

        loss = output.loss
        total_train_loss += loss.item()

        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    
    # VALIDATION
    model.eval()
    total_val_loss = 0
    correct = 0
    total = 0
    
    # Store predictions for detailed analysis
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validation"):
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            output = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )
            
            total_val_loss += output.loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(output.logits, dim=1)
            correct += (predictions == b_labels).sum().item()
            total += b_labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(b_labels.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(test_loader)
    accuracy = correct / total
    
    print(f"\nTraining Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate per-class accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    fake_mask = all_labels == 0
    real_mask = all_labels == 1
    
    fake_acc = (all_predictions[fake_mask] == all_labels[fake_mask]).mean()
    real_acc = (all_predictions[real_mask] == all_labels[real_mask]).mean()
    
    print(f"Fake News Accuracy: {fake_acc:.4f} ({fake_acc*100:.2f}%)")
    print(f"Real News Accuracy: {real_acc:.4f} ({real_acc*100:.2f}%)")

print("\n" + "="*60)
print("Training complete!")
print("="*60)

# SAVE MODEL
model.save_pretrained("saved_model")
print("\n Model saved to /saved_model/")
