import torch
import pandas as pd
import numpy as np
import random
import re
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "cyber_intent_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = model.config.id2label
label2id = model.config.label2id

df = pd.read_csv("cyber_intent_final.csv")
texts = df["scenario"].tolist()
true_labels = df["label"].tolist()

def random_delete(text):
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words)-1)
        words.pop(idx)
    return " ".join(words)

def random_mask(text):
    words = text.split()
    if len(words) > 3:
        idx = random.randint(0, len(words)-1)
        words[idx] = "[MASK]"
    return " ".join(words)

def add_typos(text):
    text = text.replace("oo", "o").replace("ss", "s").replace("ll", "l")
    text = re.sub(r"ing\b", "in", text)
    return text

def basic_noise(text):
    noise_words = ["please", "urgent", "sir", "help", "now"]
    if random.random() < 0.6:
        text = random.choice(noise_words) + " " + text
    return text

def lower_and_clean(text):
    text = text.lower()
    text = text.replace(",", "").replace(".", "")
    return text

def add_noise(text):
    text = random_delete(text)
    text = random_mask(text)
    text = add_typos(text)
    text = basic_noise(text)
    text = lower_and_clean(text)
    return text

texts_noisy = [add_noise(t) for t in texts]

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
    return int(np.argmax(probs))

preds = [predict(t) for t in texts_noisy]

acc = accuracy_score(true_labels, preds)
print("\nAccuracy:", round(acc * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(true_labels, preds, target_names=list(label2id.keys())))

cm = confusion_matrix(true_labels, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=list(label2id.keys()),
            yticklabels=list(label2id.keys()))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
