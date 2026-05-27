# Spam/Ham Classifier - Training Script
# Dataset: spam.csv  |  v1 = label (spam/ham)  |  v2 = message

import os
import re
import string
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — no window popup
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# Force UTF-8 output so special box-drawing chars print cleanly
sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spam.csv")
MODEL_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Download NLTK data ────────────────────────────────────────────────────────
print("[*] Downloading NLTK data...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)

# ── 1. Load Dataset ───────────────────────────────────────────────────────────
print("\n[*] Loading dataset from:", DATASET_PATH)
df = pd.read_csv(DATASET_PATH, encoding='latin-1')

# Keep only needed columns and rename
df = df[['v1', 'v2']].copy()
df.columns = ['label', 'message']
df.dropna(inplace=True)

print(f"    Total samples : {len(df)}")
print(f"    Label counts  :\n{df['label'].value_counts()}")

# ── 2. Text Preprocessing ─────────────────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    """Lowercase > remove digits/punct > tokenise > drop stopwords > stem."""
    text   = text.lower()
    text   = re.sub(r'\d+', '', text)
    text   = text.translate(str.maketrans('', '', string.punctuation))
    text   = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

print("\n[*] Preprocessing messages (this may take a moment)...")
df['clean_message'] = df['message'].apply(preprocess)

# ── 3. Encode Labels ──────────────────────────────────────────────────────────
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['clean_message']
y = df['label_enc']

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n    Train size : {len(X_train)}")
print(f"    Test  size : {len(X_test)}")

# ── 5. TF-IDF Vectorisation ───────────────────────────────────────────────────
print("\n[*] Vectorising with TF-IDF (unigrams + bigrams, max_features=5000)...")
vectorizer    = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ── 6. Train & Evaluate Models ────────────────────────────────────────────────
models = {
    "Naive Bayes"        : MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Linear SVM"         : LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

results   = {}
best_name = None
best_f1   = 0.0

print("\n" + "="*60)
print("   MODEL COMPARISON")
print("="*60)

for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    acc  = accuracy_score (y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score   (y_test, y_pred)
    f1   = f1_score       (y_test, y_pred)

    results[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                         model=clf, y_pred=y_pred)

    print(f"\n  >> {name}")
    print(f"     Accuracy  : {acc*100:.2f}%")
    print(f"     Precision : {prec*100:.2f}%")
    print(f"     Recall    : {rec*100:.2f}%")
    print(f"     F1-Score  : {f1*100:.2f}%")

    if f1 > best_f1:
        best_f1   = f1
        best_name = name

print("\n" + "="*60)
print(f"  BEST MODEL: {best_name}  (F1 = {best_f1*100:.2f}%)")
print("="*60)

# ── 7. Detailed Report ────────────────────────────────────────────────────────
best_model  = results[best_name]['model']
best_y_pred = results[best_name]['y_pred']

print(f"\nClassification Report -- {best_name}:")
print(classification_report(y_test, best_y_pred, target_names=['Ham', 'Spam']))

# ── 8. Save Model + Vectorizer ────────────────────────────────────────────────
joblib.dump(best_model,  os.path.join(MODEL_DIR, "spam_model.pkl"))
joblib.dump(vectorizer,  os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(best_name,   os.path.join(MODEL_DIR, "model_name.pkl"))

print(f"\n[OK] Model saved     -> models/spam_model.pkl")
print(f"[OK] Vectorizer saved-> models/tfidf_vectorizer.pkl")

# ── 9. Visualisations ─────────────────────────────────────────────────────────
print("\n[*] Generating visualisations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Spam / Ham Classifier - Analysis", fontsize=15, fontweight='bold')

# (a) Label Distribution pie
counts = df['label'].value_counts()
axes[0].pie(counts, labels=counts.index, autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'], startangle=90,
            textprops={'fontsize': 12})
axes[0].set_title("Dataset Distribution", fontsize=13)

# (b) Model Comparison bar chart
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x      = np.arange(len(metric_names))
width  = 0.25
colors_bar = ['#2196F3', '#9C27B0', '#FF9800']

for i, (name, res) in enumerate(results.items()):
    vals = [res['accuracy'], res['precision'], res['recall'], res['f1']]
    axes[1].bar(x + i*width, vals, width, label=name,
                color=colors_bar[i], alpha=0.85)

axes[1].set_xticks(x + width)
axes[1].set_xticklabels(metric_names, fontsize=10)
axes[1].set_ylim(0.8, 1.02)
axes[1].set_title("Model Comparison", fontsize=13)
axes[1].legend(fontsize=8)
axes[1].set_ylabel("Score")
axes[1].grid(axis='y', alpha=0.3)

# (c) Confusion Matrix for best model
cm = confusion_matrix(y_test, best_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'], ax=axes[2],
            linewidths=0.5, linecolor='gray')
axes[2].set_xlabel("Predicted", fontsize=11)
axes[2].set_ylabel("Actual",    fontsize=11)
axes[2].set_title(f"Confusion Matrix - {best_name}", fontsize=13)

plt.tight_layout()
chart_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.png")
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"[OK] Results chart saved -> results.png")

print("\n[DONE] Training complete! Run app.py to test messages interactively.")
