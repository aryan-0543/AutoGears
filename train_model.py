
import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
csv_path = 'uploads/sample_transactions.csv'
df = pd.read_csv(csv_path)

# Clean and prepare data
df = df.dropna(subset=['Description', 'Category'])
descriptions = df['Description'].astype(str).tolist()
categories = df['Category'].astype(str).tolist()

# Convert text to embeddings
model_name = 'all-MiniLM-L6-v2'
embedder = SentenceTransformer(model_name)
X = embedder.encode(descriptions, show_progress_bar=True)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(categories)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
clf.fit(X_train, y_train)

# Save model and label encoder
os.makedirs('models', exist_ok=True)
joblib.dump((clf, embedder, label_encoder), 'models/classifier.pkl')
print("✅ Model trained and saved to models/classifier.pkl")
