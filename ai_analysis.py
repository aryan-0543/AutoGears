import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the model, embedder, and label encoder
model, embedder, label_encoder = joblib.load('models/classifier.pkl')

def load_model():
    return model

def categorize_transaction(description):
    embedding = embedder.encode([description])
    prediction = model.predict(embedding)
    category = label_encoder.inverse_transform(prediction)[0]
    return category

def classify_transactions(df, model):
    df = df.dropna(subset=['Description'])
    df['Predicted_Category'] = df['Description'].apply(categorize_transaction)
    return df

def generate_suggestions(df):
    summary = df.groupby('Predicted_Category')['Amount'].sum().sort_values(ascending=False)
    top_categories = summary.head(3)
    suggestions = [f"You are spending a lot on {cat}. Consider reducing these expenses." for cat in top_categories.index]
    return suggestions, summary

def forecast_expenses(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)
    monthly_expense = df.resample('M').sum()['Amount']
    forecast = monthly_expense.rolling(window=2).mean().fillna(method='bfill')
    return forecast.tail(3)
