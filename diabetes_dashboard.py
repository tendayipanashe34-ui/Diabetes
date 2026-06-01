import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report
import seaborn as sns

# Assuming you have the results dictionary from your notebook
# You can copy the results dict here or load from a saved file

# Example results (replace with your actual results)
results = {
    "Deep ANN (5-layer)": {
        "accuracy": 0.85,
        "roc_auc": 0.90,
        "f1": 0.82,
        "cv_auc": 0.88,
        "cv_std": 0.02,
        # Add other metrics if available
    },
    "TabTransformer": {
        "accuracy": 0.87,
        "roc_auc": 0.92,
        "f1": 0.85,
        "cv_auc": 0.89,
        "cv_std": 0.03,
    },
    "Hybrid Model": {
        "accuracy": 0.89,
        "roc_auc": 0.94,
        "f1": 0.87,
        "cv_auc": 0.91,
        "cv_std": 0.02,
    },
}

st.title("Diabetes Prediction Model Comparison Dashboard")

st.header("Model Performance Metrics")

# Create comparison table
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'F1 Score': [results[m]['f1'] for m in results],
    'AUC': [results[m]['roc_auc'] for m in results],
    'CV AUC': [results[m]['cv_auc'] for m in results],
    'CV Std': [results[m]['cv_std'] for m in results]
})

st.dataframe(comparison_df.style.highlight_max(axis=0))

# Bar chart comparison
st.subheader("Metric Comparison")
metrics = ['Accuracy', 'F1 Score', 'AUC']
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.25

for i, model in enumerate(results.keys()):
    ax.bar(x + i*width, [results[model][m.lower().replace(' ', '_')] for m in metrics], width, label=model)

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_title("Model Performance Comparison")
st.pyplot(fig)

# ROC Curves (placeholder - you'd need actual y_test and probabilities)
st.subheader("ROC Curves")
# Add your ROC curve plotting code here

st.header("About")
st.write("This dashboard compares three deep learning models for diabetes prediction: Deep ANN, TabTransformer, and Hybrid Model.")