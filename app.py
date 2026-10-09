import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.datasets import load_breast_cancer

# Page configuration
st.set_page_config(
    page_title="Classification Algorithms", page_icon="🤖", layout="wide"
)

# Load the trained model
MODEL_PATH = "models/best_classification_model.joblib"

st.title("🤖 Classification Algorithms")
st.subheader("Breast Cancer Classification Demo")
st.write(
    "Explore a machine learning model trained on the "
    "scikit-learn breast cancer dataset."
)

st.warning(
    "Educational demonstration only. This tool is not "
    "intended for medical diagnosis or treatment decisions."
)

# Check whether the model exists
if not os.path.exists(MODEL_PATH):
    st.error(
        "Trained model not found. Run `python main.py` "
        "first to train and save the model."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
data = load_breast_cancer()

# Sidebar information
st.sidebar.header("About the Project")
st.sidebar.write("Dataset: Breast Cancer Wisconsin")
st.sidebar.write("Task: Binary classification")
st.sidebar.write("Model: Best model selected by F1-score")

# Show model comparison if available
results_path = "outputs/algorithm_comparison.csv"

if os.path.exists(results_path):
    st.header("Model Comparison")

    results = pd.read_csv(results_path)
    st.dataframe(results, use_container_width=True)

    if "Algorithm" in results.columns and "Accuracy (%)" in results.columns:
        st.bar_chart(results.set_index("Algorithm")["Accuracy (%)"])

# Prediction section
st.header("Try a Sample Prediction")

st.write(
    "Choose a sample from the test dataset and see the "
    "model's predicted class. No manual clinical interpretation "
    "is intended."
)

X = pd.DataFrame(data.data, columns=data.feature_names)

sample_number = st.number_input(
    "Select sample number", min_value=1, max_value=len(X), value=1, step=1
)

if st.button("Predict Class", type="primary"):
    sample = X.iloc[[int(sample_number) - 1]]
    prediction = int(model.predict(sample)[0])

    st.subheader("Prediction Result")
    st.write("Sample number:", int(sample_number))
    st.write("Predicted class:", data.target_names[prediction])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample)[0]

        probability_df = pd.DataFrame(
            {"Class": data.target_names, "Probability (%)": probabilities * 100}
        )

        st.write("Model probability estimates:")
        st.dataframe(probability_df.round(2), use_container_width=True)

# Footer
st.divider()
st.caption("Machine Learning Project | Python, scikit-learn, Streamlit")
