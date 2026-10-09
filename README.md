# Classification Algorithms: Breast Cancer Classification

A machine learning project that trains and compares multiple classification algorithms using the Breast Cancer Wisconsin dataset available through scikit-learn.

## Overview

This project demonstrates a complete introductory machine learning workflow, from loading and splitting a dataset to training classification models, evaluating their performance, visualizing results, and demonstrating predictions through a Streamlit web interface.

## Objectives

* Understand supervised classification.
* Train multiple machine learning algorithms.
* Compare models using accuracy, precision, recall, and F1-score.
* Visualize model performance using graphs and a confusion matrix.
* Save a trained model for reuse.
* Present the results through a simple interactive web application.

## Algorithms Implemented

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Support Vector Machine (SVM)
6. Gaussian Naive Bayes
7. Gradient Boosting

## Technology Stack

* Python
* pandas
* scikit-learn
* Matplotlib
* Joblib
* Streamlit

## Dataset

The project uses the Breast Cancer Wisconsin dataset included in scikit-learn.

* Samples: 569
* Features: 30 numeric features
* Classes: malignant and benign

The dataset is divided into training and testing sets using an 80:20 split.

## Evaluation Metrics

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of predicted positive cases that are correct.
* **Recall:** Proportion of actual positive cases correctly identified.
* **F1-score:** Harmonic mean of precision and recall.
* **Confusion matrix:** Shows correct and incorrect predictions by class.

The current implementation selects the model with the highest F1-score for the benign class. The metric used for selection should be considered when interpreting the results.

## Project Structure

```text
classification_algorithms/
├── main.py
├── app.py
├── README.md
├── .gitignore
├── models/
│   └── best_classification_model.joblib
└── outputs/
    ├── algorithm_comparison.csv
    ├── accuracy_comparison.png
    ├── f1_score_comparison.png
    ├── confusion_matrix.png
    └── test_predictions.csv
```

The model and output files are generated when the training script runs. Exact generated files may vary.

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd classification_algorithms
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the repository URL.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install pandas scikit-learn matplotlib joblib streamlit
```

### 4. Train and evaluate the models

```bash
python main.py
```

### 5. Run the web application

```bash
python -m streamlit run app.py
```

The application should open in your browser, usually at `http://localhost:8501`.

## Results

The training script produces a model comparison table, evaluation metrics, a confusion matrix, and performance graphs.

Add your actual results from `outputs/algorithm_comparison.csv` after running the project. Do not insert estimated or unverified accuracy values.

## Future Improvements

* Add more classification algorithms.
* Compare both malignant-class and benign-class performance.
* Include cross-validation and hyperparameter tuning.
* Add more interactive visualizations.
* Improve the Streamlit interface.
* Add automated tests and deployment instructions.

## Disclaimer

This is an educational machine learning project. It is not a validated medical device and must not be used for medical diagnosis or treatment decisions.
