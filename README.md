# Iris Flower Classification using Machine Learning

This project predicts the species of an Iris flower (**Setosa**, **Versicolor**, or **Virginica**) based on sepal and petal dimensions using Logistic Regression.

## Dataset
The Iris dataset is a classic benchmark dataset in machine learning. It consists of 150 instances (50 for each of the 3 species).

## Features
- `sepal length (cm)`
- `sepal width (cm)`
- `petal length (cm)`
- `petal width (cm)`

**Target Variable:** `species` (0 = Setosa, 1 = Versicolor, 2 = Virginica)

## Model Used
- **Logistic Regression**: A multi-class classification model parameterized with `max_iter=200`.

## Training Process
1. Dataset loaded via `sklearn.datasets`.
2. Checked and cleaned duplicates/missing values.
3. Dataset split into 80% training and 20% testing sets (`stratify=y`).
4. Trained the model using `LogisticRegression.fit()`.

## Evaluation Results
- **Accuracy**: ~96.67%
- Evaluated using **Confusion Matrix** and **Classification Report** (Precision, Recall, F1-Score).

## How to Run the Project

1. Clone or download this repository.
2. Install requirements:
   ```bash
   pip install -r requirements.txt