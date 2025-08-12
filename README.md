# Breast Cancer Prediction Web App

## Project Overview
This project is a **Breast Cancer Prediction** web application built using **Flask**.  
It uses two machine learning models:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**  

Users can **choose their preferred model** using a toggle switch in the web interface.  
The app predicts whether a tumor is:
- **B - Benign (Non-cancerous)**
- **M - Malignant (Cancerous)**  

The dataset used is the **Breast Cancer Wisconsin dataset** from Scikit-learn.

---

## Features
- User-friendly web interface with model selection toggle
- Accepts **30 medical features** as input
- Predicts tumor type with either Logistic Regression or KNN
- Styled with custom CSS for better UX

---

## Dataset
The dataset is loaded directly from **Scikit-learn**:

## It contains:
569 samples
30 numeric features
Target labels: 0 = Malignant, 1 = Benign

## Technologies Used
Python
Flask
Scikit-learn
HTML, CSS
Pickle (for saving trained models)

## Installation
Clone the repository:
git clone https://github.com/your-username/breast-cancer-prediction.git

Navigate to the project folder:
cd breast-cancer-prediction

Install dependencies:
pip install -r requirements.txt

Run the Flask app:
python app.py

Open your browser and visit:
http://127.0.0.1:5000/

## Model Training
The Logistic Regression and KNN models were trained on all 30 features of the dataset.
Both models were saved using pickle and loaded into the Flask app for predictions.

## Acknowledgments
Scikit-learn
Flask
Wisconsin Breast Cancer Dataset
```python
from sklearn.datasets import load_breast_cancer
