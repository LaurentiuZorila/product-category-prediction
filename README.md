# Product Category Prediction

Machine Learning project for automatic classification of product titles into categories.  
This project implements the full solution required in **Sarcina 3 – Predictia categoriei produsului pe baza titlului** (Introduction to Machine Learning using Python).

The goal is to build a complete ML pipeline that can assign the correct category to any new product based only on its title.

---

## 📦 Project Structure

```text
product-category-prediction/
│
├── data/
│     └── products.csv
│
├── notebooks/
│     └── analysis.ipynb
│
├── src/
│     ├── train_model.py
│     └── predict_category.py
│
├── models/
│     ├── final_model.pkl
│     └── tfidf_vectorizer.pkl
│
└── README.md
```
## How to Install and Run
1. Create virtual environment (optional)
```
python -m venv .venv
source .venv/bin/activate       # macOS / Linux
.venv\Scripts\activate          # Windows
```
2. Install dependencies
```pip install pandas scikit-learn numpy```
## Train the Model
Run the training script:
```python src/train_model.py```

This script will:

 - load the dataset
 - clean and preprocess the product titles
 - vectorize text using TF-IDF
 - train Logistic Regression, Random Forest, Linear SVC
 - pick the best performing model
 - save the trained model and the TF-IDF vectorizer

Output:
```
models/final_model.pkl
models/tfidf_vectorizer.pkl
```

## Test the Model Interactively
Run: 
```text
python src/predict_category.py
```
You can test titles like:
```text
iphone 7 32gb gold
smeg sbs8004po
bosch serie 4 kgv39vl31g
```
The script loads the trained model and prints the predicted category.

## Notebook Contents
The analysis.ipynb notebook includes:

 - dataset exploration
 - cleaning and preprocessing
 - feature engineering attempts
 - TF-IDF vectorization
 - model comparison
 - evaluation metrics
 - confusion matrix and heatmap
 - conclusions

## Algorithms Used
Models tested:
 - Logistic Regression
 - Random Forest Classifier
 - Linear SVC

The best model is saved as ```final_model.pkl```