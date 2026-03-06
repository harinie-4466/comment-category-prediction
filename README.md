# Comment Category Prediction (Machine Learning)

## Overview

This project builds a machine learning model to classify comments into predefined categories using both **text features and structured metadata**.

The model processes comment text using **TF-IDF vectorization** and combines it with additional numeric and categorical features to improve prediction performance.

The final model is trained using **Logistic Regression**.

---

## Dataset

The dataset contains comments along with additional attributes such as:

* comment (text input)
* upvote count
* downvote count
* gender
* race
* religion
* disability
* additional indicator variables (`if_1`, `if_2`)

The target variable is:

* **label** → category assigned to the comment

---

## Workflow

### 1. Data Loading

The dataset is loaded from CSV files:

* `train.csv`
* `test.csv`

### 2. Data Cleaning

Categorical attributes are standardized by:

* converting to lowercase
* fixing spelling errors
* replacing missing values with `unknown`

### 3. Feature Engineering

#### Text Features

Comments are converted to numerical vectors using:

TF-IDF Vectorizer (max_features = 3000)

#### Numerical Features

Additional features used:

* upvote
* downvote
* if_1
* if_2
* gender
* race
* religion
* disability

### 4. Feature Combination

Text and numerical features are combined using sparse matrix stacking.

### 5. Model Training

A **Logistic Regression classifier** is trained using:

* 80% training data
* 20% validation data

### 6. Evaluation

Model performance is evaluated using:

* Accuracy
* Classification Report

### 7. Prediction

Predictions are generated for the test dataset and saved as:

`submission.csv`

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Matplotlib / Seaborn

---

## Output

The final predictions are exported as:

submission.csv

This file follows the format required for submission to the competition.

---

## Future Improvements

Possible improvements include:

* using advanced NLP models (BERT, RoBERTa)
* hyperparameter tuning
* feature selection
* handling class imbalance
* deep learning approaches

---

## Disclaimer

This project is created for educational and machine learning experimentation purposes.
