# Comment Category Prediction

A hybrid **NLP + Machine Learning pipeline** for classifying user-generated comments into categories using both textual and structured features.

## Overview
This project focuses on predicting the category of online comments by combining:
- **Textual features** extracted using TF-IDF
- **Structured metadata** such as engagement and temporal signals

The model leverages both sources of information to improve classification performance on real-world discussion data.


## Key Features
- Hybrid modeling: **NLP + tabular data integration**
- TF-IDF vectorization with:
  - 18,000 features  
  - Uni-grams + Bi-grams
- Feature engineering:
  - Engagement score (upvotes - downvotes)
  - Total votes
  - Time-based features (year, month, day, hour)
- Evaluation of multiple models:
  - Logistic Regression  
  - SGD Classifier  
  - Naive Bayes  
  - SVM  
  - KNN  
  - LightGBM  
  - XGBoost  
- Ensemble prediction using boosting models


## Methodology

### 1. Data Preprocessing
- Handled missing text values
- Converted categorical/binary features
- Cleaned and standardized dataset

### 2. Feature Engineering
- Extracted temporal features from timestamps
- Created engagement-based metrics
- Scaled numeric features

### 3. Text Representation
- Applied **TF-IDF Vectorizer**
- Captured contextual meaning using n-grams (1,2)

### 4. Model Training
- Train-validation split (80-20)
- Compared multiple classification models
- Tuned boosting models for optimal performance

### 5. Ensemble Strategy
- Combined predictions from LightGBM and XGBoost
- Improved robustness and generalization


## Results
- Achieved **~90% validation accuracy**
- Boosting models (LightGBM, XGBoost) performed best
- Hybrid feature approach significantly improved results over text-only models


## Tech Stack
- **Language:** Python  
- **Libraries:**  
  - Pandas, NumPy  
  - Scikit-learn  
  - LightGBM  
  - XGBoost  
  - SciPy  


## Dataset
The dataset contains:
- Comment text  
- Engagement metrics (upvotes, downvotes)  
- Metadata (timestamps, categories, indicators)

> Source: Kaggle Competition – Comment Category Prediction Challenge
