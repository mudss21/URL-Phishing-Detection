# URL-Based Phishing Detection Using Machine Learning

This project implements a URL-based phishing detection system using multiple machine learning algorithms. The objective is to accurately classify URLs as **phishing** or **legitimate** based on extracted lexical and statistical features.

---

## 🚀 Features
- Handles missing values using median imputation
- Applies feature standardization for better model performance
- Implements and compares multiple ML classifiers
- Provides detailed evaluation using accuracy, confusion matrix, and classification report

---

## 🧠 Machine Learning Models Used
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN / MLP)

---

## 📊 Workflow
1. Load and preprocess dataset  
2. Split data into training and testing sets  
3. Handle missing values using `SimpleImputer`  
4. Normalize features using `StandardScaler`  
5. Train multiple classification models  
6. Evaluate performance using standard metrics  

---

## 📁 Dataset
- File: `Dataset(2).csv`
- Target Column: `Type`
- Labels:
  - `0` → Legitimate URL  
  - `1` → Phishing URL  

---

## 🛠️ Technologies Used
- Python
- Pandas
- Scikit-learn

---

