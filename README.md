# 🔐 Hybrid URL Phishing Detection System (AI + Blacklist)

This project implements a **hybrid phishing detection system** that combines traditional **blacklist-based detection** with advanced **machine learning models** to accurately classify URLs as **phishing** or **legitimate**. The system is designed to handle both known threats (via blacklist) and unknown/zero-day attacks (via ML models).

---

## 🚀 Key Features
- Hybrid detection approach (Blacklist + Machine Learning)
- Real-time URL classification (Phishing / Legitimate)
- Detection of **zero-day phishing attacks**
- Feature extraction from URLs (lexical, domain-based, security indicators)
- Data preprocessing including missing value handling and normalization
- Model comparison and performance evaluation
- Modular pipeline for scalability and improvement

---

## 🧠 Machine Learning Models Used
- Random Forest Classifier (RF)
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN / MLP)

---

## ⚙️ System Architecture
1. **Blacklist Check**  
   - URL is first checked against a database of known malicious domains  
   - Immediate classification if found  

2. **Feature Extraction**  
   - URL-based features (length, special characters, keywords, etc.)  
   - Domain & security-based attributes  

3. **Data Preprocessing**  
   - Missing value handling using `SimpleImputer`  
   - Feature scaling using `StandardScaler`  

4. **Machine Learning Classification**  
   - Trained models (RF, KNN, ANN) predict whether URL is phishing or legitimate  

5. **Final Prediction**  
   - Combined result ensures high accuracy and faster detection  

---

## 📊 Dataset
- File: `Dataset_useful_top20.csv`
- Target Column: `Type`
- Labels:
  - `0` → Legitimate URL  
  - `1` → Phishing URL  

---

## 🛠️ Technologies Used
- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn (RF, KNN), TensorFlow/Keras (ANN)  
- **Data Processing:** Pandas, NumPy  
- **Feature Engineering:** URL-based + Domain-based features  
- **Visualization:** Matplotlib, Seaborn  
- **Other Tools:** Jupyter Notebook, Git  

---

## 📈 Evaluation Metrics
- Accuracy Score  
- Confusion Matrix  
- Classification Report (Precision, Recall, F1-score)  

---

## 💡 Project Highlights
- Combines **speed of blacklist detection** with **intelligence of ML models**  
- Capable of detecting **previously unseen phishing URLs**  
- Demonstrates real-world application of **AI in cybersecurity**  

---
