❤️ Heart Disease Prediction using Machine Learning
📌 Project Overview

This project develops a supervised machine learning model to predict the presence of heart disease using structured clinical data. The objective is to classify patients into:

0 → No Heart Disease

1 → Heart Disease Present

The model is built using a complete end-to-end machine learning workflow including data preprocessing, exploratory analysis, model training, hyperparameter tuning, cross-validation, and performance evaluation.

📊 Dataset Description

The dataset contains clinical attributes collected from patients undergoing cardiovascular examination.

Features:

age – Age of the patient (years)

sex – Gender (1 = male, 0 = female)

chest_pain_type – Chest pain type (0–3)

resting_bp – Resting blood pressure (mm Hg)

cholestoral – Serum cholesterol (mg/dl)

fasting_blood_sugar – FBS > 120 mg/dl (1 = true)

restecg – Resting electrocardiographic results

max_hr – Maximum heart rate achieved

exang – Exercise induced angina

oldpeak – ST depression induced by exercise

slope – Slope of peak exercise ST segment

num_major_vessels – Number of major vessels (0–3)

thal – Thalassemia type

target – Output class (0 = healthy, 1 = disease)

🔎 Exploratory Data Analysis (EDA)

Distribution analysis of target variable

Boxplot visualization for outlier detection

Correlation heatmap for feature relationships

Duplicate row detection and removal

Class distribution inspection

⚙️ Data Preprocessing

Removed duplicate records

Verified missing values

Stratified train-test split (to preserve class balance)

No feature scaling applied (tree-based models used)

Train-Test Split:

70% Training

30% Testing

🤖 Model Development
Random Forest Classifier

Initial configuration:

RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)

📈 Model Evaluation
Test Set Performance

Accuracy: ~80%

Cross-Validation Accuracy: ~83%

Recall (Heart Disease): 0.91

OOB Score: Evaluated for model stability

Confusion Matrix
[[28 14]
 [ 4 45]]


Interpretation:

Strong detection of heart disease cases

Low false negatives (important in medical applications)

🔧 Hyperparameter Tuning

GridSearchCV was used to optimize model parameters:

param_grid = {
    'max_depth': [3, 5, 10],
    'n_estimators': [100, 200]
}


Best parameters selected based on cross-validation performance.

📊 Feature Importance

Top contributing features:

chest_pain_type

thal

oldpeak

max_hr

exang

num_major_vessels

These features significantly influence prediction outcomes.

🧠 Key Insights

Chest pain type is the strongest predictor.

The model prioritizes minimizing false negatives.

The dataset is moderately balanced.

Cross-validation confirms model generalization.

Tree-based models perform well without feature scaling.

🚀 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Launch the notebook
jupyter notebook heart_diseases.ipynb

📦 Requirements

Python 3.9+

pandas

numpy

matplotlib

seaborn

scikit-learn

📌 Project Structure
Heart-Disease-Prediction/
│
├── Raw_Datasets/               # Raw dataset files
│
├── heart_diseases.ipynb        # Jupyter notebook (EDA + model training)
├── heart_disease_model.pkl     # Saved trained model
│
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies

🏁 Conclusion

The Random Forest model achieves strong recall for heart disease detection while maintaining stable cross-validation performance. The model demonstrates practical effectiveness for binary medical classification tasks and highlights the importance of feature interpretation in healthcare prediction systems.

- 📄 License

MIT License

Copyright (c) 2026 Mann Rani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGSIN THE SOFTWARE