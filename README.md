# Diabetes Prediction using Support Vector Machine (SVM)

## 📌 Project Overview

This project uses a Support Vector Machine (SVM) model to predict diabetes based on a given dataset. The goal is to enhance model accuracy beyond 88% through data preprocessing, feature engineering, and model optimization.

## 📂 Dataset

The dataset used is diabetes.csv, which contains various medical attributes related to diabetes prediction.

Features include Glucose Level, Blood Pressure, BMI, Age, Insulin, and more.

The target variable (Outcome) is 0 for non-diabetic and 1 for diabetic patients.

### 🔬 Steps Followed

#### 1️⃣ Data Preprocessing

- Checked for missing values and handled zero values in key columns.
- Replaced zeros in columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI with their median values.
- Scaled the features using StandardScaler.
- Balanced the dataset using SMOTE (Synthetic Minority Over-sampling Technique).

#### 2️⃣ Model Training & Optimization

- Used SVM with different kernels (rbf, poly, sigmoid).
- Applied GridSearchCV to fine-tune hyperparameters:
  - C (Regularization Parameter)
  - gamma (Kernel Coefficient)
  - kernel (Kernel Type)
- Evaluated model performance using Accuracy, Precision, Recall, and F1-Score.

#### 3️⃣ Achieved Accuracy

The optimized SVM model achieved an accuracy of over 88% after applying hyperparameter tuning and data balancing techniques.

### 🛠 Installation & Usage

#### 🔹 Prerequisites

Ensure you have Python 3.x installed along with the required libraries:

```bash
pip install numpy pandas seaborn scikit-learn imbalanced-learn matplotlib

Ensure you have Python 3.x installed along with the required libraries:

pip install numpy pandas seaborn scikit-learn imbalanced-learn matplotlib

🔹 Running the Project
Clone this repository:

bash
git clone https://github.com/MarahSaleem/Diabetes-Prediction-Analysis.git
Navigate to the project directory:

bash
cd Diabetes-Prediction-Analysis
Run the Jupyter Notebook (.ipynb) using:

bash
jupyter notebook
Execute the cells to train and evaluate the model.

📊 Results & Future Improvements
The SVM model successfully predicted diabetes with high accuracy.

Further improvements can be made by:

Testing with Deep Learning models (e.g., Neural Networks).
Adding more relevant features to the dataset.
Collecting more real-world patient data for better generalization.
🤝 Contributing
Feel free to fork the repository and submit a pull request if you have suggestions or improvements!

📜 License
This project is open-source and available under the MIT License.
