# 🏥 Synthetic Medical Diagnosis Record Generation Using CTGAN

## 📌 Project Overview

This project focuses on generating realistic synthetic medical diagnosis records using **Generative AI**, specifically a **Conditional Tabular Generative Adversarial Network (CTGAN)**.

Healthcare datasets contain sensitive patient information, making data sharing and machine learning development challenging. This project uses CTGAN to learn patterns from a real healthcare dataset and generate synthetic patient records that can be used for research, experimentation, data augmentation, and machine learning development.

The synthetic data is evaluated by comparing it with the original dataset using statistical analysis and visualization techniques. The original and synthetic datasets are then used to train and evaluate a **Random Forest classification model** for diabetes prediction.

---

## 🎯 Objectives

* Analyze and preprocess healthcare data.
* Perform Exploratory Data Analysis (EDA).
* Train a CTGAN model on healthcare data.
* Generate synthetic medical diagnosis records.
* Compare real and synthetic data distributions.
* Augment the original dataset using synthetic records.
* Train a Random Forest classification model.
* Compare model performance using original and augmented datasets.
* Evaluate the usefulness of synthetic data for machine learning.
* Demonstrate a privacy-preserving approach to healthcare data generation.

---

## 🧠 Generative AI Model

### CTGAN

**CTGAN (Conditional Tabular Generative Adversarial Network)** is a Generative AI model designed for generating synthetic tabular data.

It consists of two main components:

### Generator

The Generator creates synthetic patient records based on the patterns learned from the real dataset.

### Discriminator

The Discriminator attempts to distinguish between real and synthetic records.

Through adversarial training, the Generator improves its ability to generate realistic synthetic records.

---

## 📊 Dataset

### Diabetes Prediction Dataset

The project uses a structured healthcare dataset containing patient health and lifestyle information.

### Features

* `gender`
* `age`
* `hypertension`
* `heart_disease`
* `smoking_history`
* `bmi`
* `HbA1c_level`
* `blood_glucose_level`

### Target Variable

* `diabetes`

  * `0` → Non-diabetic
  * `1` → Diabetic

---

## 🏗️ Project Workflow

```text
                    ┌────────────────────────┐
                    │  Real Healthcare Data   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Data Cleaning & EDA     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Train CTGAN Model     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Generate Synthetic Data │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Real vs Synthetic Data │
                    │       Validation       │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Data Augmentation    │
                    │ Real + Synthetic Data  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Random Forest Model    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Model Evaluation       │
                    └────────────────────────┘
```

---

## 🔬 Project Methodology

### 1. Data Collection

A structured healthcare dataset was selected for the project.

### 2. Data Preprocessing

The following preprocessing steps were performed:

* Missing value analysis
* Duplicate record detection
* Duplicate removal
* Categorical feature encoding
* Numerical feature preprocessing
* Data validation

### 3. Exploratory Data Analysis

EDA was performed to understand:

* Patient age distribution
* BMI distribution
* Blood glucose distribution
* HbA1c distribution
* Diabetes class distribution
* Correlation between healthcare features

### 4. CTGAN Training

The CTGAN model was trained on the cleaned healthcare dataset.

The model learned the statistical relationships between the different patient features.

### 5. Synthetic Data Generation

After training, CTGAN was used to generate synthetic medical diagnosis records.

The generated records were saved as:

```text
synthetic_patient_data.csv
```

### 6. Data Validation

The synthetic data was compared with the original data using:

* Statistical summaries
* Distribution comparison
* Correlation heatmaps
* PCA
* t-SNE visualization

### 7. Data Augmentation

The real and synthetic datasets were combined:

```text
Original Healthcare Data
          +
Synthetic Healthcare Data
          =
Augmented Dataset
```

### 8. Machine Learning

A Random Forest Classifier was trained for diabetes prediction.

Two experiments were performed:

#### Experiment 1

Model trained using original healthcare data.

#### Experiment 2

Model trained using augmented data containing real and synthetic records.

### 9. Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Generative AI

* CTGAN
* SDV (Synthetic Data Vault)

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Classifier

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Google Colab
* Jupyter Notebook

---

## 📁 Project Structure

```text
Healthcare-CTGAN-Project/
│
├── 📂 Data/
│   ├── diabetes_prediction_dataset.csv
│   ├── clean_diabetes_dataset.csv
│   ├── synthetic_patient_data.csv
│   └── augmented_dataset.csv
│
├── 📂 Notebooks/
│   ├── 1_EDA.ipynb
│   ├── 2_CTGAN_Training.ipynb
│   ├── 3_Model_Training.ipynb
│   └── 4_Evaluation_Visualization.ipynb
│
├── 📂 Models/
│   ├── ctgan_model.pkl
│   └── random_forest.pkl
│
├── 📂 Images/
│   ├── original_heatmap.png
│   ├── synthetic_heatmap.png
│   ├── age_distribution.png
│   ├── bmi_distribution.png
│   ├── blood_glucose_distribution.png
│   ├── hba1c_distribution.png
│   ├── PCA_plot.png
│   ├── tsne_plot.png
│   └── feature_importance.png
│
├── 📂 Report/
│   └── Final_Report.pdf
│
├── 📂 PPT/
│   └── Project_Presentation.pptx
│
├── 📄 README.md
└── 📄 requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Healthcare-CTGAN-Project.git
```

Navigate to the project directory:

```bash
cd Healthcare-CTGAN-Project
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Create a file named:

```text
requirements.txt
```

Add:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
sdv
joblib
jupyter
```

Install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Project

### Step 1: Run EDA

Open:

```text
Notebooks/1_EDA.ipynb
```

Perform:

* Data cleaning
* Duplicate removal
* Encoding
* Exploratory data analysis

Output:

```text
clean_diabetes_dataset.csv
```

---

### Step 2: Train CTGAN

Open:

```text
Notebooks/2_CTGAN_Training.ipynb
```

Train the CTGAN model and generate synthetic healthcare records.

Outputs:

```text
synthetic_patient_data.csv
ctgan_model.pkl
```

---

### Step 3: Train Machine Learning Model

Open:

```text
Notebooks/3_Model_Training.ipynb
```

Combine original and synthetic data.

Output:

```text
augmented_dataset.csv
random_forest.pkl
```

---

### Step 4: Evaluate Results

Open:

```text
Notebooks/4_Evaluation_Visualization.ipynb
```

Generate:

* PCA visualization
* t-SNE visualization
* Distribution comparison
* Correlation heatmaps
* Feature importance plots
* Evaluation summaries

---

## 📈 Model Evaluation

The performance of the model trained using original data is compared with the model trained using augmented data.

| Metric    | Original Data | Augmented Data |
| --------- | ------------: | -------------: |
| Accuracy  |    Add Result |     Add Result |
| Precision |    Add Result |     Add Result |
| Recall    |    Add Result |     Add Result |
| F1-Score  |    Add Result |     Add Result |
| ROC-AUC   |    Add Result |     Add Result |

> Replace the values above with the actual results generated from the project notebooks.

---

## 📊 Key Visualizations

The project includes:

* Original healthcare data distributions
* Synthetic healthcare data distributions
* Real vs synthetic age comparison
* Real vs synthetic BMI comparison
* Real vs synthetic blood glucose comparison
* Real vs synthetic HbA1c comparison
* Original data correlation heatmap
* Synthetic data correlation heatmap
* PCA visualization
* t-SNE visualization
* Confusion matrix
* ROC curve
* Feature importance

---

## 🔐 Privacy Considerations

Healthcare data is highly sensitive.

Synthetic data can help reduce the need to directly share real patient records. However, synthetic data must still be carefully evaluated.

Important considerations include:

* Privacy leakage
* Memorization of real records
* Bias replication
* Data quality
* Statistical similarity
* Clinical validity

Synthetic data should not automatically be considered completely private or suitable for clinical diagnosis.

---

## ⚠️ Limitations

* CTGAN-generated records may not perfectly represent real-world patient populations.
* The model can reproduce biases present in the original dataset.
* Synthetic records require statistical validation.
* Synthetic healthcare data should not be used directly for clinical decisions without medical validation.
* Results depend on the quality and size of the original dataset.

---

## 🚀 Future Scope

Future improvements may include:

* Comparing CTGAN with TVAE and Gaussian Copula models.
* Applying differential privacy techniques.
* Generating synthetic healthcare time-series data.
* Generating synthetic ECG signals.
* Generating synthetic medical imaging data.
* Deploying the model using Streamlit.
* Creating a REST API using FastAPI.
* Performing advanced privacy risk evaluation.
* Applying fairness and bias detection techniques.

---

## 🎓 Internship Project

This project was developed as part of an:

### AI/ML Internship

**Project Domain:** Healthcare
**Project Type:** Generative AI and Data Augmentation
**Generative AI Model:** CTGAN
**Machine Learning Model:** Random Forest

---

## 👨‍💻 Author

**Simran Bhadouria**

AI/ML Intern

---

## 📄 Disclaimer

This project is intended for educational and research purposes only.

The generated synthetic medical records are not real patient records and must not be used for clinical diagnosis, medical treatment, or healthcare decision-making.

---
