# 🧬 Obesity Level Estimation — Data Analysis & Prediction Platform

> An end-to-end data science web application that analyzes obesity risk factors based on eating habits and physical condition, powered by machine learning models and an AI chatbot.

---

## 📌 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)
- [Analysis Sections](#analysis-sections)
---

## 📖 About the Project

This project investigates the relationship between lifestyle habits and obesity levels using a dataset of 2,100+ individuals. It combines **statistical analysis**, **machine learning**, and a **Streamlit web application** to make findings accessible to both technical and non-technical audiences.

An integrated **AI chatbot (Gemini)** reads the live analysis results and answers questions in plain English — making this more than just a notebook, but a fully interactive data product.

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository — Obesity Levels Dataset
- **File:** `ObesityDataSet_raw_and_data_sinthetic.csv`
- **Records:** 2,111 individuals
- **Features:** 17 variables

| Variable | Description |
|----------|-------------|
| `Gender` | Male / Female |
| `Age` | Age in years |
| `Height` | Height in meters |
| `Weight` | Weight in kg (regression target) |
| `family_history_with_overweight` | Family obesity history |
| `FAVC` | Frequent consumption of high-caloric food |
| `FCVC` | Frequency of vegetable consumption |
| `NCP` | Number of main meals per day |
| `CAEC` | Eating between meals |
| `SMOKE` | Smoking habit |
| `CH2O` | Daily water intake |
| `SCC` | Calories consumption monitoring |
| `FAF` | Physical activity frequency |
| `TUE` | Time using technology devices |
| `CALC` | Alcohol consumption |
| `MTRANS` | Transportation mode |
| `NObeyesdad` | Obesity level (classification target) |

**Obesity Labels (WHO + Mexican Normativity):**
- Underweight — BMI < 18.5
- Normal — BMI 18.5 to 24.9
- Overweight — BMI 25.0 to 29.9
- Obesity I — BMI 30.0 to 34.9
- Obesity II — BMI 35.0 to 39.9
- Obesity III — BMI > 40

---

## ✨ Features

- 📁 **CSV Upload** — drag and drop your dataset directly in the browser
- 📊 **Interactive EDA** — dropdown to explore all 17 variables with charts and stats
- 🧪 **Hypothesis Testing** — T-test, Chi-square, Pearson Correlation with plain English conclusions
- 🧩 **Exploratory Factor Analysis** — eigenvalues, scree plot, factor loadings, variance explained
- 🔵 **KMeans Clustering** — elbow method + interactive K slider
- 📈 **Linear Regression** — predict weight, actual vs predicted plot, residuals, feature importance
- 🎯 **LDA Classification** — classify obesity levels, confusion matrix, classification report
- 🤖 **AI Chatbot** — Gemini-powered assistant with full awareness of your live results

---

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Statistics | SciPy |
| Factor Analysis | factor_analyzer |
| AI Chatbot | Google Gemini API (gemini-1.5-flash) |

---

## 📁 Project Structure

```
obesity-analysis/
│
├── app.py                                        # Main Streamlit application
├── requirements.txt                              # Python dependencies
├── ObesityDataSet_raw_and_data_sinthetic.csv     # Dataset (add manually)
├── DARP_Project_Notebook.ipynb                   # Original Jupyter notebook
└── README.md                                     # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- A free Gemini API key from [aistudio.google.com](https://aistudio.google.com)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/obesity-analysis.git
cd obesity-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Gemini API key**

Open `app.py` and replace:
```python
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```
with your actual key.

**4. Run the app**
```bash
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## 🖥️ How to Use

1. **Upload your dataset** using the sidebar file uploader
2. **Navigate** between sections using the sidebar radio buttons
3. **Explore** each analysis section — charts and results generate live
4. **Ask the AI chatbot** questions about your results in plain English

---

## 📚 Analysis Sections

### 📊 Overview
Dataset shape, preview, and descriptive statistics at a glance.

### 🔍 EDA
Select any of the 17 variables from a dropdown — instantly get a relevant chart (countplot, histogram, pie chart, barplot) plus value counts and descriptive stats.

### 🧪 Hypothesis Testing
Three statistical tests with automated conclusions:
- **H1:** Independent T-Test — Is there a significant weight difference between genders?
- **H2:** Chi-Square Test — Is high-caloric food consumption related to obesity level?
- **H3:** Pearson Correlation — Does physical activity frequency correlate with weight?

### 🧩 Exploratory Factor Analysis
- Bartlett's Test of Sphericity and KMO Measure of Sampling Adequacy
- Eigenvalue table with Kaiser criterion (retain factors with eigenvalue > 1)
- Color-coded scree plot
- Varimax-rotated factor loadings heatmap
- Most relevant variables per factor (loading threshold > 0.4)
- Cumulative variance explained table

### 🔵 Clustering
- KMeans with elbow method to find optimal K
- Interactive slider to change K and see cluster distribution live

### 📈 Linear Regression
- Predicts **Weight** from all other features
- Actual vs Predicted scatter plot
- Residuals distribution
- Top 10 feature importances by coefficient magnitude

### 🎯 Classification
- **LDA (Linear Discriminant Analysis)** to classify obesity levels
- Accuracy score and full classification report
- Confusion matrix heatmap

### 🤖 AI Assistant
- Powered by Google Gemini API
- Reads all live results as context
- Answers questions about methodology, results, and interpretations in plain English


## 📄 License

This project is for academic purposes under the DARP (Data Analysis Research Project) curriculum.

---

> Built with 🧬 by [Your Name] — feel free to fork, star, or reach out!
