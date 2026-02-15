 # 📊 Content Monetization Modeler

## 🚀 Project Overview

Content Monetization Modeler is a Machine Learning project that predicts YouTube ad revenue using video performance metrics and contextual features.

This project builds and evaluates multiple regression models to estimate `ad_revenue_usd` and deploys the best-performing model using a Streamlit web application.

---

## 🎯 Problem Statement

As video creators and media companies increasingly rely on YouTube for income, predicting potential ad revenue becomes essential for business planning and content strategy.

This project builds a regression model that accurately estimates ad revenue for individual videos based on performance metrics such as views, likes, watch time, subscribers, and contextual features.

---

## 🧠 Skills & Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib & Seaborn
- Feature Engineering
- Data Cleaning
- Outlier Detection
- Categorical Encoding
- Model Evaluation (R², RMSE, MAE)
- Streamlit
- Git & GitHub

---

## 📂 Dataset Information

- **Name:** YouTube Monetization Modeler
- **Format:** CSV
- **Rows:** ~122,000
- **Target Variable:** `ad_revenue_usd`
- **Type:** Synthetic dataset for learning purposes

### Key Features:
- video_id
- date
- views
- likes
- comments
- watch_time_minutes
- video_length_minutes
- subscribers
- category
- device
- country
- ad_revenue_usd (Target)

---

## ⚙️ Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Data inspection
- Correlation analysis
- Outlier detection
- Revenue distribution analysis
- Feature relationship visualization

### 2️⃣ Data Preprocessing
- Handled ~5% missing values
- Removed ~2% duplicate records
- Encoded categorical variables
- Feature scaling
- Created new feature:
  - Engagement Rate = (likes + comments) / views

### 3️⃣ Model Building
Tested multiple regression models:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

### 4️⃣ Model Evaluation Metrics
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### 5️⃣ Streamlit Web App
- User input-based revenue prediction
- Interactive dashboard
- Basic visualization of model insights

---

## 📊 Results

- Trained and optimized regression model
- Identified key revenue-driving features
- Built fully functional Streamlit app
- Generated business insights for content optimization

---

## 💡 Business Use Cases

- Content Strategy Optimization
- Revenue Forecasting
- Creator Analytics Tools
- Ad Campaign ROI Planning

---

DATASET :- LINk["https://drive.google.com/drive/folders/1ybhXuva11b6zm20j35vXC33E75FS0sHN"](url)
