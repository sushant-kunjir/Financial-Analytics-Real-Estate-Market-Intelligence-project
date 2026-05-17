# Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Project Overview
This project focuses on analyzing buyer behavior in the real estate market using Machine Learning clustering techniques. The objective is to identify hidden customer segments and understand investment patterns for better market intelligence and customer targeting.

The project combines:
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- K-Means Clustering
- Hierarchical Clustering
- PCA Visualization
- Tableau Dashboard Visualization

---

# Problem Statement

Real estate buyers have diverse investment behaviors such as:
- Individual home buyers
- Corporate buyers
- Luxury investors
- First-time buyers
- International investors

Traditional analytics treat all buyers similarly, which leads to:
- Poor customer targeting
- Inefficient marketing campaigns
- Missed investment opportunities

This project applies Machine Learning clustering algorithms to segment buyers into meaningful groups based on:
- Demographics
- Geographic location
- Property investment behavior
- Financing patterns
- Customer satisfaction

---

# Objectives

- Perform buyer segmentation using Machine Learning
- Identify investment behavior patterns
- Analyze loan and financing trends
- Study geographic buyer distribution
- Visualize clusters using PCA and Tableau dashboards
- Generate actionable business insights

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data preprocessing & Machine Learning |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine Learning algorithms |
| Tableau | Interactive dashboards |
| Jupyter Notebook | Development environment |

---

# Dataset Information

The project uses two datasets:

## 1. Clients Dataset
Contains buyer/customer information.

### Important Fields
- client_id
- client_type
- gender
- country
- region
- acquisition_purpose
- loan_applied
- referral_channel
- satisfaction_score

---

## 2. Properties Dataset
Contains property transaction details.

### Important Fields
- listing_id
- sale_price
- floor_area_sqft
- unit_category
- transaction_date
- listing_status
- client_ref

---

# Project Workflow

```text
Load Datasets
      ↓
Merge Datasets
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Encoding
      ↓
Feature Scaling
      ↓
K-Means Clustering
      ↓
Hierarchical Clustering
      ↓
PCA Visualization
      ↓
Business Insights
      ↓
Tableau Dashboard
```

---

# Data Preprocessing

## Steps Performed
- Removed duplicates
- Handled missing values
- Converted sale_price to numeric
- Converted date_of_birth into Age
- Encoded categorical variables
- Standardized numerical features

---

# Feature Engineering

## Age Calculation

```python
df['Age'] = current_year - df['date_of_birth'].dt.year
```

---

# Feature Scaling

StandardScaler was used to normalize features before clustering.

Formula:

```text
z = (x - μ) / σ
```

---

# Machine Learning Models

## 1. K-Means Clustering

Used to group buyers into similar segments.

### Evaluation Methods
- Elbow Method
- Silhouette Score

### K-Means Objective Function

```text
J = Σ Σ ||x - μ||²
```

---

## 2. Hierarchical Clustering

Used to validate clustering structure and visualize customer hierarchy.

---

# PCA Visualization

Principal Component Analysis (PCA) was used to reduce high-dimensional data into 2D space for cluster visualization.

---

# Tableau Dashboard

The Tableau dashboard includes:
- Cluster Distribution
- PCA Scatter Plot
- Geographic Buyer Analysis
- Loan Behavior Analysis
- Sale Price Analysis
- Buyer Purpose Analysis
- KPI Cards

---

# Key Insights

- Luxury investors tend to purchase high-value properties.
- First-time buyers are more loan dependent.
- Corporate buyers invest in larger properties.
- Geographic regions show different investment behaviors.
- Customer satisfaction varies across buyer segments.

---

# Business Impact

This project helps real estate companies:
- Improve customer segmentation
- Personalize marketing campaigns
- Identify high-value investors
- Understand regional investment trends
- Optimize business decision-making

---

# Future Improvements

- Deploy Streamlit web application
- Add predictive analytics
- Integrate real-time market data
- Build recommendation system
- Use advanced clustering algorithms

---

# Conclusion

This project demonstrates how Machine Learning and Business Intelligence can be combined to analyze real estate buyer behavior effectively. By applying clustering techniques and visualization tools, meaningful customer segments and investment patterns were identified, enabling smarter business strategies and data-driven decision making.

---

