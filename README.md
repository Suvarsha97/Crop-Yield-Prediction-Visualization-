# Crop Yield Prediction & Visualization

## Overview

This project is a data-driven system designed to forecast crop yields in India. It leverages machine learning to predict agricultural output based on historical data, including weather patterns (rainfall) and farming practices (fertilizer and pesticide usage). The application provides an interactive interface for users to input various parameters and receive an instant yield prediction, alongside insightful data visualizations.

This project was developed to explore the application of machine learning in agriculture, aiming to provide a tool that could potentially assist in planning and decision-making for farmers and agricultural organizations.



##  Features

* **Interactive Yield Prediction:** An easy-to-use interface to predict crop yield (in tonnes per hectare) based on user-selected inputs.
* **Data Visualization:** Includes interactive charts and maps to explore historical trends and relationships between different agricultural factors.
* **Feature Importance Analysis:** Shows which factors (e.g., rainfall, fertilizer) are the most influential in the model's predictions.
* **Multiple Model Training:** The core notebook explores and compares different regression models, including CatBoost, Random Forest, and XGBoost.

---

## 🛠 How It Works (Technology Stack)

The project is built with the following technologies:

* **Backend & Modeling:** Python, Pandas, Scikit-learn, CatBoost
* **Frontend & UI:** Streamlit
* **Data Visualization:** Matplotlib, Seaborn
* **Jupyter Notebook** for data exploration, model training, and evaluation.

The machine learning model (`model.pkl`) was trained on a historical dataset of Indian crop production. The Streamlit application loads this pre-trained model to make live predictions.

**Live App Link:** **[➡️ Click here to view the live app!](https://cropyeildpredictionvisualization.streamlit.app/)**

##  Screenshots

![Crop Yield App Screenshot](<img width="1914" height="948" alt="image" src="https://github.com/user-attachments/assets/de68e449-23ce-4c91-ab56-1014c4b50416" />)

