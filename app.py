import streamlit as st
import pandas as pd
import pickle
st.title('🌾 Crop Yield Prediction')

st.markdown("""
Welcome to the Crop Yield Prediction App! 
This tool uses a machine learning model to forecast agricultural output based on key factors. 
**Use the sidebar** on the left to input values and see the predicted yield.
""")

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get the list of crops, seasons, and states for the dropdowns
df = pd.read_csv('crop_yield.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()


# --- Streamlit App ---

st.title('🌾 Crop Yield Prediction')

# --- User Input in a Sidebar ---
st.sidebar.header('Input Features')

# Dropdowns for categorical features
crop = st.sidebar.selectbox('Crop', sorted(df['crop'].unique()))
season = st.sidebar.selectbox('Season', sorted(df['season'].unique()))
state = st.sidebar.selectbox('State', sorted(df['state'].unique()))

# Sliders for numerical features
crop_year = st.sidebar.slider('Crop Year', 1997, 2020, 2015)
area = st.sidebar.slider('Area (Hectares)', float(df['area'].min()), float(df['area'].max()), float(df['area'].mean()))
annual_rainfall = st.sidebar.slider('Annual Rainfall (mm)', float(df['annual_rainfall'].min()), float(df['annual_rainfall'].max()), float(df['annual_rainfall'].mean()))
fertilizer = st.sidebar.slider('Fertilizer (tonnes)', float(df['fertilizer'].min()), float(df['fertilizer'].max()), float(df['fertilizer'].mean()))
pesticide = st.sidebar.slider('Pesticide (tonnes)', float(df['pesticide'].min()), float(df['pesticide'].max()), float(df['pesticide'].mean()))


# --- Prediction ---
if st.sidebar.button('Predict Yield'):
    # Create a dataframe from user inputs
    input_data = pd.DataFrame({
        'crop': [crop],
        'crop_year': [crop_year],
        'season': [season],
        'state': [state],
        'area': [area],
        'annual_rainfall': [annual_rainfall],
        'fertilizer': [fertilizer],
        'pesticide': [pesticide]
    })

    # One-hot encode the categorical features
    # This should match the encoding used during model training
    input_data = pd.get_dummies(input_data, columns=['crop', 'season', 'state'])
    
    # Align the columns of the input data with the training data
    # Get the column names from the training data (X_train)
    # You would need to save these column names from your notebook
    # For now, we will create a placeholder. 
    # Replace this with the actual columns from your training data.
    train_cols = model.feature_names_ # Get columns from CatBoost
    input_data = input_data.reindex(columns=train_cols, fill_value=0)


    # Make prediction
    prediction = model.predict(input_data)

    st.subheader('Predicted Crop Yield')
    st.write(f'The predicted crop yield is **{prediction[0]:.2f} tonnes per hectare**.')


# --- Adding Visualizations to Streamlit app ---
st.header("Project Visualizations")

# Example of embedding a plot from your EDA
st.subheader("Average Yield Over Time")
avg_yield = df.groupby('crop_year')['yield'].mean()
st.line_chart(avg_yield)

# You can add more plots here, like the correlation heatmap or feature importance plot.
# For matplotlib/seaborn plots, you can use st.pyplot()

# correlation heatmap
st.subheader("Correlation Heatmap of Numerical Features")
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 

# This is the key line: it selects ONLY the number columns for the heatmap
numeric_df = df.select_dtypes(include=np.number)

fig, ax = plt.subplots(figsize=(10, 8))
# Notice we are now using numeric_df.corr() instead of df.corr()
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# --- Adding Visualizations to Streamlit app ---
st.header("Project Visualizations")

# (Keep your existing line chart and heatmap code here...)


# --- Feature Importance Plot ---
st.subheader("What Matters Most for Crop Yields?")

# Check if the model has the 'feature_importances_' attribute
if hasattr(model, 'feature_importances_'):
    # Create a dataframe for feature importances
    # The FIX is here: we use model.feature_names_
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, model.feature_names_)), columns=['Value','Feature'])

    # Create the plot
    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False), ax=ax_imp)
    ax_imp.set_title('Model Feature Importance')
    ax_imp.set_xlabel('Importance')
    ax_imp.set_ylabel('Features')
    st.pyplot(fig_imp)

    st.markdown("""
    This chart shows the most influential factors in our model's predictions. The factors at the top are the most important drivers of crop yield in this dataset.
    """)
else:
    st.write("The loaded model does not support feature importance.")