import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

def load_model_components():
    """Load the trained model and preprocessing components"""
    try:
        models_dir = 'saved_models'
        
        # Load model metadata first to get the correct model filename
        metadata = joblib.load(os.path.join(models_dir, 'model_metadata.joblib'))
        model_name = metadata['best_model_name'].replace(" ", "_").lower()
        
        # Load all components
        model = joblib.load(os.path.join(models_dir, f'best_wine_quality_model_{model_name}.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'feature_scaler.joblib'))
        encoder = joblib.load(os.path.join(models_dir, 'label_encoder.joblib'))
        feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
        
        return model, scaler, encoder, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None

def main():
    st.title("Wine Quality Predictor")
    st.write("Predict wine quality based on chemical properties")
    
    # Load model components
    model, scaler, encoder, feature_names, metadata = load_model_components()
    
    if model is None:
        st.stop()
    
    st.write("Enter the wine characteristics:")
    
    # Create input fields for each feature
    inputs = {}
    
    # Fixed acidity
    if 'fixed acidity' in feature_names:
        inputs['fixed acidity'] = st.number_input(
            "Fixed Acidity (g/L)", 
            min_value=3.8, 
            max_value=15.9, 
            value=7.0, 
            step=0.1
        )
    
    # Volatile acidity
    if 'volatile acidity' in feature_names:
        inputs['volatile acidity'] = st.number_input(
            "Volatile Acidity (g/L)", 
            min_value=0.08, 
            max_value=1.58, 
            value=0.4, 
            step=0.01
        )
    
    # Citric acid
    if 'citric acid' in feature_names:
        inputs['citric acid'] = st.number_input(
            "Citric Acid (g/L)", 
            min_value=0.0, 
            max_value=1.66, 
            value=0.3, 
            step=0.01
        )
    
    # Residual sugar
    if 'residual sugar' in feature_names:
        inputs['residual sugar'] = st.number_input(
            "Residual Sugar (g/L)", 
            min_value=0.6, 
            max_value=65.8, 
            value=5.0, 
            step=0.1
        )
    
    # Chlorides
    if 'chlorides' in feature_names:
        inputs['chlorides'] = st.number_input(
            "Chlorides (g/L)", 
            min_value=0.009, 
            max_value=0.611, 
            value=0.05, 
            step=0.001
        )
    
    # Free sulfur dioxide
    if 'free sulfur dioxide' in feature_names:
        inputs['free sulfur dioxide'] = st.number_input(
            "Free Sulfur Dioxide (mg/L)", 
            min_value=1.0, 
            max_value=289.0, 
            value=30.0, 
            step=1.0
        )
    
    # Total sulfur dioxide
    if 'total sulfur dioxide' in feature_names:
        inputs['total sulfur dioxide'] = st.number_input(
            "Total Sulfur Dioxide (mg/L)", 
            min_value=6.0, 
            max_value=440.0, 
            value=100.0, 
            step=1.0
        )
    
    # Density
    if 'density' in feature_names:
        inputs['density'] = st.number_input(
            "Density (g/cm³)", 
            min_value=0.987, 
            max_value=1.039, 
            value=0.995, 
            step=0.001
        )
    
    # pH
    if 'pH' in feature_names:
        inputs['pH'] = st.number_input(
            "pH Level", 
            min_value=2.72, 
            max_value=4.01, 
            value=3.2, 
            step=0.01
        )
    
    # Sulphates
    if 'sulphates' in feature_names:
        inputs['sulphates'] = st.number_input(
            "Sulphates (g/L)", 
            min_value=0.22, 
            max_value=2.0, 
            value=0.5, 
            step=0.01
        )
    
    # Alcohol
    if 'alcohol' in feature_names:
        inputs['alcohol'] = st.number_input(
            "Alcohol Content (%)", 
            min_value=8.0, 
            max_value=14.9, 
            value=10.0, 
            step=0.1
        )
    
    # Wine type
    if 'wine_type_encoded' in feature_names:
        wine_type = st.selectbox("Wine Type", ["Red", "White"])
        inputs['wine_type_encoded'] = 0 if wine_type == "Red" else 1
    
    # Predict button
    if st.button("Predict Wine Quality"):
        try:
            # Create input array in the correct order
            input_array = []
            for feature in feature_names:
                input_array.append(inputs[feature])
            
            # Convert to numpy array and reshape
            input_array = np.array(input_array).reshape(1, -1)
            
            # Scale the features
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display results
            predicted_class = encoder.classes_[prediction]
            
            st.success(f"Predicted Wine Quality: {predicted_class}")
            
            st.write("Prediction Probabilities:")
            for i, class_name in enumerate(encoder.classes_):
                st.write(f"- {class_name}: {prediction_proba[i]:.2%}")
            
            # Show model info
            st.write(f"Model: {metadata['best_model_name']}")
            st.write(f"Model Accuracy: {metadata['accuracy']:.2%}")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Information about quality categories
    st.write("---")
    st.write("Quality Categories:")
    st.write("- High: Quality score 7-9")
    st.write("- Medium: Quality score 6") 
    st.write("- Low: Quality score 3-5")

if __name__ == "__main__":
    main()
