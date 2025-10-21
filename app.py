"""
Streamlit Web Application for Telecom Customer Churn Prediction
================================================================
This app provides an interactive interface for:
- Uploading customer data
- Making churn predictions
- Visualizing results
- Exploring model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .churn-yes {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .churn-no {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)


class ChurnPredictor:
    """
    Class to handle churn prediction operations.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the predictor with model directory.
        
        Parameters:
        -----------
        models_dir : str
            Directory containing saved models and preprocessing objects
        """
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = None
        
    def load_model(self, model_name='best_model.pkl'):
        """
        Load the trained model and preprocessing objects.
        
        Parameters:
        -----------
        model_name : str
            Name of the model file to load
        
        Returns:
        --------
        bool : True if successful, False otherwise
        """
        try:
            # Load model
            model_path = os.path.join(self.models_dir, model_name)
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            
            # Load label encoders
            encoders_path = os.path.join(self.models_dir, 'label_encoders.pkl')
            self.label_encoders = joblib.load(encoders_path)
            
            return True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction.
        
        Parameters:
        -----------
        input_data : pd.DataFrame
            Input data to preprocess
        
        Returns:
        --------
        np.array : Preprocessed data
        """
        # Make a copy to avoid modifying original
        data = input_data.copy()
        
        # Apply label encoding to categorical features
        for col, encoder in self.label_encoders.items():
            if col in data.columns and col != 'Churn':
                try:
                    data[col] = encoder.transform(data[col])
                except:
                    pass
        
        # Apply one-hot encoding if needed
        # This should match the preprocessing done during training
        
        # Scale features
        scaled_data = self.scaler.transform(data)
        
        return scaled_data
    
    def predict(self, input_data):
        """
        Make prediction on input data.
        
        Parameters:
        -----------
        input_data : pd.DataFrame
            Input data for prediction
        
        Returns:
        --------
        tuple : (prediction, probability)
        """
        # Preprocess input
        processed_data = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)
        
        return prediction, probability


def main():
    """
    Main function to run the Streamlit app.
    """
    
    # Title and description
    st.title("📱 Telecom Customer Churn Prediction System")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #1f77b4; margin-top: 0;'>Welcome to the Churn Prediction System!</h3>
        <p>This application uses advanced machine learning algorithms to predict customer churn probability.
        Upload your customer data or enter details manually to get instant predictions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = {
        'Best Model': 'best_model.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'Support Vector Machine': 'support_vector_machine.pkl'
    }
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        list(model_options.keys())
    )
    
    # Initialize predictor
    predictor = ChurnPredictor(models_dir='models')
    
    # Load selected model
    model_loaded = predictor.load_model(model_options[selected_model])
    
    if model_loaded:
        st.sidebar.success(f"✓ {selected_model} loaded successfully!")
    else:
        st.sidebar.error("⚠ Model not found. Please train the model first.")
        st.info("""
        **To train the model:**
        1. Place your dataset in the `data/` folder
        2. Run the preprocessing script: `python scripts/data_preprocessing.py`
        3. Run the training script: `python scripts/model_training.py`
        4. Refresh this page
        """)
        return
    
    st.sidebar.markdown("---")
    
    # Prediction mode
    st.sidebar.subheader("Prediction Mode")
    prediction_mode = st.sidebar.radio(
        "Choose input method:",
        ["Single Prediction", "Batch Prediction", "Model Performance"]
    )
    
    # Main content based on selected mode
    if prediction_mode == "Single Prediction":
        show_single_prediction(predictor)
    elif prediction_mode == "Batch Prediction":
        show_batch_prediction(predictor)
    else:
        show_model_performance()


def show_single_prediction(predictor):
    """
    Display single prediction interface.
    
    Parameters:
    -----------
    predictor : ChurnPredictor
        Predictor instance
    """
    st.header("🔍 Single Customer Prediction")
    st.markdown("Enter customer details below to predict churn probability.")
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Account Information")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    
    with col2:
        st.subheader("💰 Billing Information")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=600.0)
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    
    with col3:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    
    # Additional features
    with st.expander("➕ Additional Features (Optional)"):
        col4, col5 = st.columns(2)
        with col4:
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with col5:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    
    # Predict button
    if st.button("🎯 Predict Churn", key="predict_single"):
        # Create input DataFrame
        # Note: Adjust column names based on your actual dataset
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'Contract': [contract],
            'PaymentMethod': [payment_method],
            'PaperlessBilling': [paperless_billing],
            'PhoneService': [phone_service],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'MultipleLines': [multiple_lines]
        })
        
        try:
            # Make prediction
            prediction, probability = predictor.predict(input_data)
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                churn_prob = probability[0][1] * 100
                st.metric("Churn Probability", f"{churn_prob:.2f}%")
            
            with col2:
                retention_prob = probability[0][0] * 100
                st.metric("Retention Probability", f"{retention_prob:.2f}%")
            
            with col3:
                prediction_label = "High Risk" if prediction[0] == 1 else "Low Risk"
                st.metric("Risk Level", prediction_label)
            
            # Visual representation
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob,
                title={'text': "Churn Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred" if churn_prob > 50 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            if prediction[0] == 1:
                st.error("""
                **⚠️ High Churn Risk Detected!**
                
                **Recommended Actions:**
                - Reach out to customer with retention offers
                - Review service quality and address concerns
                - Consider loyalty rewards or discounts
                - Schedule follow-up call
                """)
            else:
                st.success("""
                **✅ Low Churn Risk**
                
                **Customer Status:** Satisfied and likely to stay
                
                **Suggested Actions:**
                - Continue providing excellent service
                - Consider upselling opportunities
                - Request feedback and testimonials
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all required fields are filled correctly.")


def show_batch_prediction(predictor):
    """
    Display batch prediction interface.
    
    Parameters:
    -----------
    predictor : ChurnPredictor
        Predictor instance
    """
    st.header("📁 Batch Prediction")
    st.markdown("Upload a CSV or Excel file containing customer data for batch predictions.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with customer data. Make sure column names match the training data."
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✓ File uploaded successfully! {df.shape[0]} records found.")
            
            # Show data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
            
            # Predict button
            if st.button("🎯 Generate Predictions", key="predict_batch"):
                with st.spinner("Making predictions..."):
                    try:
                        # Make predictions
                        predictions, probabilities = predictor.predict(df)
                        
                        # Add predictions to dataframe
                        df['Churn_Prediction'] = predictions
                        df['Churn_Probability'] = probabilities[:, 1]
                        df['Risk_Level'] = df['Churn_Probability'].apply(
                            lambda x: 'High' if x > 0.7 else 'Medium' if x > 0.3 else 'Low'
                        )
                        
                        # Display results
                        st.success("✓ Predictions completed!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_customers = len(df)
                            st.metric("Total Customers", total_customers)
                        
                        with col2:
                            churn_count = (predictions == 1).sum()
                            st.metric("Predicted Churners", churn_count)
                        
                        with col3:
                            churn_rate = (churn_count / total_customers) * 100
                            st.metric("Churn Rate", f"{churn_rate:.2f}%")
                        
                        with col4:
                            high_risk = (df['Risk_Level'] == 'High').sum()
                            st.metric("High Risk Customers", high_risk)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Churn distribution
                            fig1 = px.pie(
                                values=df['Churn_Prediction'].value_counts().values,
                                names=['No Churn', 'Churn'],
                                title='Churn Distribution',
                                color_discrete_sequence=['#2ecc71', '#e74c3c']
                            )
                            st.plotly_chart(fig1, use_container_width=True)
                        
                        with col2:
                            # Risk level distribution
                            risk_counts = df['Risk_Level'].value_counts()
                            fig2 = px.bar(
                                x=risk_counts.index,
                                y=risk_counts.values,
                                title='Risk Level Distribution',
                                labels={'x': 'Risk Level', 'y': 'Count'},
                                color=risk_counts.index,
                                color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        
                        # Show results table
                        st.subheader("📊 Detailed Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("Please ensure your data format matches the training data.")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def show_model_performance():
    """
    Display model performance metrics and visualizations.
    """
    st.header("📈 Model Performance")
    st.markdown("View detailed performance metrics and comparisons of different models.")
    
    # Check if performance plots exist
    plots_dir = 'plots'
    
    if os.path.exists(plots_dir):
        # Display confusion matrices
        cm_path = os.path.join(plots_dir, 'confusion_matrices.png')
        if os.path.exists(cm_path):
            st.subheader("🎯 Confusion Matrices")
            st.image(cm_path, use_column_width=True)
        
        # Display ROC curves
        roc_path = os.path.join(plots_dir, 'roc_curves.png')
        if os.path.exists(roc_path):
            st.subheader("📊 ROC Curves")
            st.image(roc_path, use_column_width=True)
        
        # Display metrics comparison
        metrics_path = os.path.join(plots_dir, 'metrics_comparison.png')
        if os.path.exists(metrics_path):
            st.subheader("📉 Metrics Comparison")
            st.image(metrics_path, use_column_width=True)
    else:
        st.info("""
        **Performance plots not found.**
        
        To generate performance visualizations:
        1. Run the model training script: `python scripts/model_training.py`
        2. Plots will be saved in the `plots/` directory
        3. Refresh this page
        """)
    
    # Model information
    st.subheader("ℹ️ Model Information")
    st.markdown("""
    **Models Used:**
    - **Random Forest**: Ensemble learning method using multiple decision trees
    - **XGBoost**: Gradient boosting algorithm optimized for speed and performance
    - **Logistic Regression**: Linear model for binary classification
    - **Support Vector Machine**: Finds optimal hyperplane for classification
    
    **Evaluation Metrics:**
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive predictions that are correct
    - **Recall**: Proportion of actual positives correctly identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **ROC-AUC**: Area under the receiver operating characteristic curve
    """)


# Run the app
if __name__ == "__main__":
    main()
