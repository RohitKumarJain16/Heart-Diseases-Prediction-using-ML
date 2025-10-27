import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (your existing CSS remains the same)
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #DC143C;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #B22222;
        margin-bottom: 1rem;
        font-weight: bold;
        border-bottom: 2px solid #B22222;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    
    .metric-card h3 {
        color: #FFE4E1;
        font-size: 1.1rem;
        margin: 0;
        font-weight: 500;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #32CD32 0%, #90EE90 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .prediction-result-danger {
        background: linear-gradient(135deg, #FF4500 0%, #FF6347 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin: 1rem 0;
    }
    
    .prediction-result-best {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 3px solid #FF8C00;
    }
    
    .prediction-result h1 {
        color: white;
        font-size: 3rem;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #E6F3FF 0%, #CCE7FF 100%);
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFE6E6 0%, #FFCCCC 100%);
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load models function
@st.cache_resource
def load_models():
    """Load all saved models and scaler"""
    models = {}
    try:
        # Check if all files exist first
        required_files = ['logistic_model.pkl', 'svm_model.pkl', 'knn_model.pkl', 'scaler.pkl']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            st.error(f"❌ Missing files: {', '.join(missing_files)}")
            st.info("📁 Please ensure these files are in the same directory as app.py:")
            for file in missing_files:
                st.write(f"   • {file}")
            return None, None
        
        # Load models
        with open('logistic_model.pkl', 'rb') as file:
            models['Logistic Regression'] = pickle.load(file)
        
        with open('svm_model.pkl', 'rb') as file:
            models['SVM (Best Model)'] = pickle.load(file)  # Mark as best
        
        with open('knn_model.pkl', 'rb') as file:
            models['KNN'] = pickle.load(file)
        
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        
        st.success("✅ All models loaded successfully!")
        return models, scaler
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

# Load data function
@st.cache_data
def load_data():
    """Load the heart disease dataset"""
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        st.error("❌ heart.csv file not found. Please ensure the file is in the same directory as app.py")
        return None

# Enhanced prediction function
def predict_heart_disease(models, scaler, patient_data):
    """Make predictions using all models with proper confidence handling"""
    # Scale the input data
    patient_scaled = scaler.transform(patient_data)
    
    predictions = {}
    for name, model in models.items():
        pred = model.predict(patient_scaled)[0]
        
        # Handle confidence differently for different models
        if hasattr(model, 'predict_proba'):
            # For Logistic Regression and KNN
            proba = model.predict_proba(patient_scaled)[0]
            confidence = max(proba) * 100
        elif hasattr(model, 'decision_function'):
            # For SVM - use decision function score
            decision_score = model.decision_function(patient_scaled)[0]
            # Convert decision function to confidence (0-100%)
            # The further from 0, the more confident
            confidence = min(95, max(55, abs(decision_score) * 20 + 50))
        else:
            # Fallback
            confidence = 75.0  # Default confidence
        
        predictions[name] = {
            'prediction': pred,
            'prediction_text': 'Heart Disease Risk' if pred == 1 else 'No Heart Disease',
            'confidence': confidence,
            'is_best': 'Best Model' in name  # Mark best model
        }
    
    return predictions

def main():
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    
    # DEBUG SECTION (optional - remove in production)
    st.sidebar.markdown("## 🔍 File Check")
    if st.sidebar.checkbox("Show Debug Info"):
        current_dir = os.getcwd()
        st.sidebar.write(f"Current directory: {current_dir}")
        
        files = [f for f in os.listdir('.') if f.endswith(('.pkl', '.csv'))]
        st.sidebar.write("Available files:")
        for f in files:
            st.sidebar.write(f"✅ {f}")
    
    # Sidebar
    st.sidebar.markdown("## 🏥 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Dataset Analysis", "🔮 Make Prediction", "📈 Model Performance", "ℹ️ About"]
    )
    
    # Load models and data
    models, scaler = load_models()
    df = load_data()
    
    if models is None or scaler is None:
        st.warning("⚠️ Models not loaded. Please check the files.")
        st.stop()
    
    if df is None:
        st.warning("⚠️ Dataset not loaded. Please check heart.csv file.")
        st.stop()
    
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.image("https://cdn.pixabay.com/photo/2017/02/15/20/58/ekg-2069872_1280.png", 
                    caption="Heart Health Monitoring", width=400)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This application uses machine learning to predict heart disease risk based on patient medical data. 
        The system employs three different algorithms with <strong>SVM being the best performing model</strong>:</p>
        <ul>
        <li><strong>🏆 Support Vector Machine (SVM)</strong> - Best performing model (87% accuracy)</li>
        <li><strong>Logistic Regression</strong> - Linear probabilistic model (85% accuracy)</li>
        <li><strong>K-Nearest Neighbors (KNN)</strong> - Similarity-based prediction (82% accuracy)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset overview
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
            <h3>👥 Total Patients</h3>
            <h2>{len(df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
            <h3>📋 Features</h3>
            <h2>{len(df.columns)-1}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            heart_disease_count = df['HeartDisease'].sum()
            st.markdown(f"""
            <div class="metric-card">
            <h3>❤️ Heart Disease</h3>
            <h2>{heart_disease_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            healthy_count = len(df) - heart_disease_count
            st.markdown(f"""
            <div class="metric-card">
            <h3>💚 Healthy</h3>
            <h2>{healthy_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<h2 class="sub-header">📋 Dataset Quick View</h2>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.bar(
                x=['No Heart Disease', 'Heart Disease'],
                y=[healthy_count, heart_disease_count],
                title="Heart Disease Distribution",
                color=['No Heart Disease', 'Heart Disease'],
                color_discrete_map={'No Heart Disease': '#32CD32', 'Heart Disease': '#FF4500'}
            )
            fig_dist.update_layout(showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_pie = px.pie(
                values=[healthy_count, heart_disease_count],
                names=['No Heart Disease', 'Heart Disease'],
                title="Heart Disease Percentage",
                color_discrete_map={'No Heart Disease': '#32CD32', 'Heart Disease': '#FF4500'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    elif page == "📊 Dataset Analysis":
        st.markdown('<h2 class="sub-header">📊 Dataset Analysis</h2>', unsafe_allow_html=True)
        
        # Statistical summary
        st.markdown("### 📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Feature analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 👥 Age Distribution")
            fig_age = px.histogram(df, x='Age', color='HeartDisease', 
                                 title="Age Distribution by Heart Disease Status",
                                 color_discrete_map={0: '#32CD32', 1: '#FF4500'})
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.markdown("### ⚖️ Gender Analysis")
            gender_counts = df.groupby(['Sex', 'HeartDisease']).size().reset_index(name='Count')
            fig_gender = px.bar(gender_counts, x='Sex', y='Count', color='HeartDisease',
                              title="Heart Disease by Gender",
                              color_discrete_map={0: '#32CD32', 1: '#FF4500'})
            st.plotly_chart(fig_gender, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🌡️ Feature Correlations")
        numeric_df = df.select_dtypes(include=[np.number])
        fig_corr = px.imshow(
            numeric_df.corr(),
            text_auto=True,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Additional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💓 Max Heart Rate Analysis")
            fig_hr = px.box(df, x='HeartDisease', y='MaxHR',
                           title="Max Heart Rate by Heart Disease Status",
                           color='HeartDisease',
                           color_discrete_map={0: '#32CD32', 1: '#FF4500'})
            st.plotly_chart(fig_hr, use_container_width=True)
        
        with col2:
            st.markdown("### 🩺 Chest Pain Type Analysis")
            chest_pain_counts = df.groupby(['ChestPainType', 'HeartDisease']).size().reset_index(name='Count')
            fig_chest = px.bar(chest_pain_counts, x='ChestPainType', y='Count', 
                             color='HeartDisease',
                             title="Chest Pain Type vs Heart Disease",
                             color_discrete_map={0: '#32CD32', 1: '#FF4500'})
            st.plotly_chart(fig_chest, use_container_width=True)
    
    elif page == "🔮 Make Prediction":
        st.markdown('<h2 class="sub-header">🔮 Heart Disease Prediction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📝 Enter Patient Information")
            
            # Create input form
            col_a, col_b = st.columns(2)
            
            with col_a:
                age = st.slider("Age", 20, 100, 50)
                sex = st.selectbox("Sex", ["M", "F"])
                chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
                resting_bp = st.slider("Resting Blood Pressure", 80, 200, 120)
                cholesterol = st.slider("Cholesterol", 100, 400, 200)
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120", ["0", "1"])
            
            with col_b:
                resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
                max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
                exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])
                oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
                st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
            
            if st.button("🔍 Predict Heart Disease Risk", type="primary"):
                # Prepare input data
                input_data = pd.DataFrame({
                    'Age': [age],
                    'Sex': [sex],
                    'ChestPainType': [chest_pain],
                    'RestingBP': [resting_bp],
                    'Cholesterol': [cholesterol],
                    'FastingBS': [int(fasting_bs)],
                    'RestingECG': [resting_ecg],
                    'MaxHR': [max_hr],
                    'ExerciseAngina': [exercise_angina],
                    'Oldpeak': [oldpeak],
                    'ST_Slope': [st_slope]
                })
                
                # Apply same preprocessing as training data
                input_encoded = pd.get_dummies(input_data, drop_first=True)
                
                # Ensure all columns are present (same as training data)
                expected_columns = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                                  'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
                                  'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
                                  'ST_Slope_Flat', 'ST_Slope_Up']
                
                for col in expected_columns:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0
                
                input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)
                
                # Make predictions
                predictions = predict_heart_disease(models, scaler, input_encoded)
                
                st.session_state.predictions = predictions
                st.session_state.patient_data = {
                    'Age': age, 'Sex': sex, 'ChestPainType': chest_pain,
                    'RestingBP': resting_bp, 'Cholesterol': cholesterol
                }
        
        with col2:
            if hasattr(st.session_state, 'predictions'):
                st.markdown("### 🎯 Prediction Results")
                
                for model_name, result in st.session_state.predictions.items():
                    # Use different styling for the best model
                    if result['is_best']:
                        card_class = "prediction-result-best"
                        icon = "🏆"
                    elif result['prediction'] == 1:
                        card_class = "prediction-result-danger"
                        icon = "⚠️"
                    else:
                        card_class = "prediction-result"
                        icon = "✅"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                    <h3>{model_name}</h3>
                    <h2>{icon} {result['prediction_text']}</h2>
                    <p>Confidence: {result['confidence']:.1f}%</p>
                    {f"<small>⭐ Best Performing Model</small>" if result['is_best'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Consensus prediction
                predictions_list = [pred['prediction'] for pred in st.session_state.predictions.values()]
                consensus = 1 if sum(predictions_list) >= 2 else 0
                
                st.markdown("### 🏥 Medical Consensus")
                if consensus == 1:
                    st.markdown("""
                    <div class="warning-box">
                    <h3>⚠️ Recommendation</h3>
                    <p>Multiple models indicate potential heart disease risk. The SVM (best model) should be given more weight in the decision. Please consult with a healthcare professional for proper medical evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="info-box">
                    <h3>✅ Good News</h3>
                    <p>Models suggest low heart disease risk. The SVM (best model) shows no significant risk. Continue maintaining a healthy lifestyle and regular check-ups.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    elif page == "📈 Model Performance":
        st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
        
        # Updated model performance with SVM as best
        model_performance = {
            'SVM (Best)': {'Accuracy': 0.870, 'F1 Score': 0.869, 'Status': 'Best'},
            'Logistic Regression': {'Accuracy': 0.852, 'F1 Score': 0.846, 'Status': 'Good'},
            'KNN': {'Accuracy': 0.826, 'F1 Score': 0.812, 'Status': 'Fair'}
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, metrics) in enumerate(model_performance.items()):
            with [col1, col2, col3][i]:
                status_color = "#FFD700" if metrics['Status'] == 'Best' else "#FF6B6B"
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {status_color} 0%, #FF8E53 100%);">
                <h3>{model_name}</h3>
                <h2>{metrics['Accuracy']:.3f}</h2>
                <p>Accuracy</p>
                {'<small>🏆 Best Model</small>' if metrics['Status'] == 'Best' else ''}
                </div>
                """, unsafe_allow_html=True)
        
        # Performance comparison chart
        st.markdown("### 📊 Model Comparison")
        
        models_df = pd.DataFrame(model_performance).T.reset_index()
        models_df.columns = ['Model', 'Accuracy', 'F1 Score', 'Status']
        
        fig_performance = px.bar(models_df, x='Model', y=['Accuracy', 'F1 Score'],
                                title="Model Performance Comparison",
                                barmode='group',
                                color_discrete_sequence=['#FFD700', '#87CEEB'])
        st.plotly_chart(fig_performance, use_container_width=True)
        
        st.markdown("""
        ### 🏗️ Model Information
        
        **🏆 Support Vector Machine (SVM) - BEST MODEL:**
        - **Accuracy**: 87.0%
        - **F1 Score**: 86.9%
        - Effective for complex patterns
        - Good performance on small datasets
        - Robust to outliers
        - **Confidence Calculation**: Uses decision function score
        
        **Logistic Regression:**
        - **Accuracy**: 85.2%
        - **F1 Score**: 84.6%
        - Linear model good for interpretability
        - Provides probability estimates
        - Fast training and prediction
        
        **K-Nearest Neighbors (KNN):**
        - **Accuracy**: 82.6%
        - **F1 Score**: 81.2%
        - Simple, intuitive algorithm
        - Non-parametric approach
        - Good for local patterns
        
        ### 🎯 Why SVM is the Best:
        - **Highest accuracy** among all models
        - **Best F1 score** for balanced performance
        - **Robust to overfitting** with proper regularization
        - **Good generalization** on unseen data
        """)
    
    elif page == "ℹ️ About":
        st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🏥 Medical Disclaimer</h3>
        <p><strong>This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</strong> Always consult with qualified healthcare providers for medical decisions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 🔬 Technical Details
        
        **Dataset Information:**
        - **Source**: Heart Disease Dataset
        - **Features**: 11 medical measurements and indicators
        - **Target**: Binary classification (Heart Disease/No Heart Disease)
        - **Size**: 918 patient records
        
        **Machine Learning Models:**
        - **🏆 Support Vector Machine (SVM)**: Best performing model (87% accuracy)
        - **Logistic Regression**: Linear probabilistic classifier (85.2% accuracy)
        - **K-Nearest Neighbors**: Distance-based classifier (82.6% accuracy)
        
        **Data Preprocessing:**
        - Feature encoding for categorical variables
        - StandardScaler for numerical feature normalization
        - Train-test split (80/20) for model validation
        
        ### 📊 Features Used:
        - **Age**: Patient age in years
        - **Sex**: Gender (M/F)
        - **ChestPainType**: Type of chest pain experienced
        - **RestingBP**: Resting blood pressure
        - **Cholesterol**: Serum cholesterol level
        - **FastingBS**: Fasting blood sugar > 120 mg/dl
        - **RestingECG**: Resting electrocardiogram results
        - **MaxHR**: Maximum heart rate achieved
        - **ExerciseAngina**: Exercise-induced angina
        - **Oldpeak**: ST depression induced by exercise
        - **ST_Slope**: Slope of peak exercise ST segment
        
        ### 🎯 Key Features:
        - **Multi-Model Prediction**: Uses 3 different algorithms
        - **SVM Confidence**: Special handling for SVM confidence calculation
        - **Best Model Highlighting**: SVM marked as best performing model
        - **Interactive Interface**: User-friendly web application
        - **Real-time Predictions**: Instant risk assessment
        - **Data Visualization**: Comprehensive dataset analysis
        - **Model Persistence**: Saved models for quick deployment
        
        ### 🔧 SVM Confidence Calculation:
        Since SVM doesn't have built-in `predict_proba()`, we use the `decision_function()` 
        which returns the distance to the separating hyperplane. The further from 0, 
        the more confident the prediction.
        
        ---
        
        **⚠️ Important**: This tool is designed for educational and research purposes. 
        Medical decisions should always be made in consultation with healthcare professionals.
        """)

if __name__ == "__main__":
    main()