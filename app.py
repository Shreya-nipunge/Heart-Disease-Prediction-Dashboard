import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_lottie import st_lottie

# ------------------------------------------------------
# Page Config
# ------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="Ã¢ÂÂ¤Ã¯Â¸Â",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------
# Styling
# ------------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
h1, h2, h3, h4 {
    color: #ff4b4b;
}
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff7b7b);
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 18px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff7b7b, #ff4b4b);
}
.sidebar .sidebar-content {
    background-color: #161a1f;
}
hr {
    border: 1px solid #333;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------
st.sidebar.title("HeartCare Dashboard")
menu = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "About"])

# ------------------------------------------------------
# Load Model and Data
# ------------------------------------------------------
model = joblib.load("heart_model.pkl")
df = pd.read_csv("heart.csv")

# ------------------------------------------------------
# Dashboard Page
# ------------------------------------------------------
if menu == "Dashboard":
    st.title("Heart Disease Prediction Dashboard")
    st.write("An AI-powered system that predicts heart disease and provides key data insights for better understanding of patient health trends.")

    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Patients", df.shape[0])
    with col2:
        st.metric("Average Age", int(df['age'].mean()))
    with col3:
        st.metric("Avg Cholesterol", int(df['chol'].mean()))
    with col4:
        st.metric("Heart Disease Cases (%)", round(df['target'].mean()*100, 2))

    st.markdown("---")

    # Section 1: Distributions
    st.subheader("Dataset Overview and Distributions")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Heart Disease Distribution**")
        sns.countplot(x='target', data=df, palette='Reds')
        st.pyplot(plt.gcf())
    with colB:
        st.markdown("**Gender Distribution**")
        fig, ax = plt.subplots()
        df['sex'].replace({0: 'Female', 1: 'Male'}).value_counts().plot.pie(
            autopct='%1.1f%%', colors=['#ff4b4b','#ffb3b3'], ax=ax)
        ax.set_ylabel('')
        st.pyplot(fig)

    st.markdown("---")

    # Section 2: Relationships
    st.subheader("Key Health Feature Relationships")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cholesterol vs Target (Box Plot)**")
        sns.boxplot(x='target', y='chol', data=df, palette='coolwarm')
        st.pyplot(plt.gcf())
    with col2:
        st.markdown("**Max Heart Rate vs Age (Trend)**")
        sns.regplot(x='age', y='thalach', data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
        st.pyplot(plt.gcf())

    st.markdown("---")

    # Section 3: Comparison
    st.subheader("Comparison Between Healthy and Diseased Patients")

    mean_df = df.groupby('target').mean()[['age', 'chol', 'trestbps', 'thalach']]
    st.bar_chart(mean_df)

    st.caption("This chart compares average values of key health indicators for healthy vs diseased individuals.")

    st.markdown("---")

    # Section 4: Correlation and Pairplot
    st.subheader("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), cmap="Reds", annot=False, ax=ax)
    st.pyplot(fig)

    st.markdown("### Pairplot (Selected Features)")
    sns.pairplot(df[['age', 'chol', 'thalach', 'target']], hue='target', palette='coolwarm')
    st.pyplot(plt.gcf())

    st.caption("These visualizations help identify which factors most strongly correlate with heart disease risk.")

# ------------------------------------------------------
# Prediction Page
# ------------------------------------------------------
elif menu == "Prediction":
    st.title("Heart Disease Risk Prediction")

    st.markdown("Provide your health parameters below to predict the risk of heart disease:")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 5, 120, 40)
        sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.slider("Resting BP (mmHg)", 60, 260, 120)
    with col2:
        chol = st.slider("Cholesterol (mg/dL)", 50, 700, 200)
        fbs = st.radio("Fasting Blood Sugar >120 mg/dL?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
        thalach = st.slider("Max Heart Rate", 40, 220, 150)
    with col3:
        exang = st.radio("Exercise-Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope (0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal (0=Normal,1=Fixed,2=Reversible,3=Other)", [0, 1, 2, 3])

    # Prepare data
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                                       'thalach','exang','oldpeak','slope','ca','thal'])

    if st.button("Predict"):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")
        if pred == 1:
            st.error(f"Prediction: This person **HAS heart disease**.\n**Probability:** {prob:.2f}%")
        else:
            st.success(f"Prediction: This person is **Healthy**.\n**Probability of disease:** {prob:.2f}%")

        # Results Summary
        st.markdown("## Results Summary & Health Insights")
        avg = df.mean()

        comparison = pd.DataFrame({
            "Feature": ["Age", "Cholesterol", "Resting BP", "Max Heart Rate"],
            "Patient": [age, chol, trestbps, thalach],
            "Average": [avg['age'], avg['chol'], avg['trestbps'], avg['thalach']]
        })

        st.dataframe(comparison, use_container_width=True)

        # Comparison Chart
        fig, ax = plt.subplots()
        sns.barplot(x="Feature", y="value", hue="variable",
                    data=pd.melt(comparison, ["Feature"]), palette="Reds")
        st.pyplot(fig)

        # Health Tips (realistic insights)
        st.markdown("### Health Recommendations")
        if pred == 1:
            st.warning("""
            - Maintain cholesterol below 200 mg/dL with a balanced diet.  
            - Engage in 30 mins of exercise daily.  
            - Monitor blood pressure regularly.  
            - Avoid smoking and manage stress.  
            - Consult a cardiologist for regular checkups.
            """)
        else:
            st.info("""
            - Keep maintaining a healthy lifestyle!  
            - Regular exercise and a balanced diet are key.  
            - Get annual heart checkups to stay on track.  
            """)

        st.caption("These insights are data-driven and based on average health statistics from the dataset.")

# ------------------------------------------------------
# About Page
# ------------------------------------------------------
elif menu == "About":
    st.title("About This Project")
    st.markdown("""
    **Project:** Heart Disease Prediction System  
    **Developer:** Shreya Nipunge  
    **Tools:** Python, Scikit-Learn, Streamlit, Seaborn, Matplotlib  

    **Overview:**  
    This project uses a trained ML model to predict the probability of heart disease based on user health inputs.  
    It also provides visual insights and recommendations for better health awareness.

    **Key Features:**  
    - Real-time prediction  
    - Dataset analytics dashboard  
    - Personalized result summary and health recommendations  
    - Modern dark-mode web interface  

     *AI for Good - Empowering health awareness through data science.*
    """)
