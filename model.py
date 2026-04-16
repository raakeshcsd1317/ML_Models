import numpy as np
import pandas as pd
import streamlit as st
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import google.generativeai as genai
key=os.getenv('GOOGLE_API_KEY')    
genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-2.5-flash")
#TO get methods to generate code and suggest improvements
from analysis import generate_code, suggest_improvements

st.set_page_config(page_title='🤖 ML Studio', page_icon='⚠️', layout='wide')

st.title('🤖 Machine Learning Studio')
st.markdown('Upload your dataset and let AI train models automatically 🧠✨')

uploaded_file = st.file_uploader('📂 Upload a CSV file', type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.markdown('### 👀 Data Preview')
    st.dataframe(df.head())

    target = st.selectbox('🎯 Select Target Column', df.columns)

    if target:
        st.markdown('---')
        st.markdown('## ⚙️ Data Processing')

        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(include='object').columns.tolist()

        # Missing values
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        X[cat_cols] = X[cat_cols].fillna('Missing')

        # 🚨 FIX: limit high-cardinality columns (prevents crash)
        for col in cat_cols:
            if X[col].nunique() > 50:
                X[col] = X[col].astype('category').cat.codes

        # Encoding
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype='int8')

        # Target handling
        if df[target].dtype == 'object':
            label = LabelEncoder()
            y = label.fit_transform(y.astype(str))
            problem_type = '📊 Classification'
        else:
            problem_type = '📊 Classification' if len(np.unique(y)) < 15 else '📈 Regression'

        st.success(f'🧠 Problem Type Detected: **{problem_type}**')

        # Split
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)

        st.markdown('---')
        st.markdown('## 🚀 Training Models')

        results = []

        if 'Regression' in problem_type:
            models = {
                '📈 Linear Regression': LinearRegression(),
                '🌲 Random Forest': RandomForestRegressor(random_state=42),
                '⚡ Gradient Boosting': GradientBoostingRegressor(random_state=42)
            }
        else:
            models = {
                '📊 Logistic Regression': LogisticRegression(max_iter=1000),
                '🌲 Random Forest': RandomForestClassifier(random_state=42),
                '⚡ Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

        progress = st.progress(0)

        for i, (name, model_obj) in enumerate(models.items()):
            model_obj.fit(xtrain_scaled, ytrain)
            ypred = model_obj.predict(xtest_scaled)

            if 'Regression' in problem_type:
                results.append({
                    'Model': name,
                    'R2': round(r2_score(ytest, ypred), 3),
                    'RMSE': round(np.sqrt(mean_squared_error(ytest, ypred)), 3)
                })
            else:
                results.append({
                    'Model': name,
                    'accuracy_score': round(accuracy_score(ytest, ypred), 3),
                    'precision_score': round(precision_score(ytest, ypred, average='weighted'), 3),
                    'recall_score': round(recall_score(ytest, ypred, average='weighted'), 3),
                    'f1_score': round(f1_score(ytest, ypred, average='weighted'), 3)
                })

            progress.progress((i + 1) / len(models))

        st.markdown('---')
        st.success('✅ Training Complete! 🎉')

        results_df = pd.DataFrame(results)

        st.markdown('## 📊 Model Performance')
        st.dataframe(results_df, use_container_width=True)
        if problem_type == '📈 Regression':
            st.bar_chart(results_df.set_index('Model')[['R2', 'RMSE']])
            st.bar_chart(results_df.set_index('Model')['RMSE'])
        else:
            st.bar_chart(results_df.set_index('Model')['accuracy_score'])
            st.bar_chart(results_df.set_index('Model')['f1_score'])  
            
        # Generate code and suggestions using Gemini
        if st.button('💡 Get AI Insights')  :
            with st.spinner('Generating insights...'):
                code = generate_code(results_df)
             

            st.markdown('### 🧠 AI Analysis')
            st.markdown(f'**Best Model Analysis:**\n{code}')
            
        
        else:
            if st.button('🔧 Suggest Improvements'):
                with st.spinner('Generating suggestions...'):
                    suggestions = suggest_improvements(results_df)

                st.markdown('### 🔧 AI Suggestions')
                st.markdown(f'**Model Improvement Suggestions:**\n{suggestions}')    