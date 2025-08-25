import streamlit as st
import pandas as pd
import pickle

#Loading our model

model = tf.keras.models.load_model('/Users/mohdsafeenkhan/Desktop/Machine_Learning/NLP/NLP_Project/Churn_modelling/model/model.h5')

with open('/Users/mohdsafeenkhan/Desktop/Machine_Learning/NLP/NLP_Project/Churn_modelling/model/label_encoder_gender.pkl','rb') as file:
    label_encoder = pickle.load(file)
with open('/Users/mohdsafeenkhan/Desktop/Machine_Learning/NLP/NLP_Project/Churn_modelling/model/ohe_encoder_geo.pkl','rb') as file:
    ohe = pickle.load(file)
with open('/Users/mohdsafeenkhan/Desktop/Machine_Learning/NLP/NLP_Project/Churn_modelling/model/standardScaler.pkl','rb') as file:
    standardScaler = pickle.load(file)

#streamlit app
st.title('Customer Churn Prediction')

#User Input

geography = st.selectbox('Geography',ohe.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# Prepare the input data

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender':[label_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#one hot encode geography 

geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded , columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

#scale the input data

input_data_scaled = standardScaler.transform(input_data)

#predict churn

prediction  = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob >= 0.5:
    st.write(f"The customer is likerly to churn {prediction_prob}")
else:
    st.write(f"The customer is not likely to churn {prediction_prob}")