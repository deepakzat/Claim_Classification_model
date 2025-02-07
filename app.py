import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


model=tf.keras.models.load_model('model.h5')

with open('label_enc_own.pkl','rb') as file:
    label_enc_own=pickle.load(file)

with open('label_enc_gen.pkl','rb') as file:
    label_enc_gen=pickle.load(file)

with open('label_enc_type.pkl','rb') as file:
    label_enc_type=pickle.load(file)

with open('label_enc_year.pkl','rb') as file:
    label_enc_year=pickle.load(file)

with open('scaling.pkl','rb') as file:
    scaler=pickle.load(file)

with open('one_en_age.pkl','rb') as file:
    one_en_age=pickle.load(file)

with open('one_en_dr.pkl','rb') as file:
    one_en_dr=pickle.load(file)




st.title('Car Insurance Claim Prediction')

#User Input.
AGE=st.selectbox('AGE',one_en_age.categories_[0])
GENDER=st.selectbox('GENDER',label_enc_gen.classes_)
DRIVING_EXPERIENCE=st.selectbox('DRIVING_EXPERIENCE',one_en_dr.categories_[0])
CREDIT_SCORE=st.slider('CREDIT_SCORE',0.05,0.99)
VEHICLE_OWNERSHIP=st.selectbox('VEHICLE_OWNERSHIP',label_enc_own.classes_)
VEHICLE_YEAR=st.selectbox('VEHICLE_YEAR',label_enc_year.classes_)
ANNUAL_MILEAGE=st.slider('ANNUAL_MILEAGE',10000,16000,step=1000)
VEHICLE_TYPE=st.selectbox('VEHICLE_TYPE',label_enc_type.classes_)
PAST_ACCIDENTS=st.slider('PAST_ACCIDENTS',0,5)


input_data=pd.DataFrame({'AGE':[AGE],
                        'GENDER':[label_enc_gen.transform([GENDER])[0]],
                         'CREDIT_SCORE':[CREDIT_SCORE],
                        'VEHICLE_OWNERSHIP':[label_enc_own.transform([VEHICLE_OWNERSHIP])[0]],
                        'VEHICLE_YEAR':[label_enc_year.transform([VEHICLE_YEAR])[0]],
                        'ANNUAL_MILEAGE':[ANNUAL_MILEAGE],
                        'VEHICLE_TYPE':[label_enc_type.transform([VEHICLE_TYPE])[0]],
                        'PAST_ACCIDENTS':[PAST_ACCIDENTS]
})


onehot_encoded = one_en_dr.transform([[DRIVING_EXPERIENCE]]).toarray()
feature_names = one_en_dr.get_feature_names_out(['DRIVING_EXPERIENCE'])
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=feature_names)

input_data=pd.concat([input_data,onehot_encoded_df],axis=1)


onehot_encoded1 = one_en_age.transform([[AGE]]).toarray()
feature_names1 = one_en_age.get_feature_names_out(['AGE'])
onehot_encoded_df1 = pd.DataFrame(onehot_encoded1, columns=feature_names1)

input_data=pd.concat([input_data,onehot_encoded_df1],axis=1)
input_data.drop('AGE',axis=1,inplace=True)


input_data_scaled=scaler.transform(input_data)

prediction=model.predict(input_data_scaled)
prediction_probs=prediction[0][0]

st.write(f'Claim Probability: {prediction_probs:.2f}')

if prediction_probs>0.5:
    st.write('Claim will be filed.')
else:
    st.write('Claim will not be filed.')