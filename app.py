import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function for prediction
def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # Convert inputs into a NumPy array and reshape
    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    input_data = scaler.transform(input_data)  # Scale input
    prediction = model.predict(input_data)  # Predict class (0 = Not Survived, 1 = Survived)
    prediction_prob = model.predict_proba(input_data)[:, 1]  # Get probability of survival
    return prediction[0], prediction_prob[0]

# Streamlit UI
st.title("Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Price", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical inputs
sex = 1 if sex == "Male" else 0
embarked_mapping = {"C": 0, "Q": 1, "S": 2}  # Same encoding as training data
embarked = embarked_mapping[embarked]

# Prediction Button
if st.button("Predict Survival"):
    prediction, probability = predict_survival(pclass, sex, age, sibsp, parch, fare, embarked)
    if prediction == 1:
        st.success(f"The passenger is predicted to **Survive** with a probability of {probability:.2f}.")
    else:
        st.error(f"The passenger is predicted **Not to Survive** with a probability of {probability:.2f}.")
