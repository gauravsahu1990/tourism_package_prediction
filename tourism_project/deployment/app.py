import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------
# Load Model from HuggingFace Hub
# ---------------------------------------------------------------
model_path = hf_hub_download(
    repo_id="gauravsahu1990/Tourism-Package-Prediction",
    filename="best_tourism_packaging_model_v1.joblib"
)

model = joblib.load(model_path)

# ---------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------
st.title("Tourism Product Purchase Prediction App")
st.write("This internal tool predicts whether a customer will purchase the tourism package based on their personal and campaign details.")
st.write("Kindly enter the details below to generate the prediction.")

# ---------------------------------------------------------------
# Input Fields
# ---------------------------------------------------------------

TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Others"])
Gender = st.selectbox("Gender", ["Male", "Female"])
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
Designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])

Age = st.number_input("Age", min_value=18, max_value=90, value=30)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=500, value=20)
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=3)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=50, value=5)
Passport = st.selectbox("Passport Available?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Own a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, value=20000)

# ---------------------------------------------------------------
# Prepare Input for Model
# ---------------------------------------------------------------

input_data = pd.DataFrame([{
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "ProductPitched": ProductPitched,
    "MaritalStatus": MaritalStatus,
    "Designation": Designation,
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome
}])

# ---------------------------------------------------------------
# Prediction Threshold
# ---------------------------------------------------------------
classification_threshold = 0.50

# ---------------------------------------------------------------
# Predict Button
# ---------------------------------------------------------------
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = 1 if prediction_proba >= classification_threshold else 0

    result = "WILL PURCHASE the package" if prediction == 1 else "WILL NOT purchase the package"

    st.subheader("Prediction Result")
    st.write(f"Based on the customer's information, they **{result}**.")
    st.write(f"Prediction Probability: **{prediction_proba:.2f}**")
