import numpy as np
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import uvicorn

from ML import internet_map, tech_map

telecom_app = FastAPI(title="TelecomCustomers Сhurn Prediction API", version="1.0")

model = joblib.load('model_telco.pkl')
scaler = joblib.load('scaler_telco.pkl')


class TelCusFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str

@telecom_app.post("/predict")
async def predict(data: TelCusFeatures):
    sample = data.model_dump()

    contract_map = {"Month-to-month":0, "One year":1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic":1, "No":2}
    online_map = {"Yes":0, "No":1, "No internet service":2}
    tech_map = {"Yes": 0, "No": 1, "No internet service":2}

    contract = contract_map.get(sample["Contract"], 0)
    internet = internet_map.get(sample["InternetService"], 0)
    online = online_map.get(sample["OnlineSecurity"], 0)
    tech = tech_map.get(sample["TechSupport"], 0)

    numeric = np.array([
        sample["tenure"],
        sample["MonthlyCharges"],
        sample["TotalCharges"]
    ]).reshape(1, -1)

    scaled = scaler.transform(numeric)

    final_features = np.concatenate([scaled[0], [contract, internet, online, tech]]).reshape(1, -1)

    prob = model.predict_proba(final_features)[0][1]
    result = "Уйдет" if prob > 0.5 else "Останется"

    return {"prediction": result, "probability": f"{round(prob*100, 1)}%"}


if __name__== '__main__':
    uvicorn.run(telecom_app, host= '127.0.0.5', port=8000)