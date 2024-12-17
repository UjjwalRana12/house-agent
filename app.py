import os
import joblib
import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()




model = joblib.load("house_price_AI-test.pkl")
app = FastAPI()

class HouseFeatures(BaseModel):
    beds: int
    bath: float
    property_sqft: float
    latitude: float
    longitude: float
    

@app.post("/predict/")
async def predict_price(features: HouseFeatures):
   
    feature_list = [
        features.beds,
        features.bath,
        features.property_sqft,
        features.latitude,
        features.longitude,
        
    ]
    
   
    feature_array = np.array(feature_list).reshape(1, -1)
    
   
    predicted_price = model.predict(feature_array)[0]
    
   
    return {"predicted_price": predicted_price}
