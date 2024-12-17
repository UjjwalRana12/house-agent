import os
import json
from typing import Dict, Union, List
from joblib import load
from groq import Groq
from dotenv import load_dotenv
import numpy as np
from dataclasses import dataclass


load_dotenv()

@dataclass
class HouseFeatures:
    beds: int = 3
    bath: float = 2.0
    squarefeet: float = 1800.5  
    latitude: float = 34.0522
    longitude: float = -118.2437

class LLMProcessorException(Exception):
   
    pass

class LLMProcessor:
    DEFAULT_MODEL = "mixtral-8x7b-32768"
    REQUIRED_FEATURES = ['beds', 'bath', 'squarefeet', 'latitude', 'longitude'] 

    def __init__(self, model_path: str = "house_price_AI-test.pkl"):
        
        self._initialize_api()
        self._load_model(model_path)
        
    def _initialize_api(self) -> None:
        
        self.api_key = os.getenv("GROQ_KEY")
        if not self.api_key:
            raise LLMProcessorException("GROQ_KEY not found in environment variables.")
        self.client = Groq(api_key=self.api_key)

    def _load_model(self, model_path: str) -> None:
        
        try:
            self.model = load(model_path)
        except Exception as e:
            raise LLMProcessorException(f"Failed to load ML model: {str(e)}")

    def _create_system_prompt(self) -> str:
       
        return (
            "You are a real estate assistant. Extract features from the user's input and return ONLY a simple JSON object "
            "with these exact keys and numeric values: 'beds', 'bath', 'squarefeet', 'latitude', 'longitude'. "  
            "alway take 'latitude', 34.0522 and longitude -118.2437 no matter the location is "
            "Example format: {'beds': 3, 'bath': 2.0, 'squarefeet': 1800.5, 'latitude': 34.0522, 'longitude': -118.2437}. "
            "Do not include any additional text, notes, or escape characters."
        )

    def _extract_json_from_response(self, response: str) -> str:
        
        try:
            
            response = response.replace('\\', '')
            
            
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise LLMProcessorException("No valid JSON object found in LLM response")
                
            json_str = response[start_idx:end_idx + 1]
            
            
            json_str = ' '.join(json_str.split())
            json_str = json_str.replace('\n', ' ').replace('\r', ' ')
            
            
            parsed_json = json.loads(json_str)
            if 'property_sqft' in parsed_json:
                parsed_json['squarefeet'] = parsed_json.pop('property_sqft')
            
            
            cleaned_json = {
                'beds': float(parsed_json.get('beds', 3)),
                'bath': float(parsed_json.get('bath', 2.0)),
                'squarefeet': float(parsed_json.get('squarefeet', 1800.5)),
                'latitude': float(parsed_json.get('latitude', 34.0522)),
                'longitude': float(parsed_json.get('longitude', -118.2437))
            }
            
            return json.dumps(cleaned_json)
            
        except json.JSONDecodeError as e:
            print(f"Original response: {response}")
            raise LLMProcessorException(f"Failed to extract valid JSON: {str(e)}")
        except Exception as e:
            raise LLMProcessorException(f"Error processing JSON: {str(e)}")

    def _extract_features(self, content: str) -> Dict[str, float]:
       
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": content}
                ],
                model=self.DEFAULT_MODEL,
                temperature=0.7,
                max_tokens=256
            )

            llm_response = chat_completion.choices[0].message.content
            print(f"LLM Response: {llm_response}")

            json_str = self._extract_json_from_response(llm_response)
            features = json.loads(json_str)
            
            
            missing_features = [f for f in self.REQUIRED_FEATURES if f not in features]
            if missing_features:
                default_features = HouseFeatures()
                for feature in missing_features:
                    features[feature] = getattr(default_features, feature)
            
            return features

        except Exception as e:
            raise LLMProcessorException(f"Feature extraction failed: {str(e)}")

    def predict_price(self, features: Dict[str, float]) -> float:
       
        try:
            feature_values = [
                float(features.get("beds", HouseFeatures.beds)),
                float(features.get("bath", HouseFeatures.bath)),
                float(features.get("squarefeet", HouseFeatures.squarefeet)),
                float(features.get("latitude", HouseFeatures.latitude)),
                float(features.get("longitude", HouseFeatures.longitude))
            ]
            
            features_array = np.array(feature_values).reshape(1, -1)
            prediction = self.model.predict(features_array)[0]
            
            if prediction <= 0:
                raise LLMProcessorException("Invalid prediction value")
                
            return prediction
        except Exception as e:
            raise LLMProcessorException(f"Price prediction failed: {str(e)}")

    def process_with_llm(self, content: str) -> str:
        
        try:
            features = self._extract_features(content)
            prediction = self.predict_price(features)
            return f"The predicted price for the house is ${prediction:,.2f}"
        except LLMProcessorException as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return "An unexpected error occurred while processing your request."