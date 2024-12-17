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
    property_sqft: float = 1800.5
    latitude: float = 40.71
    longitude: float = -74.0

class LLMProcessorException(Exception):
   
    pass

class LLMProcessor:
    DEFAULT_MODEL = "mixtral-8x7b-32768"
    REQUIRED_FEATURES = ['beds', 'bath', 'property_sqft', 'latitude', 'longitude']

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
        """Create the system prompt for feature extraction"""
        return (
            "You are a real estate assistant. Extract features from the user's input in the form of JSON. "
            "Include the number of bedrooms as 'beds', the number of bathrooms as 'bath', "
            "the property square footage as 'property_sqft', latitude as 'latitude', and longitude as 'longitude'. "
            f"If any value is missing, use these defaults: {HouseFeatures()}. "
            "Important: Return ONLY the JSON object with no additional text or notes."
        )

    def _extract_json_from_response(self, response: str) -> str:
       
        try:
            
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise LLMProcessorException("No valid JSON object found in LLM response")
                
            json_str = response[start_idx:end_idx + 1]
            
            
            json.loads(json_str) 
            return json_str
            
        except json.JSONDecodeError as e:
            raise LLMProcessorException(f"Failed to extract valid JSON: {str(e)}")

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

        except json.JSONDecodeError as e:
            raise LLMProcessorException(f"Invalid JSON from LLM: {str(e)}")
        except Exception as e:
            raise LLMProcessorException(f"Feature extraction failed: {str(e)}")

    def _validate_features(self, features: Dict[str, float]) -> List[float]:
        
        default_features = HouseFeatures()
        return [
            float(features.get("beds", default_features.beds)),
            float(features.get("bath", default_features.bath)),
            float(features.get("property_sqft", default_features.property_sqft)),
            float(features.get("latitude", default_features.latitude)),
            float(features.get("longitude", default_features.longitude))
        ]

    def predict_price(self, features: Dict[str, float]) -> float:
       
        try:
            feature_values = self._validate_features(features)
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

def main():
    
    try:
        processor = LLMProcessor()
        test_input = "I have a house with 4 bedrooms, 3 bathrooms, and 2200 square feet in Los Angeles."
        result = processor.process_with_llm(test_input)
        print(result)
    except Exception as e:
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    main()