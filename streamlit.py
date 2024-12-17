import streamlit as st
import pandas as pd
from agents import LLMProcessor, LLMProcessorException

class RealEstateApp:
    def __init__(self):
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""
        
        self.llm_processor = LLMProcessor()
        self.setup_page()

    def setup_page(self):
        st.set_page_config(
            page_title="Need help with house prices and availability",
            page_icon="🏠"
        )
        st.title("Chatbot For Price Prediction")
        st.write("Enter a the description of the property.")

    def clear_prediction_history(self):
        st.session_state.prediction_history = []
        st.success("History cleared")

    def run(self):
        
        user_input = st.text_area(
            "Describe your need:",
            placeholder="Example: i need a house with 4 bedrooms, 3 bathrooms, 2200 square feet in Los Angeles",
            value=st.session_state.user_input
        )

        if st.button("Predict Price"):
            if user_input:
                try:
                    st.session_state.user_input = user_input
                    result = self.llm_processor.process_with_llm(user_input)
                    features = self.llm_processor._extract_features(user_input)
                    prediction = self.llm_processor.predict_price(features)
                    
                    
                    st.session_state.prediction_history.append({
                        'description': user_input,
                        'price': prediction,
                        'features': features
                    })

                    
                    st.success(f"Predicted Price: ${prediction:,.2f}")
                    st.write(result)
                    
                    st.subheader("Property Features")
                    for k, v in features.items():
                        if k not in ['latitude', 'longitude']:
                            st.write(f"- {k.replace('_', ' ').title()}: {v}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a property description")

        
        st.sidebar.title("Prediction History")
        if st.sidebar.button("Clear History"):
            self.clear_prediction_history()

        if st.session_state.prediction_history:
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:]), 1):
                with st.sidebar.expander(f"Prediction {i}"):
                    st.write(f"Price: ${pred['price']:,.2f}")
                    st.write("Features:")
                    for k, v in pred['features'].items():
                        if k not in ['latitude', 'longitude']:
                            st.write(f"- {k.replace('_', ' ').title()}: {v}")

def main():
    app = RealEstateApp()
    app.run()

if __name__ == "__main__":
    main()