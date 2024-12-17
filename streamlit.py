import streamlit as st
import pandas as pd
from typing import Tuple
from agents import LLMProcessor, LLMProcessorException
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

class RealEstateApp:
    def __init__(self):
        
        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""
        if "last_prediction" not in st.session_state:
            st.session_state.last_prediction = None
        
        
        self.llm_processor = LLMProcessor()
        
        
        self.setup_page()

    def setup_page(self):
        st.set_page_config(
            page_title="Real Estate Price Predictor",
            page_icon="🏠",
            layout="wide"
        )
        st.title("🏠 Real Estate Price Predictor")
        st.markdown(
            """
            Enter a description of a property, and I'll predict its price based on the features.
            Include details like number of bedrooms, bathrooms, square footage, and location.
            """
        )

    def create_map(self, lat: float, lon: float) -> None:
        """Create an interactive map for the property location"""
        m = folium.Map(location=[lat, lon], zoom_start=13)
        folium.Marker(
            [lat, lon],
            popup="Property Location",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
        st_folium(m, width=700, height=400)

    def create_features_chart(self, features: dict) -> None:
        """Create a radar chart of property features"""
        categories = ['Bedrooms', 'Bathrooms', 'Square Footage (hundreds)']
        values = [
            features.get('beds', 0),
            features.get('bath', 0),
            features.get('squarefeet', 0) / 100
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Property Features'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2]
                )),
            showlegend=False,
            title="Property Feature Breakdown"
        )
        st.plotly_chart(fig)

    def display_price_history(self) -> None:
    
     if st.session_state.prediction_history:
        df = pd.DataFrame(st.session_state.prediction_history)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')  
        
        
        x_values = list(range(1, len(df) + 1))
        
        fig = go.Figure(data=go.Scatter(
            x=x_values, 
            y=df['price'],
            mode='lines+markers',
            name='Price History',
            marker=dict(color='blue', size=10)
        ))
        fig.update_layout(
            title='Price Prediction History',
            xaxis_title='Prediction Number',
            yaxis_title='Predicted Price ($)',
            height=400
        )
        st.plotly_chart(fig)
     else:
        st.info("No prediction history available yet.")

    def clear_prediction_history(self):
        """Clear the prediction history"""
        st.session_state.prediction_history = []
        st.success("Prediction history has been cleared!")

    def run(self):
        
        col1, col2 = st.columns([2, 1])

        with col1:
            
            user_input = st.text_area(
                "Describe the property:",
                height=100,
                placeholder="Example: I have a house with 4 bedrooms, 3 bathrooms, and 2200 square feet in Los Angeles.",
                value=st.session_state.user_input,
                key="property_description"
            )

            
            if st.button("Predict Price", type="primary"):
                if user_input:
                    try:
                        
                        st.session_state.user_input = user_input
                        
                        
                        result = self.llm_processor.process_with_llm(user_input)
                        features = self.llm_processor._extract_features(user_input)
                        prediction = self.llm_processor.predict_price(features)
                        
                        
                        prediction_entry = {
                            'description': user_input,
                            'price': prediction,
                            'features': features
                        }
                        st.session_state.prediction_history.append(prediction_entry)
                        st.session_state.last_prediction = prediction_entry

                       
                        tab1, tab2, tab3 = st.tabs(["💰 Prediction", "📊 Features", "🗺️ Location"])

                        with tab1:
                            st.success(f"Predicted Price: ${prediction:,.2f}")
                            st.write(result)

                        with tab2:
                            self.create_features_chart(features)
                            
                        with tab3:
                            self.create_map(features['latitude'], features['longitude'])

                    except Exception as e:
                        st.error(f"Error processing prediction: {str(e)}")
                else:
                    st.warning("Please enter a property description.")

        with col2:
           
            st.sidebar.header("Prediction Management")
            
            if st.sidebar.button("Clear History", type="secondary"):
                self.clear_prediction_history()

            if st.session_state.prediction_history:
                st.sidebar.subheader("Recent Predictions")
                for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:]), 1):
                    with st.sidebar.expander(f"Prediction {i}"):
                        st.write(f"**Price:** ${pred['price']:,.2f}")
                        st.write("**Features:**")
                        for k, v in pred['features'].items():
                            if k not in ['latitude', 'longitude']:
                                st.write(f"- {k.replace('_', ' ').title()}: {v}")

            
            if st.sidebar.checkbox("View Full Prediction History"):
                self.display_price_history()

def main():
    app = RealEstateApp()
    app.run()

if __name__ == "__main__":
    main()