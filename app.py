import streamlit as st
import requests
import base64
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# API configuration
API_BASE = "http://localhost:8000"

def main():
    st.title("🏨 Hotel Booking Analytics & Q&A")
    
    # File upload section
    with st.expander("📁 Upload & Process Data"):
        uploaded_file = st.file_uploader("Upload hotel bookings CSV", type="csv")
        if uploaded_file:
            with open("uploaded_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            response = requests.post(
                f"{API_BASE}/process",
                json={"file_path": "uploaded_data.csv"}  # Send as JSON object
            )
    
    # Analytics section
    with st.expander("📊 View Analytics"):
        if st.button("Generate Analytics Report"):
            response = requests.post(f"{API_BASE}/analytics")
            if response.status_code == 200:
                data = response.json()
                
                # Display stats
                st.subheader("Key Statistics")
                col1, col2 = st.columns(2)
                col1.metric("Cancellation Rate", f"{data['stats']['cancellation_rate']:.2f}%")
                col2.write("**Top Countries:**")
                for country, count in data['stats']['top_countries'].items():
                    col2.write(f"- {country}: {count}")
                
                # Display plots
                st.subheader("Visual Analytics")
                for plot_name in data['analytics']:
                    img_bytes = base64.b64decode(data['analytics'][plot_name])
                    img = mpimg.imread(BytesIO(img_bytes))
                    st.image(img, 
                           caption=plot_name.replace('_', ' ').title(), 
                           use_container_width=True)  # Changed here
            else:
                st.error(f"Error: {response.json()['detail']}")
    
    # Q&A Section
    with st.expander("❓ Ask Questions"):
        question = st.text_input("Ask about hotel bookings:")
        if question:
            response = requests.post(
                f"{API_BASE}/ask",
                json={"question": question}  # Send as JSON object
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Answer")
                st.info(data['answer'])
                
                with st.expander("See context used"):
                    st.write(data['context'])
            else:
                st.error(f"Error: {response.json()['detail']}")

if __name__ == "__main__":
    main()