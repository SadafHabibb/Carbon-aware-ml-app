import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Carbon-Aware ML", page_icon="🌿")
st.title("🌿 Carbon-Aware Machine Learning Dashboard")

project = st.sidebar.selectbox("Choose a project:", ["MNIST Digit Classifier", "Housing Price Predictor"])

if project == "MNIST Digit Classifier":
    st.header("📄 MNIST Digit Classifier")
    st.markdown("- Accuracy: **96.8%**")
    st.markdown("- Trained with CodeCarbon: **0.000379 kWh**")
    
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="Uploaded Image", width=150)
        img_array = np.array(image).reshape(1, 28, 28) / 255.0

        model = tf.keras.models.load_model("mnist_model.h5")
        prediction = model.predict(img_array)
        st.success(f"Predicted Digit: **{np.argmax(prediction)}**")

elif project == "Housing Price Predictor":
    st.header("🏠 Housing Price Predictor")
    st.markdown("- RMSE: **~0.73**")
    st.markdown("- Trained with CodeCarbon: **0.0004 kWh**")

    st.markdown("### Input Housing Data")
    MedInc = st.slider("Median Income", 0.0, 15.0, 5.0)
    AveRooms = st.slider("Average Rooms", 0.0, 10.0, 5.0)
    AveOccup = st.slider("Average Occupancy", 0.0, 10.0, 2.0)
    HouseAge = st.slider("House Age", 1, 50, 20)

    if st.button("Predict Price"):
        model = joblib.load("california_model.pkl")
        sample = np.array([[MedInc, AveRooms, 0, AveOccup, HouseAge, 0, 0, 0]])
        price = model.predict(sample)[0]
        st.success(f"💰 Estimated Price: ${price * 100000:.2f}")
