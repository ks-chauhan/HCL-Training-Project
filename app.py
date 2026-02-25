import streamlit as st
from utils import load_artifacts, predict

st.title("News Topic Classifier")

# Load Model
try:
    tokenizer, model, label_mapping = load_artifacts()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

# User Input
user_input = st.text_area("Enter news text for classification:")

# Prediction
if st.button("Predict"):

    # Validate input
    if not user_input.strip():
        st.warning("Please enter some text for prediction.")
    
    # Make prediction
    else:
        try:
            pred_id, probs = predict(user_input, tokenizer, model)
            pred_label = label_mapping[pred_id]

            st.success("Prediction successful!")
            
            st.subheader("Prediction Result")
            st.write(f"**Predicted Topic:** {pred_label}")
            st.write("**Probabilities:**")
            for label_id, prob in enumerate(probs):
                label_name = label_mapping[label_id]
                st.write(f"{label_name}: {prob:.4f}")
        
        except ValueError as e:
            st.error(f"inference failed: {e}")