import os
import streamlit as st
from src.training.trainer import YOLOv5Trainer

st.title("YOLOv5 Training")

trainer = YOLOv5Trainer()
st.write(trainer.get_device_info())

# Input dataset dalam format .zip
data_zip = st.file_uploader("Upload dataset ZIP file", type="zip")

# Input batch size dan epoch
batch_size = st.number_input("Batch Size", min_value=1, value=16)
epochs = st.number_input("Epochs", min_value=1, value=50)

# Tombol untuk memulai training
if st.button("Start Training") and data_zip is not None:
    # Create a temporary directory for the dataset
    temp_data_dir = "temp_dataset"
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)

    # Save file ZIP dataset to disk
    zip_path = os.path.join(temp_data_dir, "data.zip")
    with open(zip_path, "wb") as f:
        f.write(data_zip.getbuffer())

    # Extract dataset
    extract_path = os.path.join(temp_data_dir, "extracted_data")
    trainer.extract_zip(zip_path, extract_path)

    # Run YOLOv5 training
    st.text("Training in progress...")
    if trainer.train(os.path.join(extract_path, os.listdir(extract_path)[0]), batch_size, epochs, st.text):
        st.success("Training completed!")
    else:
        st.error("Training failed. Check the logs for details.")

# Tombol untuk menyimpan model hasil training
if st.button("Save Model"):
    if trainer.save_model():
        st.success("Model saved successfully to the 'models' directory!")
    else:
        st.error("Failed to save model. Ensure training was successful and weights exist.")
