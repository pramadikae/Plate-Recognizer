import streamlit as st

st.set_page_config(page_title="Plate Recognizer", page_icon="🚗")

def home_page():
    st.title("Plate Recognizer")
    st.write(
        "Selamat datang di aplikasi Plate Recognizer. Aplikasi ini memungkinkan Anda untuk mendeteksi dan mengenali plat nomor kendaraan."
    )
    st.write("Silakan pilih menu di sidebar untuk memulai.")

if __name__ == "__main__":
    home_page()

