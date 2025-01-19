import streamlit as st
import prediksi
import visualisasi
from visualisasi import show_visualisasi
import pandas as pd
# Set page config
st.set_page_config(
    page_title="Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mapping label ke deskripsi teks
label_to_sentiment = {
    0: "Negatif",
    1: "Positif",
    2: "Netral"
}

# Pilihan halaman di sidebar
pages = ["Halaman Beranda", "Visualisasi Data", "Confusion Matrix"]
choice = st.sidebar.selectbox("Pilih Halaman", pages)

if choice == "Halaman Beranda":
    st.title("Selamat Datang di Halaman Beranda")
    st.write("Aplikasi ini membantu menganalisis sentimen dengan model yang telah dilatih.")
    
    # Form input untuk teks yang akan dianalisis
    user_input = st.text_area("Masukkan teks untuk dianalisis")
    if st.button("Analisis"):
        model, vectorizer = prediksi.load_model()
        prediction_label = model.predict(vectorizer.transform([user_input]))[0]
        prediction_text = label_to_sentiment.get(prediction_label, "Tidak Diketahui")  # Konversi label ke teks
        st.write(f"Hasil analisis sentimen: **{prediction_text}**")

elif choice == "Visualisasi Data":
    # Pastikan Anda memiliki data yang sudah dianalisis untuk visualisasi
    # Misalnya, kita gunakan DataFrame untuk contoh ini
    data = pd.DataFrame({
        'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral', 'Positive']
    })
    
    # Panggil fungsi visualisasi dan berikan data sebagai argumen
    visualisasi.show_visualisasi(data)

elif choice == "Confusion Matrix":
    prediksi.show_confusion_matrix()
