import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Fungsi untuk memuat model
@st.cache_data
def load_model():
    model = pickle.load(open("skripsianalisis.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

def show_confusion_matrix():
    model, vectorizer = load_model()

    # Misalkan data testing dan hasil prediksi sudah ada
    # Gantilah dengan data dan prediksi yang sesuai
    # Contoh data dan prediksi (ubah dengan data nyata dari datasetmu)
    y_test = [0, 1, 2, 0, 1, 1, 2, 2, 0, 1]  # Label aktual
    y_pred = model.predict(vectorizer.transform([
        "Teks positif 1", "Teks negatif 1", "Teks netral 1", 
        "Teks positif 2", "Teks negatif 2", "Teks positif 3",
        "Teks netral 2", "Teks netral 3", "Teks negatif 3", "Teks positif 4"
    ]))  # Prediksi berdasarkan model

    # Buat confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualisasikan confusion matrix dengan heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Wistia',
                xticklabels=['Negatif', 'Positif', 'Netral'],
                yticklabels=['Negatif', 'Positif', 'Netral'])
    plt.title('Confusion Matrix')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')

    # Tampilkan plot
    st.pyplot(plt)
