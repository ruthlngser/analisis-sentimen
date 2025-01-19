import matplotlib.pyplot as plt

def show_visualisasi(data):
    # Menghitung jumlah data untuk setiap sentimen
    sentiment_counts = data['Label'].value_counts()

    # Label untuk diagram
    labels = sentiment_counts.index.tolist()
    label_mapping = {0: 'Negatif', 1: 'Positif', 2: 'Netral'}
    labels = [label_mapping[label] for label in labels]

    # Membuat subplot untuk diagram batang dan lingkaran
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Diagram Batang
    ax1.bar(labels, sentiment_counts, color=['red', 'green', 'blue'])
    ax1.set_title('Distribusi Sentimen (Diagram Batang)')
    ax1.set_xlabel('Sentimen')
    ax1.set_ylabel('Jumlah Data')

    # Diagram Lingkaran
    ax2.pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'green', 'blue'])
    ax2.set_title('Distribusi Sentimen (Diagram Lingkaran)')
    ax2.axis('equal')

    return fig # Mengembalikan figure yang berisi kedua diagram