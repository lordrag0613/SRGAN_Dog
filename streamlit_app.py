# app.py

import streamlit as st
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from model import resolve_single
from utils import load_image
from model.srgan import generator

# Configuration GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Dossier pour les poids
WEIGHTS_DIR = 'weights/srgan'
weights_file = lambda filename: os.path.join(WEIGHTS_DIR, filename)

# Chargement des générateurs pré-entrainés
pre_generator = generator()
gan_generator = generator()

pre_generator.load_weights(weights_file('pre_generator.weights.h5'))
gan_generator.load_weights(weights_file('gan_generator.weights.h5'))

# Fonction pour afficher les résultats
def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)
    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)

    # Affichage des résultats
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(lr)
    ax[0].set_title("Low Resolution")
    ax[0].axis('off')

    ax[1].imshow(pre_sr)
    ax[1].set_title("Super-Resolution (Pretrained)")
    ax[1].axis('off')

    ax[2].imshow(gan_sr)
    ax[2].set_title("Super-Resolution (GAN)")
    ax[2].axis('off')

    st.pyplot(fig)

# Interface Streamlit
st.title("SRGAN: Super-Resolution GAN")

st.write("Cette application permet d'améliorer la résolution d'images de chiens à l'aide de SRGAN.")

uploaded_file = st.file_uploader("Téléchargez une image basse résolution", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Sauvegarde de l'image téléchargée
    image_path = os.path.join("demo", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Résolution et affichage
    resolve_and_plot(image_path)

st.write("Résultats de quelques images prédéfinies:")
resolve_and_plot('demo/dog-crop.jpg')
resolve_and_plot('demo/husky_test.jpg')
