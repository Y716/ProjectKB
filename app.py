"""
This is the main script for the streamlit app of song recommender.
The script is written in Python 3.11.1. 
author: Yannan Su
date: 2023-05-10

modified by: Yasin
date: 2023-11-14
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO
import plotly.io as pio

from packages.search_song import search_song
from packages.run_recommender import get_feature_vector, show_similar_songs, radar_chart

# load data
dat = pd.read_csv('data/Processed/dat_for_recommender.csv')

song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

all_features = song_features_normalized + song_features_not_normalized + ['popularity']

# set app layout
# st.set_page_config(layout="wide")

# set a good looking font
st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



def main():
    st.markdown("# Sistem Rekomendasi Lagu hanya untukmu!")
    st.markdown("Selamat datang di Sistem Rekomendasi Lagu! \
                \n Kamu dapat memasukkan lagu dan mendapatkan rekomendasi berdasarkan ciri-ciri lagu yang telah kau masukkan. \
                \n Kamu juga dapat mengkostumisasi rekomendasinya dengan memilih ciri-ciri yang kamu inginkan. Selamat Mencari! \
                \n NB: Rekomendasi lagu ini hanya menelusuri lagu antara 2018 - 2020.     ")

    # add selectbox for selecting the features
    st.sidebar.markdown("### Memilih Ciri-ciri")
    features = st.sidebar.multiselect('Pilih ciri-ciri yang kamu inginkan', all_features, default=all_features)
    # add a slider for selecting the number of recommendations
    st.sidebar.markdown("### Banyak rekomendasi lagu yang didapatkan")
    num_recommendations = st.sidebar.slider('Pilih berapa banyak rekomendasi yang diinginkan', 10, 50, 10)

    # add a search box for searching the song by giving capital letters and year
    st.markdown("### Siap untuk mendapatkan rekomendasi dari lagu yang kamu masukkan?")
    song_name = st.text_input('Masukkan judul lagunya')
    if song_name != '':
        song_name = song_name.upper()
    year = st.text_input('Masukkan tahun dari lagu tersebut (contoh: 2019). \
                         \nJika kamu tidak yakin lagunya ada di databasenya atau tidak yakin dengan tahunnya, \
                         Tolong biarkan tahun lagunya kosong dan klik tombol dibawah ini untuk mencari lagu tersebut.')
    if year != '':
        year = int(year)

    # exmaples of song name and year:
    # song_name = 'YOUR HAND IN MINE'
    # year = 2003

    # add a button for searching the song if the user does not know the year
    if st.button('Cari lagu saya'):
        found_flag, found_song = search_song(song_name, dat)
        if found_flag:
            st.markdown("Wow, Lagu ini ada di dataset:")
            st.markdown(found_song)
        else:
            st.markdown("Maaf, lagu ini tidak ada di dataset. Tolong cari lagu yang lain!")

    # add a button for getting recommendations
    if st.button('Dapatkan Rekomendasi!'):
        if song_name == '':
            st.markdown("Tolong masukkan nama lagunya!")
        elif year == '':
            st.markdown("Tolong masukkan tahun lagunya!")
        else:
            
            # show the most similar songs in wordcloud
            fig_cloud = show_similar_songs(song_name, year, dat, features, num_recommendations, plot_type='wordcloud')
            st.markdown(f"### Keren! Inilah rekomendasinya dari lagu \
                        \n#### {song_name} ({year})!")
            st.pyplot(fig_cloud)

            # show the most similar songs in bar chart
            fig_bar = show_similar_songs(song_name, year, dat, features, top_n=10, plot_type='bar')
            st.markdown("### Lihatlah lebih dekat dari 10 lagu rekomendasi untukmu!")
            st.pyplot(fig_bar)
            
            #Menampilkan perbandingan radar chart antara lagu yang dimasukkan dengan 5 lagu teratas
            fig_radar = radar_chart(dat, song_features_normalized)
            st.markdown("### Gambaran Kemiripan Ciri-Ciri lagumu dengan Rekomendasinya!") 
            st.plotly_chart(fig_radar)

if __name__ == "__main__":
    main()

    
    


