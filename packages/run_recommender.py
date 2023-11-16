import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
sns.set_palette("Set2")
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity


def get_feature_vector(song_name, year, dat, features_list):
    print(dat.head())
    # select dat with the song name and year
    dat_song = dat.query('name == @song_name and year == @year')
    song_repeated = 0
    if len(dat_song) == 0:
        raise Exception('The song does not exist in the dataset or the year is wrong! \
                        \n Use search function first if you are not sure.')
    if len(dat_song) > 1:
        song_repeated = dat_song.shape[0]
        print(f'Warning: Multiple ({song_repeated}) songs with the same name and artist, the first one is selected!')
        dat_song = dat_song.head(1)
    feature_vector = dat_song[features_list].values
    return feature_vector, song_repeated

names = []

def dot_product(vector_a, vector_b):
    return sum(a * b for a, b in zip(vector_a, vector_b))

def magnitude(vector):
    return math.sqrt(sum(a**2 for a in vector))

def cosine_similarity_2d(array1, array2):
    
    similarities = []
    array2 = array2[0]  # Extract the single row from array2
    print(array2)
    print(len(array1))
    for row in array1:
        dot_product = np.dot(row, array2)
        magnitude_a = np.linalg.norm(row)
        magnitude_b = np.linalg.norm(array2)

        similarity = dot_product / (magnitude_a * magnitude_b) if (magnitude_a != 0 and magnitude_b != 0) else 0
        similarities.append(similarity)

    return similarities

# define a function to get the most similar songs
def show_similar_songs(song_name, year, dat, features_list, top_n=10, plot_type='wordcloud'):
    """
    Fungsi untuk mendapatkan lagu dengan tingkat kemiripan tertinggi berdasarkan rumus cosine similarity pada semua feature/ciri-ciri.
    :param song_name: Nama lagu (all letters)
    :param year: Tahun dari lagu tersebut [int]
    :param dat: dataset yang dipakai
    :param features_list: List feature yang dipakai untuk perhitungan kemiripan
    :param top_n: Banyaknya lagu yang akan direkomendasikan
    :param plot_type: Jenis plot yang dipakai untuk visualisasi (Wordcloud atau barplot)
    """
    feature_vector, song_repeated = get_feature_vector(song_name, year, dat, features_list)
    feature_for_recommendation = dat[features_list].values
    
    # menghitung nilai kemiripannya dengna cosine similarity
    similarities = cosine_similarity_2d(feature_for_recommendation, feature_vector)
    similarities = np.array(similarities)
    
    

    # mengambil index top_n, tidak termasuk lagu yang diinputkan
    if song_repeated == 0:
        related_song_indices = similarities.argsort()[-(top_n+1):][::-1][1:]
    else:
        related_song_indices = similarities.argsort()[-(top_n+1+song_repeated):][::-1][1+song_repeated:]
        
    # get the name, artist, and year of the most similar songs
    similar_songs = dat.iloc[related_song_indices][['name', 'artists', 'year']]
    
    names.clear()
    
    names.append(song_name)
    names.extend(similar_songs['name'].head().tolist())
    # names.extend(dat.iloc[related_song_indices]['name'].tolist())
    
    fig, ax = plt.subplots(figsize=(7, 5))
    if plot_type == 'wordcloud':
        # #Membuat Word cloud dari lagu dan tahun yang mirip, menggunakan nilai kemiripan untuk menentukan ukuran wordnya
        
        similar_songs['name+year'] = similar_songs['name'] + ' (' + similar_songs['year'].astype(str) + ')'
        
        # Membuat dictionary dari lagu dan nilai kemiripannya
        song_similarity = dict(zip(similar_songs['name+year'], similarities[related_song_indices]))
        
        # Mengurutkan dictionarynya berdasarkan nilainya
        song_similarity = sorted(song_similarity.items(), key=lambda x: x[1], reverse=True)
        
    

        # membuat word cloudnya
        wordcloud = WordCloud(width=1200, height=600, max_words=50, 
                            background_color='white', colormap='Set2').generate_from_frequencies(dict(song_similarity))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{top_n} most similar songs to: {song_name} ({year})', fontsize=16)
        plt.tight_layout(pad=0)
    
    elif plot_type == 'bar':
        # # Menggambarkan text dari lagu dan tahur termirip secara berurut dalam bentuk bar chart
        
        similar_songs['name+year'] = similar_songs['name'] + ' (' + similar_songs['year'].astype(str) + ')'
        
        # Membuat dictionary dari lagu dan nilai kemiripannya
        song_similarity = dict(zip(similar_songs['name+year'], similarities[related_song_indices]))
        
        # Mengurutkan dictionarynya berdasarkan nilainya
        song_similarity = sorted(song_similarity.items(), key=lambda x: x[1], reverse=True)
        
        # Menggambarkan text dari lagu dan tahur termirip secara berurut dalam bentuk bar chart
        plt.barh(range(len(song_similarity)), [val[1] for val in song_similarity], 
                 align='center', color=sns.color_palette('pastel', len(song_similarity)))
        plt.yticks(range(len(song_similarity)), [val[0] for val in song_similarity])
        plt.gca().invert_yaxis()
        plt.title(f'{top_n} most similar songs to: {song_name} ({year})', fontsize=16)
        min_similarity = min(similarities[related_song_indices])
        max_similarity = max(similarities[related_song_indices])
        
        # Menambahkan nama lagu disetiap bar chart
        for i, v in enumerate([val[0] for val in song_similarity]):
            plt.text(min_similarity*0.955, i, v, color='black', fontsize=8)
            
        plt.xlabel('Similarity', fontsize=15)
        # plt.ylabel('Song', fontsize=15)
        plt.xlim(min_similarity*0.95, max_similarity)
        
        # menghilangkan frame dan tanda 
        plt.box(False)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False)
        
    else:
        raise Exception('Plot type must be either wordcloud or bar!')
    
    return fig

def radar_chart(dat, features_list):
    # Membuat Radar Chart
    fig = go.Figure()
    angles = list(dat[features_list].columns)
    angles.append(angles[0])
    layoutdict = dict(
                radialaxis=dict(
                visible=True,
                range=[0, 1]
                ))


    for i in range(len(names)):
        subset = dat[dat['name'] == names[i]]
        data = [np.mean(subset[col]) for col in subset[features_list].columns]
        data.append(data[0])
        fig.add_trace(go.Scatterpolar(
            r=data,
            theta=angles,
            fill='toself',
            name=names[i]))

    

    fig.update_layout(
        polar=layoutdict,
        showlegend=True,
        template='plotly_dark'
    )
    
    return fig