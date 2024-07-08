import pandas as pd
import pickle

# Load the Spotify songs dataset
df = pd.read_csv('data_moods.csv')

# Print the columns to debug
print(df.columns)

# Use the correct column names based on the printed output
track_col = 'name'
artist_col = 'artist'
mood_col = 'mood'

# Preprocess the dataset to map emotions to songs
emotion_to_songs = {}

for index, row in df.iterrows():
    song = f"{row[track_col]} by {row[artist_col]}"
    emotion = row[mood_col]

    if emotion in emotion_to_songs:
        emotion_to_songs[emotion].append(song)
    else:
        emotion_to_songs[emotion] = [song]

# Save the dictionary to a pickle file
with open('emotion_to_songs.pkl', 'wb') as f:
    pickle.dump(emotion_to_songs, f)

print("Emotion to Songs mapping saved to 'emotion_to_songs.pkl'")
