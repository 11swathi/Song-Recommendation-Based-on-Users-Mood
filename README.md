# Emotion Based Music Recommendation

This is a Flask application that detects emotions from facial images and recommends songs based on the detected emotion using a pre-trained CNN model and a Spotify music dataset.

## Setup

1. Clone the repository.
2. Download the FER-2013 dataset and train the emotion detection model (`emotion_model.h5`).
3. Download the Spotify music dataset (`spotify_moods.csv`).
4. Preprocess the Spotify music dataset to create a pickle file (`emotion_to_songs.pkl`).

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
