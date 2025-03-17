import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

class SpotifyRecommender:
    def __init__(self):
        # Initialize Spotify client with necessary permissions
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=os.getenv('SPOTIFY_CLIENT_ID'),
            client_secret=os.getenv('SPOTIFY_CLIENT_SECRET'),
            redirect_uri=os.getenv('SPOTIFY_REDIRECT_URI'),
            scope='user-library-read playlist-modify-public user-top-read'
        ))
        
    def get_user_top_tracks(self, limit=50, time_range='medium_term'):
        """Get user's top tracks"""
        return self.sp.current_user_top_tracks(limit=limit, offset=0, time_range=time_range)
    
    def get_track_features(self, track_ids):
        """Get audio features for tracks"""
        features = self.sp.audio_features(track_ids)
        return features
    
    def create_recommendation_playlist(self, name="Your Recommended Playlist"):
        """Create a new playlist with recommended songs"""
        # Get current user's ID
        user_id = self.sp.current_user()['id']
        
        # Get user's top tracks
        top_tracks = self.get_user_top_tracks()
        track_ids = [track['id'] for track in top_tracks['items']]
        
        # Get audio features for top tracks
        features = self.get_track_features(track_ids)
        
        # Create a DataFrame with the features
        feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                       'speechiness', 'acousticness', 'instrumentalness',
                       'liveness', 'valence', 'tempo']
        
        df = pd.DataFrame(features)[feature_cols]
        
        # Scale the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df)
        
        # Calculate the average feature vector
        avg_features = np.mean(features_scaled, axis=0)
        
        # Get recommendations based on top tracks
        recommendations = self.sp.recommendations(
            seed_tracks=track_ids[:5],
            limit=30
        )
        
        # Create new playlist
        playlist = self.sp.user_playlist_create(user_id, name, public=True)
        
        # Add tracks to playlist
        recommended_track_uris = [track['uri'] for track in recommendations['tracks']]
        self.sp.playlist_add_items(playlist['id'], recommended_track_uris)
        
        return playlist['external_urls']['spotify']

def main():
    try:
        recommender = SpotifyRecommender()
        playlist_url = recommender.create_recommendation_playlist()
        print(f"\nSuccess! Your playlist has been created.")
        print(f"You can find it here: {playlist_url}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 