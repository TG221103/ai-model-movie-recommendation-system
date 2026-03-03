import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def train_and_save_model(csv_path='movies.csv', model_path='model.pkl'):
    print(f"Loading data from {csv_path}...")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please make sure the CSV is in the correct location.")
        
    movies_data = pd.read_csv(csv_path)
    
    # Selecting the relevant features for recommendation
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    print("Selecting and cleaning features...")
    
    # Replacing the null values with null string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
        
    # Combining all the 5 selected features
    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    
    # Converting the text data to feature vectors
    print("Converting text data to feature vectors using TF-IDF...")
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    # Create list of all movie titles for difflib matching
    list_of_all_titles = movies_data['title'].tolist()
    
    # Bundle the required artifacts for the app
    # We save feature_vectors and list_of_all_titles. We don't save the full NxN similarity matrix to save space.
    # The app will compute 1xN similarity on the fly.
    model_data = {
        'feature_vectors': feature_vectors,
        'list_of_all_titles': list_of_all_titles
    }
    
    print(f"Saving model artifacts to {model_path}...")
    joblib.dump(model_data, model_path)
    print("Model training and saving completed successfully!")

if __name__ == '__main__':
    # Determine absolute path to the current working directory (project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'movies.csv')
    model_file_path = os.path.join(script_dir, 'model.pkl')
    
    train_and_save_model(csv_file_path, model_file_path)
