import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

# ----------------- CONFIG & STYLING -----------------
st.set_page_config(
    page_title="CinemAI | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme & Glassmorphism CSS
st.markdown("""
<style>
/* Dark Theme Base */
body {
    background-color: #0E1117;
    color: white;
}
/* Main App Background Gradient */
.stApp {
    background: linear-gradient(to top, #0E1117 0%, #1a1c29 100%);
}

/* Glassmorphism Card Style */
.movie-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 25px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    height: 100%;
    display: flex;
    flex-direction: column;
}

/* Selected Movie Highlights */
.selected-card {
    background: rgba(255, 0, 0, 0.05);
    border: 1px solid rgba(229, 9, 20, 0.5);
    box-shadow: 0 0 20px rgba(229, 9, 20, 0.2);
}
.selected-card:hover {
    box-shadow: 0 0 30px rgba(229, 9, 20, 0.4);
}

/* Hover Animations */
.movie-card:hover {
    transform: translateY(-5px) scale(1.02);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.2);
}



/* Typography styles */
.movie-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin-bottom: 5px;
    color: #ffffff;
}

h1 {
    font-weight: 800 !important;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #FF0000, #ff7373);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 20px;
}

/* Metric styling */
.metric-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    color: #cccccc;
    margin-bottom: 10px;
}
.metric-badge {
    background: rgba(255, 0, 0, 0.2);
    color: #ffcccc;
    padding: 3px 8px;
    border-radius: 5px;
    font-weight: 600;
    font-size: 0.8rem;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #E50914;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 5px;
    padding: 10px 25px;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    background-color: #f40612;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING -----------------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "movies.csv")
    model_path = os.path.join(script_dir, "model.pkl")

    if not os.path.exists(csv_path):
        st.error(f"Error: Dataset not found. Please ensure 'movies.csv' is in {script_dir}")
        st.stop()
    if not os.path.exists(model_path):
        st.error("Error: Model file 'model.pkl' not found. Please run 'python train_model.py' first.")
        st.stop()

    movies_data = pd.read_csv(csv_path)
    model_data = joblib.load(model_path)
    
    # Extract unique genres dynamically
    # Genres are space separated in the original dataset 'genres' column, but let's clean it up
    # We will use the 'genres' column from the dataframe if available and split by space
    all_genres = []
    for genres_str in movies_data['genres'].dropna():
        genres_list = str(genres_str).split() # Assuming they are space separated based on preprocessing
        all_genres.extend(genres_list)
    unique_genres = sorted(list(set(all_genres)))
    
    return movies_data, model_data, unique_genres



# ----------------- HELPER FUNCTIONS -----------------
def format_currency(amount):
    if pd.isna(amount) or amount == 0:
        return "Unknown"
    return f"${amount:,.0f}"

# ----------------- MAIN APP -----------------
with st.spinner("Initializing Cinematic AI Engine..."):
    movies_data, model_data, unique_genres = load_data()

feature_vectors = model_data['feature_vectors']
list_of_all_titles = model_data['list_of_all_titles']

# Header
st.markdown("<h1>🍿 CinemAI Discovery Engine</h1>", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80", use_container_width=True)
st.sidebar.markdown("## 🎛️ Filter Controls")

# Filtering and Sorting Options
selected_genre = st.sidebar.selectbox("Filter by Genre", ["All"] + unique_genres)

# Filters for Recommendations
st.sidebar.markdown("### Recommendation Preferences")
min_rating = st.sidebar.slider("Minimum Rating (TMDB)", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
runtime_range = st.sidebar.slider("Runtime Range (minutes)", min_value=0, max_value=300, value=(60, 200), step=10)

sort_by = st.sidebar.selectbox("Sort Results By", ["Similarity Score", "Popularity", "Rating", "Release Year (Newest)"])

# --- MAIN METADATA AREA ---
st.markdown("### Step 1: Tell us what you like")

# Filter movie list for the dropdown based on genre selection
if selected_genre != "All":
    filtered_df = movies_data[movies_data['genres'].str.contains(selected_genre, case=False, na=False)]
    available_movies = filtered_df['title'].tolist()
else:
    available_movies = list_of_all_titles

if not available_movies:
    st.warning(f"No movies found for the genre '{selected_genre}'. Displaying all movies.")
    available_movies = list_of_all_titles

# User Selection
selected_movie = st.selectbox(
    "Select a movie you love:",
    options=[""] + available_movies,
    format_func=lambda x: "--- Select a Movie ---" if x == "" else x
)

st.markdown("<br>", unsafe_allow_html=True)

# Generate Button
col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
with col_btn2:
    recommend_pressed = st.button("Generate Recommendations", use_container_width=True)

if recommend_pressed:
    if selected_movie == "":
        st.warning("⚠️ Please select a movie first.")
    else:
        with st.spinner(f"Analyzing multi-dimensional feature space for '{selected_movie}'..."):
            
            # Find closest match (just to be safe, though dropdown ensures exact match)
            closest_matches = difflib.get_close_matches(selected_movie, list_of_all_titles)
            
            if not closest_matches:
                st.error("Movie not found in internal database.")
            else:
                close_match = closest_matches[0]
                index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
                
                # --- DISPLAY SELECTED MOVIE METADATA ---
                st.markdown("## 🎬 Selected Movie Details")
                
                selected_row = movies_data.iloc[index_of_the_movie]
                
                # Set base CSV Data
                sel_rating = selected_row.get('vote_average', 'N/A')
                sel_votes = selected_row.get('vote_count', 0)
                sel_runtime = f"{int(selected_row.get('runtime', 0))} min" if pd.notna(selected_row.get('runtime')) else "Unknown"
                sel_release = str(selected_row.get('release_date', 'Unknown'))[:4]
                sel_genres = selected_row.get('genres', 'Unknown')
                sel_overview = selected_row.get('overview', 'No overview available.')
                sel_director = selected_row.get('director', 'Unknown')
                
                sel_cast_raw = selected_row.get('cast', 'Unknown')
                if pd.notna(sel_cast_raw) and isinstance(sel_cast_raw, str):
                    sel_cast_display = " ".join(sel_cast_raw.split()[:5]) + "..."
                else:
                    sel_cast_display = "Data Not Available"
                    
                sel_budget = selected_row.get('budget', 0)
                sel_revenue = selected_row.get('revenue', 0)
                sel_countries = selected_row.get('production_countries', 'Unknown')
                sel_companies = selected_row.get('production_companies', 'Unknown')
                sel_popularity = selected_row.get('popularity', 0)

                # Selected Movie UI Container (Poster replaced by Text Blocks)
                st.markdown('<div class="movie-card selected-card">', unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align: left; padding-bottom: 10px;'>🎬 {close_match}</h1>", unsafe_allow_html=True)
                
                # Metrics Row
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("📅 Release Date", sel_release)
                m_col2.metric("⭐ Rating", f"{sel_rating}/10")
                m_col3.metric("🗳️ Vote Count", sel_votes)
                m_col4.metric("📊 Popularity", round(sel_popularity, 1) if pd.notna(sel_popularity) else "N/A")
                
                st.markdown("---")
                
                # Details Row
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    st.write(f"**🎥 Director:** {sel_director}")
                    st.write(f"**👥 Cast:** {sel_cast_display}")
                    st.write(f"**⏳ Runtime:** {sel_runtime}")
                    st.write(f"**🏷️ Genres:** {sel_genres}")
                with d_col2:
                    st.write(f"**💰 Budget:** {format_currency(sel_budget)}")
                    st.write(f"**💵 Revenue:** {format_currency(sel_revenue)}")
                    
                    
                with st.expander("📝 View Full Overview"):
                    st.write(sel_overview)
                    st.write(f"**Production Companies:** {sel_companies}")
                    
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
                
                # --- GENERATE RECOMMENDATIONS ---
                # Fast Vector Math
                movie_vector = feature_vectors[index_of_the_movie]
                similarity_scores = cosine_similarity(movie_vector, feature_vectors).flatten()
                
                # Pre-filter by sidebar constraints
                # If a specific genre is selected, retrieve indices of movies inside that genre
                genre_indices = None
                if selected_genre != "All":
                    genre_indices = set(filtered_df.index.tolist())
                
                filtered_indices = []
                for idx, score in enumerate(similarity_scores):
                    if idx == index_of_the_movie:
                        continue # Skip source movie
                        
                    # Strict Context Bound: Skip if it's not the selected genre
                    if genre_indices is not None and idx not in genre_indices:
                        continue
                        
                    row = movies_data.iloc[idx]
                    
                    # Apply min rating
                    if pd.notna(row.get('vote_average')) and row.get('vote_average', 0) < min_rating:
                        continue
                        
                    # Apply runtime constraints
                    if pd.notna(row.get('runtime')):
                        duration = row.get('runtime', 0)
                        if duration < runtime_range[0] or duration > runtime_range[1]:
                            continue
                    
                    filtered_indices.append((idx, score))

                # If no movies match constraints, fallback
                if not filtered_indices:
                    st.warning("Filters are too strict! Relaxing runtime and rating constraints to show top matches.")
                    filtered_indices = [(i, s) for i, s in enumerate(similarity_scores) if i != index_of_the_movie]

                # Sort based on User Preference
                if sort_by == "Similarity Score":
                    sorted_movies = sorted(filtered_indices, key=lambda x: x[1], reverse=True)
                else:
                    # We need to map back to dataframe for other sorts
                    temp_df = []
                    for idx, score in filtered_indices:
                        row = movies_data.iloc[idx].to_dict()
                        row['similarity_score'] = score
                        temp_df.append(row)
                        
                    temp_df = pd.DataFrame(temp_df)
                    
                    if not temp_df.empty:
                        if sort_by == "Popularity":
                            temp_df = temp_df.sort_values(by='popularity', ascending=False)
                        elif sort_by == "Rating":
                            temp_df = temp_df.sort_values(by='vote_average', ascending=False)
                        elif sort_by == "Release Year (Newest)":
                            # Attempt to parse year safely
                            temp_df['release_year'] = pd.to_datetime(temp_df['release_date'], errors='coerce').dt.year
                            temp_df = temp_df.sort_values(by='release_year', ascending=False)
                            
                        # Rebuild sorted_movies tuple list
                        sorted_movies = [(row['index'], row['similarity_score']) for _, row in temp_df.iterrows()]
                    else:
                        sorted_movies = sorted(filtered_indices, key=lambda x: x[1], reverse=True)

                # Get Top 6 just in case one is broken, but we display top 6 as the grid works well with 6
                top_recommendations = sorted_movies[:6]
                
                if len(top_recommendations) < 5 and selected_genre != "All":
                    st.warning(f"Note: Only {len(top_recommendations)} recommendation(s) fully matched the strict '{selected_genre}' filter with your sorting settings.")
                elif not top_recommendations:
                    st.error("No valid recommendations found matching your current constraints. Try broadening your Rating or Runtime sliders.")
                
                st.markdown(f"### 🎬 AI Recommendations based on *{close_match}*")
                
                # Netflix Style 3-Column Display
                for i in range(0, len(top_recommendations), 3):
                    cols = st.columns(3)
                    
                    for j in range(3):
                        if i + j < len(top_recommendations):
                            idx = top_recommendations[i + j][0]
                            score = top_recommendations[i + j][1]
                            movie_row = movies_data.iloc[idx]
                            
                            # 1. Start with robust local dataset data
                            title = movie_row['title']
                            rating = movie_row.get('vote_average', 'N/A')
                            votes = movie_row.get('vote_count', 0)
                            runtime = f"{int(movie_row.get('runtime', 0))} min" if pd.notna(movie_row.get('runtime')) else "Unknown"
                            release = str(movie_row.get('release_date', 'Unknown'))[:4]
                            genres = movie_row.get('genres', 'Unknown')
                            overview = movie_row.get('overview', 'No overview available.')
                            director = movie_row.get('director', 'Unknown')
                            budget = movie_row.get('budget', 0)
                            revenue = movie_row.get('revenue', 0)
                            
                            cast_raw = movie_row.get('cast', 'Unknown')
                            if pd.notna(cast_raw) and isinstance(cast_raw, str):
                                cast_display = " ".join(cast_raw.split()[:5]) + "..."
                            else:
                                cast_display = "N/A"
                                
                            similarity_percent = int(score * 100)
                            
                            # Card HTML Construction (No Image)
                            with cols[j]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <div class="movie-title">🎬 {title}</div>
                                    <div style="color: #ccc; margin-bottom: 15px;">📅 {release}</div>
                                    <div class="metric-row">
                                        <span class="metric-badge">AI Match: {similarity_percent}%</span>
                                        <span>⭐ {rating}/10</span>
                                    </div>
                                    <div class="metric-row" style="color: #999; font-size: 0.8rem;">
                                        <span>⏱️ {runtime}</span>
                                        <span>🗳️ {votes} votes</span>
                                    </div>
                                    <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
                                    <div style="font-size: 0.85rem; color: #bbb;">
                                        <strong>🎥 Director:</strong> {director}<br>
                                        <strong>👥 Cast:</strong> {cast_display}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Metadata Expander below the styling
                                with st.expander("📝 View Full Overview"):
                                    st.write(overview)
                                    st.write(f"**Genres:** {genres}")
                                    st.write(f"**Budget:** {format_currency(budget)}")
                                    st.write(f"**Revenue:** {format_currency(revenue)}")
                                
