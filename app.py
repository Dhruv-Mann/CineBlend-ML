import streamlit as st
import streamlit_shadcn_ui as ui
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. SETUP & DATA LOADING ---
@st.cache_resource
def load_data():
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movie_dict)
    
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    vectors = pickle.load(open('vectors.pkl', 'rb'))
    
    return movies, similarity, vectors

movies, similarity, vectors = load_data()

# --- 2. SIDEBAR & TITLE ---
st.set_page_config(page_title="CineBlend", layout="centered")
st.sidebar.header("CineBlend 🍿")

# Using Shadcn Select for the Mode Switcher (Looks cleaner)
mode = st.sidebar.radio(
    "Select Mode:",
    ("Mode A: Classic Recommend", "Mode B: The Blend (Dual)")
)

st.title("CineBlend AI 🎬")
st.markdown("---")

# --- 3. MODE A: CLASSIC ---
if mode == "Mode A: Classic Recommend":
    st.subheader("Find similar movies")
    
    # Keep st.selectbox for the movie list (Better performance for 5000+ items)
    selected_movie_name = st.selectbox(
        "Type or select a movie",
        movies['title'].values
    )

    # SHADCN BUTTON
    # 'primary' variant makes it black/bold
    if ui.button(text="Recommend", variant="primary", key="btn_a"):
        
        movie_index = movies[movies['title'] == selected_movie_name].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        st.write("### Recommended:")
        
        # Display results as SHADCN CARDS
        cols = st.columns(1) # Vertical list looks better for text cards
        for i in movies_list:
            movie_title = movies.iloc[i[0]].title
            match_score = f"{round(i[1] * 100)}% Match"
            
            # ui.card renders a beautiful box with Title, Value, and Description
            ui.card(
                title=movie_title, 
                content=match_score, 
                description="Based on Plot, Cast & Crew", 
                key=f"card_{i[0]}"
            ).render()

# --- 4. MODE B: THE BLEND ---
elif mode == "Mode B: The Blend (Dual)":
    st.subheader("Blend two movies together")
    
    col1, col2 = st.columns(2)
    with col1:
        movie_1 = st.selectbox("Select Movie 1", movies['title'].values, index=0)
    with col2:
        movie_2 = st.selectbox("Select Movie 2", movies['title'].values, index=1)

    # SHADCN BUTTON
    if ui.button(text="Blend & Recommend", variant="primary", key="btn_b"):
        
        idx1 = movies[movies['title'] == movie_1].index[0]
        idx2 = movies[movies['title'] == movie_2].index[0]
        
        vec1 = vectors[idx1]
        vec2 = vectors[idx2]
        
        blend_vector = (vec1 + vec2) / 2
        blend_vector = blend_vector.reshape(1, -1) 
        
        new_similarity = cosine_similarity(blend_vector, vectors)[0]
        movies_list = sorted(list(enumerate(new_similarity)), reverse=True, key=lambda x: x[1])[0:7]
        
        st.write(f"### The Mathematical Child of **{movie_1}** & **{movie_2}**:")
        
        count = 0
        for i in movies_list:
            title = movies.iloc[i[0]].title
            if title != movie_1 and title != movie_2:
                match_score = f"{round(i[1] * 100)}% Similarity"
                
                ui.card(
                    title=title, 
                    content=match_score, 
                    description="Hybrid Vector Match", 
                    key=f"blend_card_{i[0]}"
                ).render()
                
                count += 1
                if count == 5:
                    break