import streamlit as st
import pickle
import pandas as pd

# 1. LOAD THE SAVED BRAINS
# We use a function with @st.cache_resource so it only loads once. 
# Otherwise, every time you click a button, the app would reload 500MB of data (slow).

@st.cache_resource
@st.cache_resource
def load_data():
    # ERROR WAS HERE: Changed 'wb' to 'rb'
    movie_dict = pickle.load(open('movie_dict.pkl', 'rb')) 
    movies = pd.DataFrame(movie_dict)
    
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    
    # vectors = pickle.load(open('vectors.pkl', 'rb')) 
    
    return movies, similarity
# Load them up!
movies, similarity = load_data()

# 2. THE SIDEBAR (Navigation)
st.sidebar.title("CineBlend 🍿")
mode = st.sidebar.radio(
    "Select Mode:",
    ("Mode A: Classic Recommend", "Mode B: The Blend (Dual)")
)

# 3. MAIN PAGE LOGIC
st.title("CineBlend AI")

if mode == "Mode A: Classic Recommend":
    st.subheader("Find similar movies")
    
    # Dropdown to select a movie
    selected_movie_name = st.selectbox(
        "Type or select a movie from the dropdown",
        movies['title'].values
    )

    if st.button('Recommend'):
        st.write(f"You selected: {selected_movie_name}")
        # We will add the logic here in the next step
        
elif mode == "Mode B: The Blend (Dual)":
    st.subheader("Blend two movies together")
    st.write("Coming soon...")