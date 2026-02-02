import pandas as pd
import ast # Abstract Syntax Tree - great for fixing "stringified" lists

# 1. LOAD THE DATA
# Make sure these CSV files are in the same folder as your script
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

print("Datasets loaded successfully!")

# 2. MERGE THE DATASETS
# We merge them on the 'title' column so we have everything in one big table
movies = movies.merge(credits, on='title')

# Keep only the columns we actually need for the recommendation engine
# We don't need 'budget' or 'homepage' for content similarity
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# 3. CLEANING THE JSON COLUMNS
# The data looks like: '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
# We want: ['Action', 'Adventure']

def convert(obj):
    L = []
    # ast.literal_eval safely evaluates a string containing a Python literal (like a list)
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

# Apply this function to our messy columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Check the results
print("\nData Cleaning Check:")
print(movies[['title', 'genres', 'keywords']].head())
# --- PHASE 2: FEATURE ENGINEERING ---

# 1. EXTRACT TOP 3 ACTORS
# The cast column is also a stringified list of dictionaries.
# We need the first 3 dictionaries' 'name' value.

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert3)

# 2. EXTRACT THE DIRECTOR
# We need to find the dictionary in 'crew' where job == 'Director'

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# 3. COLLAPSE SPACES (CRITICAL STEP)
# 'Sam Worthington' -> 'SamWorthington'
# We do this so 'Sam' from 'Sam Worthington' doesn't match 'Sam' from 'Sam Mendes'.
# We want unique identifiers for people and genres.

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# 4. CREATE THE "TAGS" COLUMN (THE SOUL OF THE MOVIE)
# We handle the 'overview' separately because it's a string, not a list.
# Let's turn overview into a list of words first so we can concatenate easily.
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Merge everything into one giant "tags" column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create the final clean DataFrame
new_df = movies[['movie_id', 'title', 'tags']]

# Convert list of tags back to a string for the Vectorizer
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower()) # Lowercase for consistency

print("\nFeature Engineering Complete!")
print(new_df.head())
# --- PHASE 3: THE ENGINE ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1. VECTORIZATION
# max_features=5000: We only care about the top 5000 most frequent words.
# stop_words='english': Remove words like "the", "a", "in" which are useless for similarity.
cv = CountVectorizer(max_features=5000, stop_words='english')

# fit_transform: Learn the vocabulary and turn our tags into a giant matrix of numbers.
# .toarray(): Scikit-learn returns a sparse matrix (compressed) by default; we expand it for clarity/math.
vectors = cv.fit_transform(new_df['tags']).toarray()

print("Vector shape:", vectors.shape) # Should be (4806, 5000)

# 2. CALCULATE COSINE SIMILARITY
# This calculates the distance from EVERY movie to EVERY other movie.
# It creates a 4806x4806 matrix.
similarity = cosine_similarity(vectors)

print("Similarity Matrix shape:", similarity.shape)
print("Similarity score example (First movie vs First movie):", similarity[0][0]) # Should be 1.0

# 3. EXPORT THE BRAIN (PICKLE)
# We save these objects so the website doesn't have to recalculate this every time it loads.
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(vectors, open('vectors.pkl', 'wb'))

print("Engine built and saved as .pkl files!")
