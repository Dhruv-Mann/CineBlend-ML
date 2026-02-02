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