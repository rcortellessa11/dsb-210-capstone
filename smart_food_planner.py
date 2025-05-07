import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
import re

# Set page 
st.set_page_config(page_title = 'Smart Food Planner', layout = 'wide')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/model_df.csv')  # Update path if needed
    df['tokenized_ingredients'] = df['ingredients_str'].apply(lambda x: x.split())
    return df

df = load_data()

# Word2Vec model
@st.cache_resource
def train_word2vec(tokenized_data):
    return Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=4)

model = train_word2vec(df['tokenized_ingredients'])

# Create recipe embeddings
@st.cache_data
def get_average_embedding(tokens, _model):
    vectors = [_model.wv[word] for word in tokens if word in _model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(_model.vector_size)

@st.cache_data
def compute_recipe_embeddings(df, _model):
    return np.vstack(df['tokenized_ingredients'].apply(lambda tokens: get_average_embedding(tokens, _model)))

recipe_vectors = compute_recipe_embeddings(df, model)

#Streamlit UI
st.title('SMART FOOD PLANNER')
st.markdown('Get personalized recipes from a set of over 1800 unique recipes from allrecipes.com — all based on what you have in your pantry!')

# Ingredient normalization
lemmatizer = WordNetLemmatizer()

def normalize_and_format_ingredient(ingredient):
    words = re.findall(r'\b\w+\b', ingredient.lower())  # remove punctuation
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return '_'.join(lemmatized)

# Ingredient match scoring
def ingredient_match_percent(user_ingredients, recipe_ingredients):
    matches = sum(1 for ing in user_ingredients if ing in recipe_ingredients)
    return matches / len(user_ingredients) if user_ingredients else 0

# Substitution matching function
def find_substitutable_matches_for_df(df, user_pantry):
    
    def find_substitutable_matches(recipe_ingredients, user_pantry):
        matches = []
        missing = []
        
        recipe_list = recipe_ingredients.split()  # Split ingredients into words
        
        for ingredient in recipe_list:
            if ingredient in user_pantry:
                matches.append((ingredient, ingredient))  # Perfect match
            else:
                # Check word-level overlap
                ingredient_words = set(ingredient.split('_'))
                found = False
                for pantry_item in user_pantry:
                    pantry_words = set(pantry_item.split('_'))
                    if ingredient_words & pantry_words:  # Overlap between words
                        matches.append((ingredient, pantry_item))  # Substitution match
                        found = True
                        break
                if not found:
                    missing.append(ingredient)  # No match or substitution
        
        return matches, missing
    
    # Apply the substitution matching to each recipe in the dataframe
    df['matches'], df['missing'] = zip(*df['ingredients_str'].apply(lambda x: find_substitutable_matches(x, user_pantry)))
    
    return df

# User Ingredient Input
user_ingredients = st.text_input(
    'Enter your ingredients (minimum 4)! (please separate with commas):',
    placeholder= 'e.g. eggs, milk, green beans
)

if user_ingredients:
    ingredients_list = [
        normalize_and_format_ingredient(i.strip())
        for i in user_ingredients.split(',')
        if i.strip()
    ]

    if len(ingredients_list) < 4:
        st.error('Please enter at least 4 ingredients.')
    else:
        # Compute ingredient match %
        df['match_percent'] = df['tokenized_ingredients'].apply(
            lambda recipe_ings: ingredient_match_percent(ingredients_list, recipe_ings)
        )

        filtered_df = df[df['match_percent'] >= 0.6].copy()

        if filtered_df.empty:
            st.warning('No recipes found with at least 60% ingredient match. Try adding more ingredients.')
        else:
            # Optional filters
            st.subheader('Customize Your Recipes! (Optional)')
            with st.container():
                cook_time = st.selectbox('Cook Time? (measured in minutes)', [
                    "Don't Care", '30 minutes or less!', 'Hour or less!', 'Long recipes! (60-120)', 'Livin in the kitchin!'
                ])

                calories = st.selectbox("Calories?", [
                    "Don't Care", 'Low cal (<300)', 'Average (300–600)', 'High (600–1000)', "Don't look! (>1000)"
                ])

                fats = st.selectbox("Fats? (measured in grams)", [
                    "Don't Care", "Low Fat! (< 10)", "Average Fat! (11-25)", "High Fat! (> 25)"
                ])

                carbs = st.selectbox("Carbs? (measured in grams)", [
                    "Don't Care", "Low Carbs! (< 25)", "Average Carbs! (25-80)", "High Carbs! (> 80)"
                ])

                proteins = st.selectbox("Proteins? (measured in grams)", [
                    "Don't Care", "Low Protein! (< 10)", "Average Protein! (11-29", "High Protein! (> 30)"
                ])

                show_macros = st.radio("Do you want to see calories & macros?", [
                    "Yes please", "Ignorance is bliss"
                ])

            # --- Apply filters here ---
            if cook_time != "Don't Care":
                if cook_time == "30 minutes or less!":
                    filtered_df = filtered_df[filtered_df['total_time'] <= 30]
                elif cook_time == "Hour or less!":
                    filtered_df = filtered_df[filtered_df['total_time'] <= 60]
                elif cook_time == "Long recipes! (60-120)":
                    filtered_df = filtered_df[filtered_df['total_time'] <= 120]
                elif cook_time == "Livin in the kitchin!":
                    filtered_df = filtered_df[filtered_df['total_time'] > 120]

            if calories != "Don't Care":
                if calories == 'Low cal (<300)':
                    filtered_df = filtered_df[filtered_df['calories'] < 300]
                elif calories == 'Average (300–600)':
                    filtered_df = filtered_df[(filtered_df['calories'] >= 300) & (filtered_df['calories'] <= 600)]
                elif calories == 'High (600–1000)':
                    filtered_df = filtered_df[(filtered_df['calories'] > 600) & (filtered_df['calories'] <= 1000)]
                elif calories == "Don't look! (>1000)":
                    filtered_df = filtered_df[filtered_df['calories'] > 1000]

            if fats != "Don't Care":
                if fats == "Low Fat! (< 10)":
                    filtered_df = filtered_df[filtered_df['fat'] < 10]
                elif fats == 'Average Fat! (11-25)':
                    filtered_df = filtered_df[(filtered_df['fat'] >= 11) & (filtered_df['fat'] <= 25)]
                elif fats == 'High Fat! (> 25)':
                    filtered_df = filtered_df[filtered_df['fat'] > 25]

            if carbs != "Don't Care":
                if carbs == 'Low Carbs! (< 25)':
                    filtered_df = filtered_df[filtered_df['carbs'] < 25]
                elif carbs == 'Average Carbs! (25-80)':
                    filtered_df = filtered_df[(filtered_df['carbs'] >= 25) & (filtered_df['carbs'] <= 80)]
                elif carbs == 'High Carbs! (> 80)':
                    filtered_df = filtered_df[filtered_df['carbs'] > 80]

            if proteins != "Don't Care":
                if proteins == 'Low Protein! (< 10)':
                    filtered_df = filtered_df[filtered_df['protein'] < 10]
                elif proteins == 'Average Protein! (11-29)':
                    filtered_df = filtered_df[(filtered_df['protein'] >= 11) & (filtered_df['protein'] <= 29)]
                elif proteins == 'High Protein! (> 30)':
                    filtered_df = filtered_df[filtered_df['protein'] > 30]

            # After filtering, check again
            if filtered_df.empty:
                st.warning('No recipes matched your filters. Try relaxing one or more options.')

            # Apply substitution matching
            df_with_substitutions = find_substitutable_matches_for_df(filtered_df, ingredients_list)

            if st.button('Show me the money!'):
                user_vector = get_average_embedding(ingredients_list, model).reshape(1, -1)
                filtered_vectors = recipe_vectors[filtered_df.index]
                similarities = cosine_similarity(user_vector, filtered_vectors)[0]
                top_indices = similarities.argsort()[::-1][:5]
                top_recipes = df_with_substitutions.iloc[top_indices][['title', 'intro', 'recipe_url', 'ingredients_str', 'total_time', 'calories', 'protein', 'carbs', 'fat', 'matches', 'missing']]

                st.success("Fetching your personalized recipes... ")
                st.markdown("### Top Recipe Matches")
                for _, row in top_recipes.iterrows():
                    st.markdown(f"** {row['title']}**")
                    st.markdown(f"_{row['intro']}_")
                    st.markdown(f'[ View Full Recipe]({row['recipe_url']})')
                    st.markdown(f'**Main Ingredients:** `{row['ingredients_str']}`')
                    if show_macros == 'Yes please':
                        st.markdown(f"- **Time:** {row['total_time']} minutes | **Calories:** {row['calories']} | **Protein:** {row['protein']}g | **Carbs:** {row['carbs']}g | **Fat:** {row['fat']}g")

                    if len(row['matches']) > 0:
                        st.markdown(f"**Matches:** {', '.join([f'{m[0]} -> {m[1]}' for m in row['matches']])}")
                    if len(row['missing']) > 0:
                        st.markdown(f"**Missing:** {', '.join(row['missing'])}")

                    st.markdown("---")