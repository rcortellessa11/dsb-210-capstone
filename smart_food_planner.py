import streamlit as st

st.set_page_config(page_title="Smart Food Planner", layout="wide")
st.title("SMART FOOD PLANNER")

st.markdown("Get personalized recipes from a set of over 1800 unique recipes from allrecipes.com all based on what you have in your pantry!")

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def normalize_and_format_ingredient(ingredient):
    words = ingredient.strip().lower().split()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized)

# --- INGREDIENT INPUT ---
user_ingredients = st.text_input(
   "Enter your ingredients (minimum 4)! (please separate with commas):",
    placeholder="e.g. eggs, milk, green beans"
)

if user_ingredients:
    st.write("User entered ingredients:", user_ingredients)

ingredients_list = [
    normalize_and_format_ingredient(i)
    for i in user_ingredients.split(',')
    if i.strip()
]

st.write("Normalized Ingredients List:", ingredients_list)

# --- OPTIONAL FILTERS ---
st.subheader("Customize Your Recipes! (Optional)")

with st.container():
    cook_time = st.selectbox("Cook Time? (measured in minutes)", [
        "Don't Care", "30 min or less", "Hour or less", "1–2 hours"
    ])

    calories = st.selectbox("Calories?", [
        "Don't Care", "Low cal (<300)", "Average (300–600)", "High (600–1000)", "Don't look! (>1000)"
    ])

    fats = st.selectbox("Fats? (measured in grams)", [
        "Don't Care", "Low fat", "Average fat", "High fat"
    ])

    carbs = st.selectbox("Carbs? (measured in grams)", [
        "Don't Care", "Low carbs", "Average carbs", "High carbs"
    ])

    proteins = st.selectbox("Proteins? (measured in grams)", [
        "Don't Care", "Low protein", "Average protein", "High protein"
    ])

    show_macros = st.radio("Do you want to see calories & macros?", [
        "Yes please", "Ignorance is bliss 🙈"
    ])

# --- CTA BUTTON ---
if st.button("Show me the money!"):
    if ingredients_list:
        st.success("Fetching your personalized recipes... 💡 ")
        # 💡 Replace this with actual recipe recommendation logic
    else:
        st.error("Please enter at least 4 ingredients.")