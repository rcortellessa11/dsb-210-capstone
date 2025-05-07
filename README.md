# Smart Food Planner Capstone Project

### Problem Statement
Audience: 
- Beginner at-home cooks looking for a way an easier to find recipes that work
Problem:
- Sometimes you have a stocked fridge or just a few random ingredients but you have no idea what to make. Manually searching for different recipes can be time consuming
Goals / Objectives:
- Provide an easy, simple tool for at-home beginner cooks, that after some user input returns recipe options that best align with the users ingredients and also aims to accommodate partial recipe matches as well as allowing the user to filter recipes based on certain preferences (time, nutritional information)
Metrics for Success:
- Cosine similarity between the embedding vector or a given recipe compared to the embedding vector of a users ingredients


### Data Sets
- Scraped all recipes from allrecipes.com
- Each recipes url is saved in data/scraped_urls
- Final Dataframe modeled on: data/model_df.csv

## Data Dictionary for model_df.csv

- title (obj) : A recipes unique title
- intro (obj) : A breif description of each recipe
- total_time (int): The total time in minutes it takes for a recipe to be completed
- servings (float): The amount of servings per recipe
- recipe_url (float): The unique recipe's url
- calories (int): The number of calories in a single serving
- fat	(int): The number in grams of fat in a singls serving
- carbs (int): The number in grams of carbs in a singls serving
- protein	(int): The number in protein of fat in a singls serving
- time_category (float): A clustered time category for filtering purposes (30 minutes or less, Hour or less, Long recipes! (60-120), Livin in the kitchin (above 2 hrs))
- calorie_category (float): A clustered calorie category for filtering purposes (Low cal (<300), Average (300–600), High (600–1000), Don't look! (>1000))
- fat_category (float): A clustered fats category for filtering purpose (Low Fat! (< 10), Average Fat! (11-25), High Fat! (> 25)
- carbs_category (float): A clustered carbs category for filtering purpose (Low Carbs! (< 25), Average Carbs! (25-80), High Carbs! (> 80))
- protein_category (float): A clustered protein category for filtering purpose (Low Protein! (< 10), Average Protein! (11-29), High Protein! (> 30))
- tokenized_ingredients (float): Completely cleaned ingredients column ready to be input into model (every ingredient is lowecase, stripped of extra white space, lemmatized, and multi words are connected with a '_')
- embedding (float): The average vector value for each recipe that the users inputed ingredients list will be compared with

### Data Findings
#### Removal of any outliers or data imputation

Nulls:
- Null values in 'intro', 'prep_time', 'cook_time', 'total_time', 'servings', 'nutrition', 'recipe_url'
- Dropped a total of 161 rows containing null values (64 rows in recipe_urls, 23 rows in intro, 88 rows in nutrition, 4 rows in serving, 9 rows in total_time)
- 14 null rows remaining in column prep_time: The user will have the option to filter recipes based on total time so it's not vital that all prep_time rows are filled
- 187 null rows remaining in column cook_time: Same reasoning used from prep_time
  
Duplicates:
- There were 273 duplicated recipes in the data frame
- 273 rows dropped from dataframe
- Left with 1_870 unique recipes
  
Outliers:
- There were outliers in terms of total time and nutritional information however no outliers were removed because they were still valid recipes with higher values
  
Data Imputation
- No data was manually inputed


### Brief summary of analysis & Conclusions/Recommendations
There is an inbalance of classes in the model. There are roughly 600 more rows of data in the personal finance class and as expected the baseline model is poor. The baseline model would accurately predict that a post belongs to the subreddit r/PersonalFinance with 60.89 % accuracy.

During Preprocessing: all punctuation, extra characters, url's, extra spaces, and the 'english' stopwords were removed from the combined title and text column. After data was cleaned tokenization was applied, and lemmatization was implemented to ensure all words got reduced to their base.
Feature Extraction: The data was converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorizater method. This vectorizer captures the importance of words in the context of the subreddits by measuring the frequency and importance of a word in individual documents (posts) and then across the whole dataframe.
Model Selections: I fit 4 different models: Logistic Regression, Random Forest, XGBoost, SVC. All the models, had a stratified train test split on y in order to help with the class imbalance however, all the models besides the Logistic Regression ranged from overfit to extremely overfit. 
Evaluation: The models were evaluated using metrics of accuracy, precision, recall, and F1 score represented visually by the confusion matrix
Results: The Logistic Regression model was chosen as my main model due to its good score and the models interprability. It ended with an accuracy of 88% and an F1 average weighted score of 0.88 meaning there is a good balance of precision and recall classifying both the investing and personal finance subreddits well. Missclassifying one over the other does not have any extra meaning or weight, they have similarities in topics and missclassifying either isn't detrimental. 

Next Steps: - For future refinement/improvement of the model, I highly recommend gathering more data from the daily posts. The more posts the were appended to the train and test data the higher the accuracy score became. More data also helps prevent overfitting in the model which I found to be very common throughout the modeling process.

