import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
 
df = pd.read_csv("data/product_reviews_full.csv")
 
# drop all rows with missing values
df = df.dropna()
 
# Convert all sentiment values to lowercase and strip extra spaces
df['sentiment'] = df['sentiment'].astype(str).str.lower().str.strip()
 
# Convert column type to 'category'
df['sentiment'] = df['sentiment'].astype('category')
 
 
# Drop columns that are not useful for modeling
df = df.drop(columns=['review_uuid', 'product_name', 'product_price'])
 
# Create new column with length of each review_text
df['review_length'] = df['review_text'].astype(str).str.len()
 
# Define features and label
X = df[["review_title", "review_text", "review_length"]]
y = df["sentiment"]
 
# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "review_title"),
        ("text", TfidfVectorizer(), "review_text"),
        ("length", MinMaxScaler(), ["review_length"])
    ]
)
 
# Define pipeline with the best model (e.g. RandomForestClassifier)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier())
])
 
# Train the model on the entire dataset
pipeline.fit(X, y)
 
# Save the model to a file
joblib.dump(pipeline, "model/sentiment_model.pkl")
 
print(" Model trained and saved as 'model/sentiment_model.pkl'")