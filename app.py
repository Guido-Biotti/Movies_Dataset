import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def extract_names(data):
    return [item['name'] for item in data]

# Load your dataset
# Make sure your dataset has columns: 'year', 'genre', 'revenue'
df = pd.read_csv('Data_Cleaned/Movies_MetaData_ETL.csv')

# Extract genres
df['genres'] = df['genres'].apply(eval)  # Convert string representation of list to actual list
df['genres'] = df['genres'].apply(extract_names)

# Explode the genres column to have one genre per row
df = df.explode('genres')

# Preprocess the data
# Group by 'release_year' and 'genres' to get aggregated metrics
grouped = df.groupby(['release_year', 'genres']).agg({
    'revenue': 'sum',      # Sum of revenue per genre per year
    'budget': 'mean',      # Average budget per genre per year
    'vote_count': 'sum'    # Total votes per genre per year
}).reset_index()

# Get the genre with the highest revenue for each year
max_revenue_genre = grouped.loc[grouped.groupby('release_year')['revenue'].idxmax()]

# Prepare the feature set (X) and target variable (y)
X = max_revenue_genre[['release_year', 'budget', 'vote_count']]
y = max_revenue_genre['genres']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Use recent average values of budget and vote_count for predictions
average_budget = grouped['budget'].mean()
average_vote_count = grouped['vote_count'].mean()

# Predict for the next 3 years using the average values
next_years = pd.DataFrame({
    'release_year': [2024, 2025, 2026],
    'budget': [average_budget] * 3,
    'vote_count': [average_vote_count] * 3
})

predicted_genres = model.predict(next_years)
for year, genre in zip(next_years['release_year'], predicted_genres):
    print(f'Predicted genre with the highest revenue for {year}: {genre}')