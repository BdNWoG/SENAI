import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Sample user-item interaction data (child, activity, rating)
data = [
    ('Child1', 'Reading', 4),
    ('Child1', 'Math', 3),
    ('Child1', 'Art', 5),
    ('Child2', 'Reading', 3),
    ('Child2', 'Math', 4),
    ('Child2', 'Art', 2),
    ('Child3', 'Reading', 5),
    ('Child3', 'Math', 2),
    ('Child3', 'Art', 4),
    ('Child4', 'Reading', 2),
    ('Child4', 'Math', 5),
    ('Child4', 'Art', 3),
]

# Convert the list to a Pandas DataFrame
df = pd.DataFrame(data, columns=['user', 'item', 'rating'])

# Define the Reader object to parse the data
reader = Reader(rating_scale=(1, 5))

# Load the data into Surprise's Dataset object
dataset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2, random_state=42)

# Build the SVD model
model = SVD()

# Train the model on the training set
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model's performance
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"RMSE: {rmse}, MAE: {mae}")

# Now, you can use the trained model to make personalized recommendations for a specific child
child_id = 'Child1'
activities = ['Reading', 'Math', 'Art']

recommended_activities = []

for activity in activities:
    predicted_rating = model.predict(child_id, activity).est
    recommended_activities.append((activity, predicted_rating))

# Sort recommended activities by predicted rating
recommended_activities.sort(key=lambda x: x[1], reverse=True)

print(f"Recommended activities for {child_id}: {recommended_activities}")
