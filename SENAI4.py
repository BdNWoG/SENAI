import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors

# Step 1: Generate synthetic data (replace this with real data)
np.random.seed(42)
num_children = 100
num_activities = 50
num_dimensions = 15

children_skills = np.random.randint(1, 101, size=(num_children, num_dimensions))
activity_difficulty = np.random.randint(1, 101, size=(num_activities, num_dimensions))

# Example Input Data
child_index = 0
activity_index = 0
child_skills = children_skills[child_index]
activity_difficulty = activity_difficulty[activity_index]

# Step 2: Similarity matching using k-nearest neighbors
knn = NearestNeighbors(n_neighbors=1)
knn.fit(children_skills)

# Step 3: Match child with activity
_, nearest_child_index = knn.kneighbors(activity_difficulty.reshape(1, -1))
matched_child_rating = children_skills[nearest_child_index]

# Calculate improvement
def calculate_improvement(child_rating, activity_difficulty):
    return np.clip(np.random.normal(loc=5, scale=2, size=num_dimensions), 0, None)

improvement = calculate_improvement(matched_child_rating, activity_difficulty)

# Update ratings
def update_ratings(old_rating, improvement):
    return old_rating + improvement

new_child_rating = update_ratings(child_skills, improvement)

# Predict new ratings for the child
def predict_ratings(child_skills, activities_difficulty):
    return child_skills + np.random.normal(loc=0, scale=2, size=num_dimensions)

predicted_child_rating = predict_ratings(new_child_rating, activity_difficulty)

# Display results
print("Original Child Skills:", child_skills)
print("Activity Difficulty:", activity_difficulty)
print("Matched Child Rating:", matched_child_rating)
print("Improvement in Skills:", improvement)
print("New Child Rating:", new_child_rating)
print("Predicted Child Rating:", predicted_child_rating)
