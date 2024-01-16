import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Example activities and children with features
activities_data = [
    {'Activity': 'Reading', 'ReadingSkill': 3, 'MathSkill': 5},
    {'Activity': 'Math', 'ReadingSkill': 4, 'MathSkill': 3},
    {'Activity': 'Art', 'ReadingSkill': 2, 'MathSkill': 4},
    {'Activity': 'Music', 'ReadingSkill': 5, 'MathSkill': 2}
]

children_data = {
    1: {'ChildID': 1, 'ReadingSkill': 4, 'MathSkill': 3},
    2: {'ChildID': 2, 'ReadingSkill': 3, 'MathSkill': 4},
    3: {'ChildID': 3, 'ReadingSkill': 2, 'MathSkill': 5},
    4: {'ChildID': 4, 'ReadingSkill': 5, 'MathSkill': 2}
}

# Ensure the features are consistent
features = ['ReadingSkill', 'MathSkill']

# Adjust the calculate_similarity method
class RecommendationSystem:
    def __init__(self, children_data, activities_data):
        self.children_data = children_data
        self.activities_data = activities_data
        self.features = features
        self.similarity_matrix = self.calculate_similarity_matrix()

    def calculate_similarity(self, child1, child2):
        # Calculate similarity based on common features
        child1_features = np.array([child1[feature] for feature in self.features])
        child2_features = np.array([child2[feature] for feature in self.features])

        return np.dot(child1_features, child2_features) / (np.linalg.norm(child1_features) * np.linalg.norm(child2_features))

    def calculate_similarity_matrix(self):
        # Create a matrix where rows represent children and columns represent children
        matrix = np.zeros((len(self.children_data), len(self.children_data)))

        for i, child1 in enumerate(self.children_data.values()):
            for j, child2 in enumerate(self.children_data.values()):
                matrix[i, j] = self.calculate_similarity(child1, child2)

        # Use cosine similarity for collaborative filtering
        similarity_matrix = cosine_similarity(matrix)

        return similarity_matrix

    def recommend_activities(self, child_id):
        # Find the index of the given child in the similarity matrix
        child_index = list(self.children_data.keys()).index(child_id)

        # Get the most similar children based on the similarity matrix
        similar_children = np.argsort(self.similarity_matrix[child_index])[::-1][1:]

        # Get activities that the most similar children have succeeded in
        recommended_activities = self.get_activities_based_on_similar_children(similar_children)

        return recommended_activities

    def get_activities_based_on_similar_children(self, similar_children):
        # Placeholder logic (in a real-world scenario, this would be more sophisticated)
        # For each similar child, recommend the activities they have succeeded in
        recommended_activities = []

        for i in similar_children:
            recommended_activities.extend(self.activities_data)

        return recommended_activities

# Example usage:
recommendation_system = RecommendationSystem(children_data, activities_data)
child_id = 2
recommended_activities = recommendation_system.recommend_activities(child_id)

print(f"Recommended activities for child {child_id}: {recommended_activities}")
