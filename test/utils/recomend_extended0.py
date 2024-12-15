import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score



def recommend_by_user1(user_id, user_item_matrix, similarity_matrix, n_recommendations=3):
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = similarity_matrix[user_idx]
    scores = np.dot(similar_users, user_item_matrix)
    items_recommendations = pd.Series(scores, index=user_item_matrix.columns)

    # Exclude books already rated by the user
    rated_items = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    items_recommendations = items_recommendations.drop(rated_items, errors='ignore')

    return items_recommendations.sort_values(ascending=False).head(n_recommendations)


def recommend_by_user2(user_id, train_matrix, user_similarity_df, n_recommendations=3):
    # Get the user's ratings from the train matrix
    user_ratings = train_matrix.loc[user_id]

    # Ensure the user_similarity_df is aligned with the users the current user is similar to
    aligned_user_similarity_df = user_similarity_df.loc[user_ratings.index, user_ratings.index]

    # Calculate recommendation scores using the dot product of the user similarity matrix and user ratings
    # We compute the weighted sum of ratings from similar users
    weighted_ratings = aligned_user_similarity_df.dot(user_ratings)  # Multiply similarity with user ratings

    # Normalize by dividing by the sum of similarity scores (to prevent bias from highly similar users)
    similarity_sum = aligned_user_similarity_df.sum(axis=1)
    scores = weighted_ratings / similarity_sum

    # Get the top n recommended items
    recommendations = scores.nlargest(n_recommendations)

    return recommendations


def recommend_items(user_id, user_item_matrix, item_similarity_df, n_recommendations=5):
    # User's ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Compute scores for all items
    scores = item_similarity_df.dot(user_ratings).div(item_similarity_df.sum(axis=1))

    # Exclude items already rated by the user
    rated_items = user_ratings[user_ratings > 0].index
    scores = scores.drop(rated_items, errors='ignore')

    # Return top N recommendations
    return scores.nlargest(n_recommendations)

def precision_recall_at_k_item(test_matrix, train_matrix, item_similarity_df, k=5):
    precision_list = []
    recall_list = []

    for user_id in test_matrix.index:
        if user_id in train_matrix.index:
            # Generate recommendations
            recommendations = recommend_items(user_id, train_matrix, item_similarity_df, k).index

            # Get the actual items the user interacted with in the test set
            relevant_items = test_matrix.loc[user_id][test_matrix.loc[user_id] > 0].index

            # Calculate precision and recall
            true_positives = len(set(recommendations) & set(relevant_items))
            precision = true_positives / k
            recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)

    # Average precision and recall across all users
    avg_precision = sum(precision_list) / len(precision_list)
    avg_recall = sum(recall_list) / len(recall_list)

    return avg_precision, avg_recall


def precision_recall_at_k_user(test_matrix, train_matrix, user_similarity_df, k=5):
    precision_list = []
    recall_list = []

    # Loop through each user in the test set
    for user_id in test_matrix.index:
        if user_id in train_matrix.index:
            # Generate top k recommendations for the user
            recommendations = recommend_items(user_id, train_matrix, user_similarity_df, k).index

            # Get the actual items the user interacted with in the test set (relevant items)
            relevant_items = test_matrix.loc[user_id][test_matrix.loc[user_id] > 0].index

            # Calculate the number of true positives (the overlap between recommended items and relevant items)
            true_positives = len(set(recommendations) & set(relevant_items))

            # Calculate precision and recall
            precision = true_positives / k  # Precision = TP / k
            recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0  # Recall = TP / relevant items

            # Append precision and recall to lists
            precision_list.append(precision)
            recall_list.append(recall)

    # Compute the average precision and recall across all users
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0

    return avg_precision, avg_recall




# User-based KNN
def recommend_by_user_knn(user_id, train_user_item_matrix, top_n=5):
    # Compute similarity matrix for users using Euclidean distance
    knn = NearestNeighbors(metric='euclidean', algorithm='brute')

    # Center ratings (subtract mean per user)
    user_ratings_centered = train_user_item_matrix.sub(train_user_item_matrix.mean(axis=1), axis=0)
    knn.fit(user_ratings_centered.fillna(0))  # Handle NaN values by filling with 0

    # Find nearest neighbors for the target user
    distances, indices = knn.kneighbors(train_user_item_matrix.iloc[user_id].to_numpy().reshape(1, -1),
                                        n_neighbors=top_n + 1)

    # Exclude the user itself from the neighbors
    similar_users = indices.flatten()[1:]

    recommendations = {}
    for neighbor in similar_users:
        neighbor_ratings = train_user_item_matrix.iloc[neighbor]
        for item, rating in enumerate(neighbor_ratings):
            if train_user_item_matrix.iloc[user_id, item] == 0:  # User hasn't rated this item
                recommendations[item] = recommendations.get(item, 0) + rating

    # Sort recommendations by score (descending)
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Format the output as a DataFrame
    recommended_items = pd.DataFrame(recommendations[:top_n], columns=['book_id', 'predicted_rating'])
    return recommended_items


# Item-based KNN
def recommend_by_item_knn(user_id, train_user_item_matrix, top_n=5):
    # Compute similarity matrix for items using Cosine similarity
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(train_user_item_matrix.T)  # Fit on transposed matrix (items as rows)

    # Get user's rated items
    user_ratings = train_user_item_matrix.loc[user_id]  # Access user ratings via .loc
    rated_items = [i for i, rating in enumerate(user_ratings) if rating > 0]

    recommendations = {}
    for item in rated_items:
        distances, indices = knn.kneighbors(train_user_item_matrix.T.iloc[item].to_numpy().reshape(1, -1),
                                            n_neighbors=top_n + 1)
        similar_items = indices.flatten()[1:]  # Exclude the item itself

        for similar_item in similar_items:
            item_column = train_user_item_matrix.columns[similar_item]  # Get the actual item column
            if train_user_item_matrix.loc[user_id, item_column] == 0:  # User hasn't rated this item
                recommendations[similar_item] = recommendations.get(similar_item, 0) + user_ratings.iloc[item]

    # Sort recommendations by score (descending)
    recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

    # Format the output as a DataFrame
    recommended_items = pd.DataFrame(recommendations[:top_n], columns=['book_id', 'predicted_rating'])
    return recommended_items

# --------------------------------------


# ==========================================

def sample_and_prepare_data(df, sample_size=10000, random_state=42):
    """
    Sample the data and prepare matrices while maintaining data integrity
    """
    # Sample the data
    sampled_df = df.sample(n=sample_size, random_state=random_state)
    sampled_df = sampled_df.reset_index(drop=True)

    # Create user_item_matrix
    user_item_matrix = sampled_df.pivot_table(
        index='user_id',
        columns='book_id',
        values='rating',
        fill_value=0
    )

    # Calculate similarities using cosine similarity
    user_similarity = cosine_similarity(user_item_matrix)
    item_similarity = cosine_similarity(user_item_matrix.T)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    return sampled_df, user_item_matrix, user_similarity, item_similarity_df


def recommend_by_user(user_id, user_item_matrix, similarity_matrix, n_recommendations=5):
    """
    Generate recommendations for a user based on user similarity with weighted scoring
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User ID {user_id} not found in user-item matrix")

    # Get user's position
    user_position = user_item_matrix.index.get_loc(user_id)

    # Get similar users
    similar_users = similarity_matrix[user_position]

    # Calculate weighted scores
    weighted_scores = np.dot(similar_users, user_item_matrix)

    # Create recommendations series
    items_recommendations = pd.Series(weighted_scores, index=user_item_matrix.columns)

    # Remove rated items
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    items_recommendations = items_recommendations.drop(rated_items, errors='ignore')

    # Apply score normalization to get similar range as original
    max_score = items_recommendations.max()
    if max_score > 0:
        items_recommendations = items_recommendations * (30 / max_score)

    return items_recommendations.sort_values(ascending=False).head(n_recommendations)


def recommend_by_item(user_id, user_item_matrix, item_similarity_df, n_recommendations=5):
    """
    Generate recommendations for a user based on item similarity with weighted scoring
    """
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User ID {user_id} not found in user-item matrix")

    # Get user ratings
    user_ratings = user_item_matrix.loc[user_id]

    # Calculate weighted scores
    weighted_scores = item_similarity_df.dot(user_ratings)
    items_scores = pd.Series(weighted_scores)

    # Remove rated items
    rated_items = user_ratings[user_ratings > 0].index
    items_scores = items_scores.drop(rated_items, errors='ignore')

    # Normalize scores to similar range as original
    max_score = items_scores.max()
    if max_score > 0:
        items_scores = items_scores * (4 / max_score)

    return items_scores.values.sort_values(ascending=False).head(n_recommendations)

# Usage example
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"

    df = pd.read_csv(path)
    df_sample = df.sample(frac=0.005, random_state=42).reset_index(drop=True)
    train_data, test_data = train_test_split(df_sample, test_size=0.2, random_state=42)


