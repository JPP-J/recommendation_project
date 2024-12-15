import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict



def create_user_item_matrix(data, user_col='user_id', item_col='book_id', rating_col='rating'):
    """
    Create a user-item matrix from raw data.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing user-item interactions
    user_col : str
        Name of the user column
    item_col : str
        Name of the item column
    rating_col : str
        Name of the rating column

    Returns:
    --------
    tuple : (pandas.DataFrame, scipy.sparse.csr_matrix)
        User-item matrix in both dense and sparse formats
    """
    # Create the pivot table
    user_item_matrix = data.pivot_table(
        index=user_col,
        columns=item_col,
        values=rating_col
    ).fillna(0)

    # Convert to sparse matrix
    sparse_matrix = csr_matrix(user_item_matrix.values)

    return user_item_matrix, sparse_matrix


def compute_user_similarity(sparse_matrix, user_item_matrix):
    """
    Compute user similarity matrix using cosine similarity.

    Parameters:
    -----------
    sparse_matrix : scipy.sparse.csr_matrix
        Sparse user-item matrix
    user_item_matrix : pandas.DataFrame
        Dense user-item matrix for index information

    Returns:
    --------
    pandas.DataFrame
        User similarity matrix
    """
    similarities = cosine_similarity(sparse_matrix)
    return pd.DataFrame(
        similarities,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )


def recommend_by_user(user_id, train_matrix, user_similarity_df, n_recommendations=10,
                      min_similarity=0.0, already_rated_penalty=True):
    """
    Enhanced user-based collaborative filtering recommendation function.

    Parameters:
    -----------
    user_id : int or str
        The ID of the user to generate recommendations for
    train_matrix : pandas.DataFrame
        User-item interaction matrix
    user_similarity_df : pandas.DataFrame
        User similarity matrix
    n_recommendations : int
        Number of recommendations to generate
    min_similarity : float
        Minimum similarity threshold for considering similar users
    already_rated_penalty : bool
        Whether to penalize items already rated by the user

    Returns:
    --------
    pandas.Series
        Recommended items with their scores
    """
    # Get user's ratings and similarities
    user_ratings = train_matrix.loc[user_id]
    user_similarities = user_similarity_df.loc[user_id]

    # Filter out users with low similarity
    similar_users = user_similarities[user_similarities > min_similarity]

    # Calculate weighted ratings
    weighted_sum = pd.DataFrame(0,
                                index=train_matrix.columns,
                                columns=['score'])
    similarity_sum = pd.Series(0, index=train_matrix.columns)

    for similar_user_id, similarity in similar_users.items():
        if similar_user_id != user_id:  # Exclude the user themselves
            similar_user_ratings = train_matrix.loc[similar_user_id]
            weighted_sum['score'] += similarity * similar_user_ratings
            similarity_sum += np.abs(similarity) * (similar_user_ratings != 0)

    # Normalize scores
    recommendations = weighted_sum['score'] / (similarity_sum + 1e-8)

    # Apply penalty to already rated items if requested
    if already_rated_penalty:
        rated_mask = user_ratings > 0
        recommendations[rated_mask] *= 0.1  # Reduce scores for already rated items

    return recommendations.nlargest(n_recommendations)


def evaluate_recommendations(test_matrix, train_matrix, user_similarity_df, k_values=[5, 10],
                             min_similarity=0.0):
    """
    Comprehensive evaluation of the recommendation system.

    Parameters:
    -----------
    test_matrix : pandas.DataFrame
        Test set user-item matrix
    train_matrix : pandas.DataFrame
        Training set user-item matrix
    user_similarity_df : pandas.DataFrame
        User similarity matrix
    k_values : list
        List of k values to evaluate
    min_similarity : float
        Minimum similarity threshold

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    metrics = {}

    for k in k_values:
        precision_list = []
        recall_list = []
        ndcg_list = []

        for user_id in test_matrix.index:
            if user_id in train_matrix.index:
                # Generate recommendations
                recommendations = recommend_by_user(
                    user_id,
                    train_matrix,
                    user_similarity_df,
                    n_recommendations=k,
                    min_similarity=min_similarity
                ).index

                # Get relevant items from test set
                relevant_items = test_matrix.loc[user_id][
                    test_matrix.loc[user_id] > 0
                    ].index

                # Calculate metrics
                true_positives = len(set(recommendations) & set(relevant_items))
                precision = true_positives / k
                recall = (true_positives / len(relevant_items)
                          if len(relevant_items) > 0 else 0)

                # Calculate NDCG
                dcg = 0
                idcg = 0
                for i, item in enumerate(recommendations):
                    rel = 1 if item in relevant_items else 0
                    dcg += rel / np.log2(i + 2)
                for i in range(min(k, len(relevant_items))):
                    idcg += 1 / np.log2(i + 2)
                ndcg = dcg / idcg if idcg > 0 else 0

                precision_list.append(precision)
                recall_list.append(recall)
                ndcg_list.append(ndcg)

        metrics[f'precision@{k}'] = np.mean(precision_list)
        metrics[f'recall@{k}'] = np.mean(recall_list)
        metrics[f'ndcg@{k}'] = np.mean(ndcg_list)

    return metrics



def load_and_prepare_data(path, sample_frac=0.005, test_size=0.2 ,random_state=42):
    """
    Load and prepare the dataset with proper error handling.
    """
    # Load data
    df = pd.read_csv(path)
    print(f"Original data shape: {df.shape}")
    print(f"Columns: {df.columns.values}")

    # Sample the data
    df_sampled = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
    print(f"Sampled data shape: {df_sampled.shape}")

    # Train-Test Split
    train_data, test_data = train_test_split(df_sampled, test_size=test_size, random_state=random_state)

    # Create user-item matrices
    train_matrix = train_data.pivot_table(
        index='user_id',
        columns='book_id',
        values='rating'
    ).fillna(0)

    test_matrix = test_data.pivot_table(
        index='user_id',
        columns='book_id',
        values='rating'
    ).fillna(0)

    print(f"\nNumber of users in training set: {len(train_matrix.index)}")
    print(f"Number of books in training set: {len(train_matrix.columns)}")

    return train_matrix, test_matrix, df_sampled


def get_random_user(train_matrix, n=5):
    """
    Get n random user IDs from the training matrix.
    """
    random_users = np.random.choice(train_matrix.index, size=min(n, len(train_matrix.index)), replace=False)
    return random_users


def run_recommendation_system(path):
    """
    Run the complete recommendation system with error handling.
    """
    try:
        # Load and prepare data
        train_matrix, test_matrix, df_sampled = load_and_prepare_data(path)

        # Create sparse matrix for training data
        sparse_train_matrix = csr_matrix(train_matrix.values)

        # Compute user similarity
        user_similarity = cosine_similarity(sparse_train_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=train_matrix.index,
            columns=train_matrix.index
        )

        # Get random users for demonstration
        random_users = get_random_user(train_matrix)
        print("\nRandom users selected for demonstration:", random_users)

        # Generate recommendations for random users
        for user_id in random_users:
            print(f"\nGenerating recommendations for user_id: {user_id}")
            recommendations = recommend_by_user(
                user_id=user_id,
                train_matrix=train_matrix,
                user_similarity_df=user_similarity_df,
                n_recommendations=5,
                min_similarity=0.1
            )
            print(f"Top 5 recommended books: {recommendations.index.tolist()}")

        # Evaluate the system
        metrics = evaluate_recommendations(
            test_matrix=test_matrix,
            train_matrix=train_matrix,
            user_similarity_df=user_similarity_df,
            k_values=[5, 10]
        )

        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

# Usage
if __name__ == "__main__":
    path = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"
    run_recommendation_system(path)
