import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score
from scipy.sparse import csr_matrix
from utils.recomend_extended import  (recommend_by_user2, recommend_items, precision_recall_at_k_user,
                                      precision_recall_at_k_item ,recommend_by_item_knn, recommend_by_user_knn)
from utils.recmd_2 import *
# Load data
path2 = "https://drive.google.com/uc?id=1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy"
path = r"C:\1.PIGGEST\06_Course\06_Advanced-Mining-Algorithms\data_06\กรุณาดาวน์โหลด Data Set ประกอบการเรียน\ratings.csv"
# df = pd.read_csv(path)
# print(df.columns.values)
# print(df.shape)
# df = df.sample(frac=0.005, random_state=42).reset_index(drop=True)

# # Train-Test Split
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
#
# # Create user-item matrices
# train_user_item_matrix = train_data.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
# test_user_item_matrix = test_data.pivot_table(index='user_id', columns='book_id', values='rating').fillna(0)
#
# # Create matrices
# user_item_matrix, sparse_matrix = create_user_item_matrix(df)
#
# # Compute similarity
# user_similarity_df = compute_user_similarity(sparse_matrix, user_item_matrix)

# Get recommendations for a user
run_recommendation_system(path)