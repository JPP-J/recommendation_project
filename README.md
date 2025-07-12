# Recommendation System Project

![Last Commit](https://img.shields.io/github/last-commit/JPP-J/recommendation_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-97.6%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/recommendation_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Recommendation System Project* , which provides; 

## 📌 Overview – Book Recommendation System

This project builds a recommendation system to suggest **book items** to users based on user-item rating interactions using both local and large-scale implementations.

### 🧩 Problem Statement
In platforms with thousands of books and users, helping readers discover relevant books is a challenge. This project aims to provide personalized book recommendations by analyzing historical ratings to detect user preferences and reading patterns.

### 🔍 Approach

Two recommendation engines were developed:

1. **Local System:** Collaborative Filtering (CF) with similarity-based methods (user/item-based, KNN)
2. **Scalable System:** Alternating Least Squares (ALS) using **Spark MLlib** and Hadoop for distributed data handling

Both systems generate top-N recommendations, evaluated via **precision@k** to measure relevance.

### 🎢 Workflow

1. **ETL** – Load & clean user-book interaction data (Approach 1 113,518 entries → sampled to ~90,814 &  Approach 2 ~ 5.9M records)
2. **EDA** – Analyze rating distribution and user/item activity levels
3. **Sparse Matrix Conversion** – Transform into CSR format for efficient similarity search
4. **Modeling Approach 1 - [`main.py`](main.py)**  
   - User-based CF  
   - Item-based CF  
   - User KNN  
   - Item KNN
   - Libraries Used: `pandas`, `NumPy`, `scikit-learn`, `scipy`
5. **Modeling Approach 2 - [`main2.py`](main2.py) using Spark**  
   - ALS via Spark MLlib on ~6 million records from HDFS  
   - Parallelized training & evaluation
   - Libraries Used: `Hadoop`, `pyspark`, `Apache Spark MLlib (ALS model)`
6. **Evaluation** – Use `precision@k` at k = 5, 10, 15
7. **Result Comparison** – Analyze local vs. distributed system performance

### 🎯 Results
- Demo example result: [Example output](Example_result.txt) 

#### 📊 Sample Output – Local System - Approach 1

- Recommendations for `user_id = 172`
  - User-based CF: `[2857, 2988, 3151, ...]`
  - Item-based CF: `[202, 1398, 1667, ...]`
  - User KNN: `[110, 1, 25, ...]`
  - Item KNN: `[8159, 4, 5, ...]`
- Evaluation Metrics

  | Metric               | Method | Target | Precision |
  |----------------------|--------|--------|-----------|
  | precision@5_cf_user  | CF     | User   | 0.0004    |
  | precision@5_cf_item  | CF     | Item   | 0.0004    |
  | precision@5_knn_user | KNN    | User   | **0.1524** |
  | precision@5_knn_item | KNN    | Item   | 0.0148    |
  | precision@10_cf_user | CF     | User   | 0.0004    |
  | precision@10_cf_item | CF     | Item   | 0.0003    |
  | precision@10_knn_user| KNN    | User   | **0.1314** |
  | precision@10_knn_item| KNN    | Item   | 0.0130    |
  | precision@15_cf_user | CF     | User   | 0.0005    |
  | precision@15_cf_item | CF     | Item   | 0.0003    |
  | precision@15_knn_user| KNN    | User   | **0.1153** |
  | precision@15_knn_item| KNN    | Item   | 0.0122    |

#### 📊 Sample Output – Spark ALS (5.9M records) - Approach 2

- Top-N recommendations per user and item ID using ALS
- Example predictions (user 28):  
  `[{3628, 4.94}, {7947, 4.89}, ..., {8978, 4.83}]`
- True items vs. predicted items joined and evaluated

    | User ID | precision@10 |
    |---------|-------------|
    | 28      | 0.0         |
    | 34      | 0.1         |


> Note: ALS model is strong in prediction ranking but struggles in sparse test ground-truth overlap, affecting precision@k directly.

### ⚙️ Key Development Challenges

| Area                    | Challenge                                                       | Resolution / Notes                                  |
|-------------------------|------------------------------------------------------------------|-----------------------------------------------------|
| Data Sparsity           | Rating matrix highly sparse                                     | Filter low-activity users/items                     |
| Cold Start Problem      | Hard to recommend for new users/items                          | Future work includes hybrid or metadata integration |
| Evaluation Complexity   | Lack of direct labels for ranking tasks                        | Used sampled evaluation with `precision@k`          |
| Performance at Scale    | Large dataset (>5M) caused memory issues                       | Solved using distributed Spark & Parquet            |
| Model Interpretability  | ALS is less interpretable than similarity models               | Supplemented with KNN & CF for transparency         |

---




## Recommendation System
- **Description:**
Recommendation System of book items for users – hands-on Python demo in [`main.py`](main.py) for Collaborative Filtering (CF) on a [book-ratings dataset](https://drive.google.com/file/d/1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy/view?usp=drive_link) featuring:  
  - User-based CF recommendations
  - Item-based CF recommendations
  - User KNN recommendations
  - Item KNN recommendations
- Libraries Used:
  - Data Analysis: `pandas`, `NumPy`
  - Machine Learning: `scikit-learn`, `scipy`
  - Model Evaluation: precision@k
- [Example output](Example_result.txt) demo result examples 

## Recommendation System with Spark MLlib
- Description: Recommendation System of book items for user - improvment from above in case large dataset with this case have dataset around 5 million records so using spark to handle this problem, hand on python code demo in [main2.py](main2.py) for Recommendation System update
- Libraries Used:
  - Data Handling: `Hadoop`, `pyspark`  
  - Machine Learning: `Apache Spark MLlib (ALS model)`
  - Model Evaluation: precision@k
- [Example output](Example_result.txt) demo result examples 
 
> Looking for a movie-based example? Check out the follow-up repo:  
> 🎬 [**Recommendation System Project No.2**](https://github.com/JPP-J/reccomd_project2) – movie dataset + LLM integration for advanced user queries.





---

