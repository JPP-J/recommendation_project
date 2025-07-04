# Recommendation System Project

![Last Commit](https://img.shields.io/github/last-commit/JPP-J/recommendation_project?style=flat-square)
![Python](https://img.shields.io/badge/Python-97.6%25-blue?style=flat-square)
![Languages](https://img.shields.io/github/languages/count/JPP-J/recommendation_project?style=flat-square)

This repo is home to the code that accompanies Jidapa's *Recommendation System Project* , which provides; 
## Recommendation System
- Description: Recommendation System of book items for user - hand on python code demo in [main.py](main.py) for Recommendation System with Collaborative Filtering (CF) on [book items rating dataset](https://drive.google.com/file/d/1HDPOyxM6cs1SDx4boqKGrRVQam1VEPfy/view?usp=drive_link) with
  - User-based CF recommendations
  - Item-based CF recommendations
  - User KNN recommendations
  - Item KNN recommendations
- Libraries Used:
  - Data Analysis: `pandas`, `NumPy`
  - Machine Learning: `scikit-learn`
  - Model Evaluation: precision@k
- [Example result](Example_result.txt) demo result examples 

## Recommendation System with Spark MLlib
- Description: Recommendation System of book items for user - improvment from above in case large dataset with this case have dataset around 5 million records so using spark to handle this problem, hand on python code demo in [main2.py](main2.py) for Recommendation System update
- Libraries Used:
  - Data Handling: `Hadoop`, `pyspark`  
  - Machine Learning: `Apache Spark MLlib (ALS model)`
  - Model Evaluation: precision@k
- [Example result](Example_result.txt) demo result examples 
 
> [Recommendation System Project No.2 repo](https://github.com/JPP-J/reccomd_project2) with movie dataset and integrate with LLM model for additional query from user

