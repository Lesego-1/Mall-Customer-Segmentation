# Clustering and Segmentation on Mall Customers dataset

## Problem Statement
You own the mall and want to understand the customers like who can be easily converge (buy more products) so that the sense can be given to marketing team and plan the strategy accordingly.

## Dataset
The dataset used is the Mall Customer Segmentation Data from Kaggle.
Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?resource=download

## Process
- Creating X to store variables to use for clustering
- Finding the optimal amount of clusters using Elbow Plot
- Creating Models (K-Means, DBSCAN, Hierarchical/Agglomerative)
- Visualizing the models' results
- Comparing silhouette scores for clustering accuracy

## Results
### Clustering Performance
- **K-Means Clustering** : This model created distinct clusters, clearly separating the clusters. Achieved a silhouette score of 0.55.
- **DBSCAN Model** : This model had poor clustering, not being able to separate segments clearly with some overlapping. Achieve a silhouette score of 0.04.
- **Agglomerative Clustering** : This model performed equally as good as the K-Means mode. Achieved a silhouette score of 0.55.

## Conclusion
The clustering models effectively identified meaningful customer segments. The results suggest that certain customers with different amounts of annual incomes spend more than others. Only the customers with an annual income ranging from 40k to 70k have a consistent amount of spending score. Customers with high spending scores in both high and low income levels can be recommended special bonuses to maintain their spending scores, while those with lower can be offered discounts to try get them to spend more. The chosen models, especially the K-Means and Hierarchical Clustering, provided the most reliable clusters, as evidenced by the high Silhouette Scores and clear visual separation.