from sklearn.cluster import KMeans , DBSCAN , AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def choose_clustering(algorithm, X, y_true):

    
    if algorithm == "kmeans":
        model = KMeans(n_clusters=7, random_state=0)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=7)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=3, min_samples=5)
    else:
        raise ValueError("Unsupported algorithm selected.")

    
    labels = model.fit_predict(X)

    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return labels, ari, nmi


def evaluate_algorithms(X, y_true):
    
    evaluation = {}

 
    for algo in ["kmeans", "agglomerative", "dbscan"]:
        labels, ari, nmi = choose_clustering(algo, X, y_true)
        evaluation[algo] = {"ARI": ari, "NMI": nmi}

        print(algo + " - ARI: " + str(round(ari, 3)) + ", NMI: " + str(round(nmi, 3)))

def scatter_plotting(alg ,df_scaled , y_true):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)
    #algorithms = ["kmeans", "agglomerative", "dbscan"]

    fig, axes = plt.subplots(1, figsize=(10, 5), sharey=True)
    
    
    labels , ari , nmi = choose_clustering(alg , df_scaled , y_true)

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", ax=axes, legend=None)
    axes.set_title(f"{alg.capitalize()} Clustering")
    axes.set_xlabel("PCA Component 1")
    axes.set_ylabel("PCA Component 2")

    plt.suptitle("Cluster Visualization")
    return fig


def algo_characteristics(df, algo):
    df['Cluster'] = algo[0]

    cluster_summary = df.groupby('Cluster').mean()

    return cluster_summary


def animals_list_forAlgorithm(algo , df):
    df["cluster"] = algo[0]
    animal_names_per_cluster = df.groupby('cluster')['animal_name'].apply(list)

    for cluster, animal_names in animal_names_per_cluster.items():
        print(f"Cluster {cluster}:")
        print(", ".join(animal_names))


