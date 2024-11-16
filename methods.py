from sklearn.cluster import KMeans , DBSCAN , AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



def choose_clustering(algorithm, X, y_true):
    if algorithm == "kmeans":
        model = KMeans(n_clusters=7, random_state=0)
    elif algorithm == "agglomerative":
        model = AgglomerativeClustering(n_clusters=7)
    elif algorithm == "dbscan":
        model = DBSCAN(eps=3, min_samples=5)
    else:
        print("Algorithm not supported")
    
    labels = model.fit_predict(X)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return labels, ari, nmi


def evaluate_algorithms(X, y_true):
    evaluation = {}
    for algo in ["kmeans", "agglomerative", "dbscan"]:
        labels, ari, nmi = choose_clustering(algo, X, y_true)
        evaluation[algo] = {"ARI": ari, "NMI": nmi, "Labels": labels}
        print(algo + " - ARI: " + str(round(ari, 3)) + ", NMI: " + str(round(nmi, 3)))


def scatter_plotting(df_scaled , y_true):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)
    algorithms = ["kmeans", "agglomerative", "dbscan"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for i, algo in enumerate(algorithms):
        labels , ari , nmi = choose_clustering(algo , df_scaled , y_true )

        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="tab10", ax=axes[i], legend=None)
        axes[i].set_title(f"{algo.capitalize()} Clustering")
        axes[i].set_xlabel("PCA Component 1")
        axes[i].set_ylabel("PCA Component 2")

    plt.suptitle("Cluster Visualization for Each Algorithm")
    plt.show()


def algo_characteristics(df, algo):
    df['Cluster'] = algo[0]

    cluster_summary = df.groupby('Cluster').mean()

    return cluster_summary

