import streamlit as st
import pandas as pd
from methods import choose_clustering ,evaluate_algorithms ,evaluate_algorithms , scatter_plotting , algo_characteristics ,animals_list_forAlgorithm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


zoo_data = pd.read_csv("zoo.csv")
class_data = pd.read_csv("class.csv")

features = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 
            'predator', 'toothed', 'backbone', 'breathes', 'venomous', 
            'fins', 'legs', 'tail', 'domestic', 'catsize']

true_class = zoo_data["class_type"]

st.title("Animal Clustering App")
st.sidebar.title("Choose Clustering Algorithm")

algorithm = st.sidebar.selectbox("Algorithm", ["kmeans", "dbscan", "agglomerative"])

df = zoo_data.iloc[:, 1:-1]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

if algorithm == "kmeans":
    labels , ari, nmi = choose_clustering(algorithm, df_scaled, true_class)
    st.write(algorithm + " - ARI: " + str(round(ari, 3)) + "\nNMI: " + str(round(nmi, 3)))

elif algorithm == "dbscan":
    labels , ari, nmi= choose_clustering(algorithm, df_scaled, true_class)
    st.write(algorithm + " - ARI: " + str(round(ari, 3)) + "\nNMI: " + str(round(nmi, 3)))

elif algorithm == "agglomerative":
    labels,  ari, nmi = choose_clustering(algorithm, df_scaled,true_class)
    st.write(algorithm + " - ARI: " + str(round(ari, 3)) + "\nNMI: " + str(round(nmi, 3)))




st.write("**Cluster Plot:**")

if algorithm == "kmeans":
    fig = scatter_plotting(algorithm, df_scaled, true_class)
elif algorithm == "dbscan":
    fig = scatter_plotting(algorithm, df_scaled,true_class)
elif algorithm == "agglomerative":
    fig = scatter_plotting(algorithm, df_scaled,true_class)

st.pyplot(fig)

st.write("**Animals in Each Cluster:**")
for cluster_id in sorted(set(labels)):
    cluster_animals = zoo_data['animal_name'][labels == cluster_id]
    st.write(f"Cluster {cluster_id}:\n{', '.join(cluster_animals)}")


