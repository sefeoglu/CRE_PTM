from sklearn.cluster import KMeans
import numpy as np

#send data emmbeddings according to relation types.
def sample_selection_kmeans(data, memory_size):
    # Extract the embeddings and convert them into a 2D numpy array
    embeddings_array = np.array([item['embedding'] for item in data])
    embeddings_array = embeddings_array.reshape(len(embeddings_array), -1)

    # Fit the KMeans model with the reshaped array

    num_clusters = min(memory_size, len(embeddings_array))

    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(embeddings_array)

    mem_set = []

    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        instance = embeddings_array[sel_index]
        mem_set.append(instance)

        mem_set = np.array(mem_set)
    return mem_set