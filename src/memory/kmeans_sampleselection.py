from sklearn.cluster import KMeans
import numpy as np

#send data emmbeddings according to relation types.
def sample_selection_kmeans(data, memory_size):
    # Extract the embeddings and convert them into a 2D numpy array
    embeddings_array = [item['embedding'] for item in data]
    prompt_array = [item['prompt'] for item in data]

    # Get the minimum embedding dimension across all embeddings
    min_dim = min(emb.shape[1] for emb in embeddings_array)  
    
    # Reshape each embedding individually and then stack them vertically
    # Truncate or pad embeddings to ensure consistent size
    reshaped_embeddings = [emb[:, :min_dim].reshape(-1) for emb in embeddings_array]  

    embeddings_array = np.vstack(reshaped_embeddings)

    # Fit the KMeans model with the reshaped array
    num_clusters = min(memory_size, len(embeddings_array))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(embeddings_array)

    mem_set = [] # Initialize mem_set as a list
    prompt_set = []
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        instance = embeddings_array[sel_index]
        mem_set.append(instance)
        prompt = prompt_array[sel_index]
        prompt_set.append(prompt)

    mem_set = np.array(mem_set) 
    prompt_set = np.array(prompt_set)
    
    return mem_set, prompt_set