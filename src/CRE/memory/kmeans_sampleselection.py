
import os
import json
from sklearn.cluster import KMeans
import numpy as np


def read_json(path):
    """ Read a json file from the given path."""
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, path):
    """ Write a json file to the given path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    with open(path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
def compute_embedding(model, tokenizer, input_path):

  data = read_json(input_path)

  embeddings = []
  for i, item in enumerate(data):
    prompt = item['prompt']
    relation  = item['relation']
    inputs = tokenizer(prompt, add_special_tokens=True, max_length=4096,return_tensors="pt").input_ids
    embedding = model.encoder(inputs)

    embeddings.append({"prompt": prompt, "relation":relation, "embedding": embedding['last_hidden_state'].data.numpy()})
  return embeddings

def select_samples(model, tokenizer, memory_size, train_data_path, tasks_path):
  
  relations = read_json(tasks_path)

  embeddings = compute_embedding(model, tokenizer, train_data_path)
  r1_emb = [ emb for emb in embeddings if emb['relation'] == relations[-1]]
  r2_emb = [ emb for emb in embeddings if emb['relation'] == relations[-2]]
  r3_emb = [ emb for emb in embeddings if emb['relation'] == relations[-3]]
  r4_emb = [ emb for emb in embeddings if emb['relation'] == relations[-4]]
  mem1, _, _,selected_samples1 = sample_selection_kmeans(r1_emb, 20)
  mem2, _, _, selected_samples2 = sample_selection_kmeans(r2_emb, 20)
  mem3, _, _, selected_samples3 = sample_selection_kmeans(r3_emb, 20)
  mem4, _, _, selected_samples4 = sample_selection_kmeans(r4_emb, 20)
  all_selected_samples = []
  all_selected_samples.extend(selected_samples1)
  all_selected_samples.extend(selected_samples2)
  all_selected_samples.extend(selected_samples3)
  all_selected_samples.extend(selected_samples4)
  all_mem, saved_mem = [],[]
  all_mem.extend(mem1)
  all_mem.extend(mem2)
  all_mem.extend(mem3)
  all_mem.extend(mem4)

  for i, mem in enumerate(all_mem):
     saved_mem.append({"prompt":all_selected_samples[i]['prompt'], "embedding":mem})

  if os.path.exists('selected_samples.npy'):
    old_mem = np.load('selected_samples.npy', allow_pickle=True)
    saved_mem.extend(old_mem)
  np.save('selected_samples.npy', saved_mem)
  return all_selected_samples

#send data emmbeddings according to relation types.
def sample_selection_kmeans(data, memory_size):
    # Extract the embeddings and convert them into a 2D numpy array
    embeddings_array = [item['embedding'] for item in data]
    prompt_array = [item['prompt'] for item in data]
    relation_array = [item['relation'] for item in data]
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
    relation_set = []
    selected_samples = []
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        instance = embeddings_array[sel_index]
        mem_set.append(instance)
        prompt = prompt_array[sel_index]
        prompt_set.append(prompt)
        relation_set.append(relation_array[sel_index])
        selected_samples.append({"prompt":prompt,"relation":relation_array[sel_index]})

    mem_set = np.array(mem_set) 
    prompt_set = np.array(prompt_set)
    relation_set = np.array(relation_set)

    return mem_set, prompt_set, relation_set, selected_samples