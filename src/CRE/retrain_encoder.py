import torch
def train_encoder(data, model, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Assuming 'batch['prompt']' contains the text for which you want to get embeddings
    for epoch in range(10):
        print("Epoch: {0}".format(epoch))
        for batch in data:
            optimizer.zero_grad()
            
            # Tokenize the prompt to get input IDs
            inputs = tokenizer(batch['prompt'], return_tensors="pt").input_ids.to("cuda")

            # Forward pass
            outputs = model.get_encoder()(inputs) 
        
            # Convert NumPy array to PyTorch tensor and requires_grad to True
            # Convert target embeddings to float16 to match model's dtype
            target_embedding = torch.tensor(batch['embedding'], dtype=torch.float16, requires_grad=True).to("cuda") 

            # Instead of reshaping, select the appropriate slice of the output tensor
            # based on the length of the original input for which the target embedding was created.
            # Assuming original embedding is for the first 2 tokens:
            output_slice = outputs.last_hidden_state[:, :target_embedding.shape[1], :]  

            # ... Calculate loss based on outputs and target (This part needs to be defined) ...
            # Example: Assuming you have a target tensor 'target_tensor'
            loss = torch.nn.MSELoss()(output_slice, target_embedding) 

            
            loss.backward()  # Replace with your actual loss calculation
            optimizer.step()
            print("batch loss: {0}".format(loss))