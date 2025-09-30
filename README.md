Yiqi Zhou  
yiqizhou@usc.edu  
Generate embeddings with a BoW(Bag of Words)-based shallow autoencoder. 
Abstracts are mapped to a top-K vocabulary K = 5000, 
then encoded as binary BoW vectors. 
The model is vocab_size - hidden_dim - embed_dim - 
hidden_dim - vocab_size with ReLU hidden layers, 
and Sigmoid output. Hidden_dim=128, 
embed_dim=64 and total parameters = 1.3M, 
smaller than 2M.