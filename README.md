# GNNFraud

GNN using [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric).

GraphBuilder.py builds the HeteroData object to create the heterogeneous graph.
GraphTrainer.py will train on datasets.
EnhancedGATModel.py contains the model.  It was designed and trained original for this project.

This was unfortunately done on data of unknown quality.  
It was known to be synthentic but it's methodology or trainable quality was never demonstrated of expressed.

Both this and FastAI FNN had similar characteristics in training in that both models started training well and then would overtrain very quickly.
