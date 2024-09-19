import torch_geometric.transforms as T
import torch

class GraphTrainer:
    def __init__(self, device):
        self.fraud = torch.tensor([[1., 0.]], dtype=torch.float)
        self.legit = torch.tensor([[0., 1.]], dtype=torch.float)
        self.fraud=self.fraud.to(device)
        self.legit=self.legit.to(device)

    @staticmethod
    def train(model, file_list):
        model.train()
        total_loss = 0
        count=0
        total_true=0
        total_false=0    
        
        #for graph in dataloader:
        for file in file_list:
            optimizer.zero_grad()
            count+=1
            try:
                graph=LoadGraph(file)            
                #graph = T.ToUndirected()(graph)
                graph = graph.to(device)
                graph = T.NormalizeFeatures()(graph)
                try:
                    out = model(graph.x_dict, graph.edge_index_dict)
                except Exception as e:
                    print(f"An error occurred: {e}. File: {file}")
                    continue
                
                #details.append([file, graph.y, out])
            except Exception as e:
                print(f'\nFile:{file}, error {e}')
                raise
                continue
                #set_trace()            
            else:
                target = self.fraud if graph.y==1 else self.legit            
                loss = F.binary_cross_entropy_with_logits(out, target)
                if train_show:
                    print((out,target,loss))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if count%1000 == 0:
                print(".", end="")
    
            del graph
        gc.collect()
                
        return total_loss / len(file_list)

    @staticmethod
    def validate(model, valid_data):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        avg_total_loss = 0.0
        true_positive, true_negative, false_positive, false_negative = 0,0,0,0
        
        with torch.no_grad():
            for file_path in valid_data:
                graph = LoadGraph(file_path)
                graph = graph.to(device)
    
                try:
                    try:
                        out = model(graph.x_dict, graph.edge_index_dict)                    
                    except Exception as e:
                        print(f"An error occurred: {e}. File: {file}")
                        continue
                    
                    target = self.fraud if graph.y==1 else self.legit
                    loss = F.binary_cross_entropy_with_logits(out, target)
                    #print((out,target,loss))
                    
                except Exception as e:
                    print(f'\nFile:{file_path}, error {e}')
                    continue;
                    set_trace()
    
                else:
                    total_loss += loss.item()
                    total += target.size(0)                
                    
                    predicted = torch.argmax(out, dim=1)
                    actual = torch.argmax(target, dim=1)
                    #print(f'out={out}, target={target}, predicted={predicted}, actual={actual}')
                    true_positive += ((predicted == 0) & (actual == 0)).sum().item()
                    true_negative += ((predicted == 1) & (actual == 1)).sum().item()
                    false_positive += ((predicted == 0) & (actual == 1)).sum().item()
                    false_negative += ((predicted == 1) & (actual == 0)).sum().item()
    
                    correct += (predicted == actual).sum().item()
                    
                if total%1000 == 0:
                    print(".", end="")
                del graph
            gc.collect()
    
        results_dict = {
            "True Positives": true_positive,
            "True Negatives": true_negative,
            "False Positives": false_positive,
            "False Negatives": false_negative
        }
    
        accuracy = correct / total
        average_loss = total_loss / total
        return accuracy, average_loss, results_dict

    def train_loop(model, train_files, valid_files, graph_builder, epochs=1, ):
        for epoch in range(epochs):
            print("\nTrain")    
            train_loss = train(model, train_files)    
            print("\nValidate")
            valid_acc, valid_loss, valid_details = validate(model, valid_files)
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Validation Accuracy: {valid_acc:.4f}, Validation Loss: {valid_loss:.4f}')
            print(valid_details)
        
            #torch.save(model.state_dict(), f'models/{run_id}_{epoch}_{train_loss}_{valid_acc}_{valid_loss}_enhanced_gat_model_state.pth')
            #np.save(f'models/{run_id}_{epoch}_confusion.npy', valid_details)
            #del train_details
            del valid_details
            gc.collect()
        