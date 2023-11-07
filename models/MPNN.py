import torch
import torch.nn as nn


class Message_Passer_NNM(nn.Module):
    def __init__(self, node_dim, nb_features):
        super(Message_Passer_NNM, self).__init__()
        self.nb_features = nb_features
        self.node_dim = node_dim
        self.nn = nn.Sequential(
                    nn.Linear(in_features=self.nb_features, out_features=self.node_dim**2),
                    nn.ReLU()
                )
      
    def forward(self, node_j, edge_ij):
        # Embed the edge as a matrix
        A = self.nn(edge_ij)
        
        # Reshape so matrix mult can be done
        A = A.view(-1, self.node_dim, self.node_dim)
        node_j = node_j.view(-1, self.node_dim, 1)
        
        # Multiply edge matrix by node and shape into message list
        messages = torch.matmul(A, node_j)
        messages = messages.view(-1, edge_ij.shape[1], self.node_dim)

        return messages
    
    
class Message_Agg(nn.Module):
    def __init__(self):
        super(Message_Agg, self).__init__()
    
    def forward(self, messages):
        return torch.sum(messages, 2)


class Update_Func_GRU(nn.Module):
    def __init__(self, state_dim):
        super(Update_Func_GRU, self).__init__()
        self.state_dim = state_dim
        self.GRU = nn.GRU(input_size=state_dim, hidden_size=state_dim, batch_first=True)
        
    def forward(self, old_state, agg_messages):
        # Concat so old_state and messages are in sequence
        n_nodes = old_state.shape[1]
        
        concat = torch.cat([old_state.view(-1, 1, self.state_dim), agg_messages.view(-1, 1, self.state_dim)], dim=1)
        # Concat size 29*batch_size, 2, state_dim
        
        # Apply GRU
        # print("Before GRU", concat.shape)
        activation, _ = self.GRU(concat)
        # print("After GRU", activation.shape)
    
        return activation[:, -1, :].view(-1, n_nodes, self.state_dim)
    

# Define the final output layer 
class Edge_Regressor(nn.Module):
    def __init__(self, state_dim, nb_features_edge, intermediate_dim):
        super(Edge_Regressor, self).__init__()
        
        self.hidden_layer_1 = nn.Sequential(
                    nn.Linear(2*state_dim + nb_features_edge, intermediate_dim),
                    nn.ReLU()
                )
        
        self.hidden_layer_2 = nn.Sequential(
                    nn.Linear(intermediate_dim, intermediate_dim),
                    nn.ReLU()
                )
        
        self.output_layer = nn.Linear(intermediate_dim, 1)

    def forward(self, nodes, edges):
        n_nodes = nodes.shape[1]
        node_dim = nodes.shape[2]

        state_i = nodes.repeat(1, 1, n_nodes).view(-1, n_nodes ** 2, node_dim)
        state_j = nodes.repeat(1, n_nodes, 1)

        concat = torch.cat([state_i, edges, state_j], dim=-1)
        activation_1 = self.hidden_layer_1(concat)
        activation_2 = self.hidden_layer_2(activation_1)

        return self.output_layer(activation_2)
    

class MP_Layer(nn.Module):
    def __init__(self, state_dim, nb_features_edge):
        super(MP_Layer, self).__init__()
        
        self.message_passers = Message_Passer_NNM(node_dim=state_dim, nb_features=nb_features_edge)
        self.message_aggs = Message_Agg()
        self.update_functions = Update_Func_GRU(state_dim=state_dim)

    def forward(self, nodes, edges, mask):
        n_nodes = nodes.shape[1]
        node_dim = nodes.shape[2]

        state_j = nodes.repeat(1, n_nodes, 1)
        messages = self.message_passers(state_j, edges)

        # Multiply messages by the mask to ignore messages from non-existent nodes
        masked = messages * mask
        masked = masked.view(messages.shape[0], n_nodes, n_nodes, node_dim)

        agg_m = self.message_aggs(masked)
        nodes_out = self.update_functions(nodes, agg_m)
        # Maybe add a batch norm.
        return nodes_out
    

class MPNN(nn.Module):
    def __init__(self, nb_features_node, nb_features_edge, out_int_dim, state_dim, T):
        super(MPNN, self).__init__()

        self.T = T
        self.embed = nn.Sequential(
                nn.Linear(nb_features_node, state_dim),
                nn.ReLU()
        )
        self.MP = MP_Layer(state_dim, nb_features_edge)
        self.edge_regressor = Edge_Regressor(state_dim, nb_features_edge, out_int_dim)
        self.relu = nn.ReLU()

    def forward(self, edges, nodes):
        # Get distances, and create a mask wherever 0 (i.e., non-existent nodes)
        # This also masks node self-interactions...
        # This assumes distance is first
        len_edges = edges.shape[-1]

        x, _ = torch.split(edges, [1, len_edges - 1], dim=2)

        mask = torch.where(x == 0, x, torch.ones_like(x))

        # Embed nodes to the chosen node dimension
        nodes = self.embed(nodes)

        # Run the T message-passing steps
        for mp in range(self.T):
            nodes = self.MP(nodes, edges, mask)

        # Regress the output values
        con_edges = self.edge_regressor(nodes, edges)

        return con_edges