import torch
import torch.nn as nn

class Basic_Aggregator(nn.Module):
    def __init__(self, node_dim, nb_features):
        super(Basic_Aggregator, self).__init__()
        self.node_dim = node_dim


    def forward(self, nodes, edges, mask):   
        size = nodes.shape[1]

        mask = mask.view(-1, size, size)
        messages = torch.matmul(mask, nodes)
        return messages

class Basic_Combiner(nn.Module):
    def __init__(self, state_dim):
        super(Basic_Combiner, self).__init__()
        self.state_dim = state_dim
        self.nn = nn.Linear(in_features=state_dim, out_features=state_dim)
        self.relu = nn.ReLU()
                
    def forward(self, old_state, agg_messages):
        activation = self.relu(self.nn(agg_messages) + old_state)
        return activation


class Basic_Learnable_Aggregator(nn.Module):
    def __init__(self, node_dim, nb_features):
        super(Basic_Learnable_Aggregator, self).__init__()
        self.node_dim = node_dim
        self.linear = nn.Linear(node_dim, node_dim)

    def forward(self, nodes, edges, mask):   
        size = nodes.shape[1]

        mask = mask.view(-1, size, size)
        messages = torch.matmul(mask, nodes)

        messages = self.linear(messages)

        return messages
    

class Basic_Learnable_Combiner(nn.Module):
    def __init__(self, state_dim):
        super(Basic_Learnable_Combiner, self).__init__()
        self.state_dim = state_dim
        self.nn = nn.Linear(in_features=state_dim, out_features=state_dim)
                
    def forward(self, old_state, agg_messages):
        activation = self.nn(agg_messages) + old_state
        return activation


class GRU_Aggregator(nn.Module):
    def __init__(self, node_dim, nb_features):
        super(GRU_Aggregator, self).__init__()
        self.nb_features = nb_features
        self.node_dim = node_dim
        self.nn = nn.Sequential(
                    nn.Linear(in_features=self.nb_features, out_features=self.node_dim**2),
                    nn.ReLU()
                )
      
    def forward(self, nodes, edges, mask):
        n_nodes = nodes.shape[1]

        node_j = nodes.repeat(1, n_nodes, 1)
    
        # Embed the edge as a matrix
        A = self.nn(edges)
        
        # Reshape so matrix mult can be done
        A = A.view(-1, self.node_dim, self.node_dim)
        node_j = node_j.view(-1, self.node_dim, 1)
        
        # Multiply edge matrix by node and shape into message list
        messages = torch.matmul(A, node_j)
        messages = messages.view(-1, edges.shape[1], self.node_dim)

        # Multiply messages by the mask to ignore messages from non-existent nodes
        masked = messages * mask
        masked = masked.view(messages.shape[0], n_nodes, n_nodes, self.node_dim)

        agg_m = torch.sum(masked, 2)

        return agg_m

class GRU_Combiner(nn.Module):
    def __init__(self, state_dim):
        super(GRU_Combiner, self).__init__()
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
    

class MP_Layer(nn.Module):
    def __init__(self, state_dim, nb_features_edge, mpnn_type):
        super(MP_Layer, self).__init__()

        self.mpnn_type = mpnn_type
        if mpnn_type == "GRU":
            self.aggregator = GRU_Aggregator(node_dim=state_dim, nb_features=nb_features_edge)
            self.combiner = GRU_Combiner(state_dim=state_dim)
        elif mpnn_type == "basic":
            self.aggregator = Basic_Aggregator(node_dim=state_dim, nb_features=nb_features_edge)
            self.combiner = Basic_Combiner(state_dim=state_dim)
        elif mpnn_type == "basic_learnable":
            self.aggregator = Basic_Learnable_Aggregator(node_dim=state_dim, nb_features=nb_features_edge)
            self.combiner = Basic_Learnable_Combiner(state_dim=state_dim)
        else:
            raise TypeError(f"{mpnn_type=} argument is inappropriate.")
        

    def forward(self, nodes, edges, mask):
        agg_m = self.aggregator(nodes, edges, mask)

        nodes_out = self.combiner(nodes, agg_m)

        # Maybe add a batch norm.
        
        return nodes_out

class MGC_Aggregator(nn.Module):
    def __init__(self, state_dim):
        super(MGC_Aggregator, self).__init__()
        self.state_dim = state_dim

    def forward(self, nodes, edges, mask):  
        n_nodes = nodes.shape[1]

        messages_nodes = edges * mask
        messages_nodes = messages_nodes.view(-1, n_nodes, n_nodes, self.state_dim)
        messages_nodes = torch.sum(messages_nodes, 2)
        return messages_nodes
    
class MGC_Combiner(nn.Module):
    def __init__(self, state_dim):
        super(MGC_Combiner, self).__init__()
        self.state_dim = state_dim
        self.W0 = nn.Sequential(
                nn.Linear(in_features=state_dim, out_features=state_dim),
                nn.ReLU(),
            )
        self.W1 = nn.Sequential(
                nn.Linear(in_features=2*state_dim, out_features=state_dim),
                nn.ReLU(),
            )
        self.W2 = nn.Sequential(
                nn.Linear(in_features=state_dim, out_features=state_dim),
                nn.ReLU(),
            )
        self.W3 = nn.Sequential(
                nn.Linear(in_features=2*state_dim, out_features=state_dim),
                nn.ReLU(),
            )
        self.W4 = nn.Sequential(
                nn.Linear(in_features=state_dim, out_features=state_dim),
                nn.ReLU(),
            )
                
    def forward(self, old_nodes, old_edges, messages_nodes):
        n_nodes = old_nodes.shape[1]

        nodes_messages_cat = torch.cat([self.W0(old_nodes), messages_nodes], dim=-1)
        activation_nodes = self.W1(nodes_messages_cat)

        node_i = old_nodes.repeat(1, 1, n_nodes).view(-1, n_nodes ** 2, self.state_dim)
        node_j = old_nodes.repeat(1, n_nodes, 1)
        node_node_concat = torch.cat([node_i, node_j], dim=-1)

        activation_edges = self.W4(self.W2(old_edges) + self.W3(node_node_concat))
        return activation_nodes, activation_edges


class MP_Layer_MGC(nn.Module):
    def __init__(self, state_dim):
        super(MP_Layer_MGC, self).__init__()

        self.aggregator = MGC_Aggregator(state_dim=state_dim)

        self.combiner = MGC_Combiner(state_dim=state_dim)        

    def forward(self, nodes, edges, mask):
        messages_nodes = self.aggregator(nodes, edges, mask)

        nodes_out, edges_out = self.combiner(nodes, edges, messages_nodes)

        # Maybe add a batch norm.
        
        return nodes_out, edges_out

# Define the final output layer 
class Readout(nn.Module):
    def __init__(self, state_dim, nb_features_edge, intermediate_dim):
        super(Readout, self).__init__()
        
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
        

class MPNN(nn.Module):
    def __init__(self, nb_features_node, nb_features_edge, out_int_dim, state_dim, T, mpnn_type="GRU", adjacency=False):
        super(MPNN, self).__init__()

        self.T = T
        self.embed_node = nn.Sequential(
                nn.Linear(nb_features_node, state_dim),
                nn.ReLU()
        )
        self.embed_edge = nn.Sequential(
                nn.Linear(nb_features_edge, state_dim),
                nn.ReLU()
        )

        self.mpnn_type = mpnn_type

        # If the type is "MGC", we define another MP_Layer class which will update the edges as well as the nodes.
        if self.mpnn_type == "MGC":
            self.MP = MP_Layer_MGC(state_dim)
            nb_features_edge = state_dim # The edges are now embedded
        else:
            self.MP = MP_Layer(state_dim, nb_features_edge, mpnn_type=mpnn_type)

        self.readout = Readout(state_dim, nb_features_edge, out_int_dim)
        self.relu = nn.ReLU()
        self.adjacency = adjacency

    def forward(self, edges, nodes, adjacency):
        len_edges = edges.shape[-1]

        # x contains the distances between every atom
        # We will create a mask wherever the distance is 0 
        # This will mask non-existant nodes (if the molecule size is not maximal), and the self-interactions
        x, _ = torch.split(edges, [1, len_edges - 1], dim=2)

        # If self.adjacency is True, we will use the three-bond connection adjacency matrix as the mask
        if self.adjacency:
            mask = adjacency
        else:
            mask = torch.where(x == 0, x, torch.ones_like(x))

        # Embed nodes to the chosen node dimension
        nodes = self.embed_node(nodes)

        if self.mpnn_type == "MGC":
            # If we choose to update the edges, we make an embedding of them.
            edges = self.embed_edge(edges)

        # Message Passing steps
        for _ in range(self.T):
            if self.mpnn_type == "MGC":
                # Updates the nodes and the edges
                nodes, edges = self.MP(nodes, edges, mask)
            else:
                # Updates only the nodes
                nodes = self.MP(nodes, edges, mask)

        # Readout
        con_edges = self.readout(nodes, edges)

        return con_edges