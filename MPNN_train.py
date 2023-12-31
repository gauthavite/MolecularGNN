from tqdm import tqdm
import numpy as np

from preprocessed.array_generator import array_generator, NB_FEATURES_EDGE, NB_FEATURES_NODE

import torch
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler

from models.MPNN import MPNN

FOLDER_DIR = "./preprocessed/"

# Training params
RATIO_TRAIN_VAL = 0.8
N_EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 16
MPNN_TYPE = "MGC"  # "basic" or "basic_learnable" or "GRU" or "MGC"
ADJACENCY = True # If True, uses only the three-bond connections

# Model params
OUT_INT_DIM = 512
STATE_DIM = 128
T = 4
SAVE = True
SAVE_PATH = "./models/MPNN_MGC_ADJACENCY.pt"

torch.manual_seed(7954501528462238601)
np.random.seed(42)

def load_arrays():
    nodes_train = np.load(FOLDER_DIR + "nodes_train.npy" )
    in_edges_train = np.load(FOLDER_DIR + "in_edges_train.npy")
    out_edges_train = np.load(FOLDER_DIR + "out_edges_train.npy")

    adjacency_train = np.load(FOLDER_DIR + "three_bonds_adjacency_train.npy")

    return nodes_train, in_edges_train, out_edges_train, adjacency_train

try : 
    print("Loading the arrays ...")
    nodes_train, in_edges_train, out_edges_train, adjacency_train = load_arrays()
except FileNotFoundError:
    print("Loading failed.")
    array_generator()
    nodes_train, in_edges_train, out_edges_train, adjacency_train = load_arrays()

# Load the model
print("Instantiating the model ...")
mpnn = MPNN(nb_features_node = NB_FEATURES_NODE, nb_features_edge = NB_FEATURES_EDGE, out_int_dim = OUT_INT_DIM, state_dim = STATE_DIM, T = T, mpnn_type=MPNN_TYPE, adjacency=ADJACENCY)
print("The MPNN type is:", MPNN_TYPE)
if ADJACENCY:
    print("Using only the three-bond connections edges.")
else:
    print("Using a complete graph.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device is :", device)
mpnn = mpnn.to(device)

# Dataset creation
print("Instantiating the dataset ...")
# Reshape 29*29 into 841
out_edges_train = out_edges_train.reshape(-1,out_edges_train.shape[1]*out_edges_train.shape[2],1)
in_edges_train = in_edges_train.reshape(-1,in_edges_train.shape[1]*in_edges_train.shape[2],in_edges_train.shape[3])
adjacency_train = adjacency_train.reshape(-1, adjacency_train.shape[1]*adjacency_train.shape[2],1)
# Shuffle 
nodes_train, in_edges_train, out_labels, adjacency_train = shuffle(nodes_train, in_edges_train, out_edges_train, adjacency_train)

class Set(Dataset):
    def __init__(self, in_nodes, in_edges, out_edges, adjacency):
        self.nodes = in_nodes
        self.in_edges = in_edges
        self.out_edges = out_edges
        self.adjacency = adjacency
        
    def __len__(self):
        return len(self.nodes)
        
    def __getitem__(self, idx):
        s1 = self.nodes[idx]
        s2 = self.in_edges[idx]
        s3 = self.out_edges[idx]
        s4 = self.adjacency[idx]
        return s1, s2, s3, s4

nb_train = int(len(nodes_train)*RATIO_TRAIN_VAL)
train_set = Set(nodes_train[:nb_train], in_edges_train[:nb_train], out_labels[:nb_train], adjacency_train[:nb_train]) 
val_set = Set(nodes_train[nb_train:], in_edges_train[nb_train:], out_labels[nb_train:], adjacency_train[nb_train:]) 

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)


# Loss function
def log_mae(orig, preds):
    # Mask values for which no scalar coupling exists
    mask = orig != 0
    nums = orig[mask]
    preds = preds[mask]
    reconstruction_error = torch.log(torch.mean(torch.abs(nums - preds)))
    return reconstruction_error

optimizer = Adam(params = mpnn.parameters(), lr=LEARNING_RATE)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=len(train_loader)*N_EPOCHS//2)
    
##### TRAINING #####
for i in range(N_EPOCHS):
    print(f"__________EPOCH {i+1}__________")
    optimizer.step()
    mpnn.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        nodes, in_edges, out_edges, adjacency = batch
        nodes, in_edges, out_edges, adjacency = nodes.to(device), in_edges.to(device), out_edges.to(device), adjacency.to(device)
        out = mpnn(in_edges, nodes, adjacency)
        loss = log_mae(out_edges, out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()

    average_loss = total_loss / len(train_loader)
    print("average train loss over an epoch :", average_loss)

    mpnn.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            nodes, in_edges, out_edges, adjacency = batch
            nodes, in_edges, out_edges, adjacency = nodes.to(device), in_edges.to(device), out_edges.to(device), adjacency.to(device)
            out = mpnn(in_edges, nodes, adjacency)
            loss = log_mae(out_edges, out)
            val_loss += loss.item()

    average_loss = val_loss / len(val_loader)
    print("average val loss", average_loss)

if SAVE:
    torch.save(mpnn.state_dict(), SAVE_PATH)
    print("Model saved in", SAVE_PATH)
