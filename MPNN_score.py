import torch 
from tqdm import tqdm
import numpy as np
from models.MPNN import MPNN
import json
import pandas as pd

from preprocessed.array_generator import array_generator, NB_FEATURES_EDGE, NB_FEATURES_NODE

# Model params
OUT_INT_DIM = 512
STATE_DIM = 128
T = 4

FOLDER_DIR = "./preprocessed/"
MODEL_PATH = "./models/mpnn.pt"

def load_arrays():
    print("Loading the arrays ...")
    nodes_test     = np.load(FOLDER_DIR + "nodes_test.npy" )
    in_edges_test  = np.load(FOLDER_DIR + "in_edges_test.npy")
    in_edges_test  = in_edges_test.reshape(-1,in_edges_test.shape[1]*in_edges_test.shape[2],in_edges_test.shape[3])

    with open(FOLDER_DIR + "normalization_params.json", "r") as f:
        normalization_params = json.load(f)

    return nodes_test, in_edges_test, normalization_params

try : 
    nodes_test, in_edges_test, normalization_params = load_arrays()
except FileNotFoundError:
    array_generator()
    nodes_test, in_edges_test, normalization_params = load_arrays()

scale_norm = normalization_params["scale_norm"]
scale_mid = normalization_params["scale_mid"]

model = MPNN(nb_features_node = NB_FEATURES_NODE, nb_features_edge = NB_FEATURES_EDGE, out_int_dim = OUT_INT_DIM, state_dim = STATE_DIM, T = T)
model.load_state_dict(torch.load(MODEL_PATH))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

print("Predicting ...")
preds = []
for node, edge in tqdm(zip(nodes_test, in_edges_test), total=len(nodes_test)):
    node =  torch.tensor(node).unsqueeze(0).to(device)
    edge =  torch.tensor(edge).unsqueeze(0).to(device)
    pred = model(edge, node)
    pred = pred.cpu().detach().numpy()
    preds.append(pred)
preds = np.array(preds)
preds = preds.reshape(len(preds), nodes_test.shape[1], nodes_test.shape[1], 1)

test_df = pd.read_csv(FOLDER_DIR + "test_df.csv")
test_group = test_df.groupby('molecule_name')

def make_outs(test_group, preds):
    x = np.array([])
    for test_gp, preds in tqdm(zip(test_group, preds), total=len(preds)):
        gp = test_gp[1]  
        x = np.append(x, (preds[gp['atom_index_0'].values, gp['atom_index_1'].values] + preds[gp['atom_index_1'].values, gp['atom_index_0'].values])/2.0)
    return x

preds = make_outs(test_group, preds)
preds = preds*scale_norm + scale_mid

def score(preds, targets):
    return np.log(np.mean(np.abs(targets - preds)))

targets = test_df['scalar_coupling_constant']
print("Score (log MAE):", score(preds, targets))
