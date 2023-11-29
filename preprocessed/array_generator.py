import pandas as pd
import os
import numpy as np
from scipy.spatial import distance_matrix
import json

FOLDER_DIR = "./preprocessed/"
NB_FEATURES_NODE = 5
NB_FEATURES_EDGE = 16


def make_nodes(train_structures_df, test_structures_df, n_train, n_test, max_size):
    nodes_train = np.zeros((n_train, max_size, NB_FEATURES_NODE), dtype=np.float32)
    nodes_test = np.zeros((n_test, max_size, NB_FEATURES_NODE), dtype=np.float32)

    for df, nodes in zip([train_structures_df, test_structures_df], [nodes_train, nodes_test]):
        molecule_indices = df["molecule_index"].values
        atom_indices = df["atom_index"].values
        features = df[["C", "F", "H", "N", "O"]].values

        nodes[molecule_indices, atom_indices] = features
 
    return nodes_train, nodes_test
    
def make_in_edges(train_df, test_df, train_structures_df, test_structures_df, train_bonds, test_bonds, train_angles, test_angles, n_train, n_test, max_size):
    in_edges_train = np.zeros((n_train, max_size, max_size, NB_FEATURES_EDGE), dtype=np.float32)
    in_edges_test = np.zeros((n_test, max_size, max_size, NB_FEATURES_EDGE), dtype=np.float32)

    # First, iterate through train_df and test_df
    for df, in_edges in zip([train_df, test_df], [in_edges_train, in_edges_test]):
        molecule_indices = df["molecule_index"].values
        atom_indices_0 = df["atom_index_0"].values
        atom_indices_1 = df["atom_index_1"].values
        features = df[["dist", '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN']].values

        in_edges[molecule_indices, atom_indices_0, atom_indices_1,:9] = features
        in_edges[molecule_indices, atom_indices_1, atom_indices_0,:9] = features

    # Then, iterate through train_structures_df and test_structures_df to complete the adjency matrix
    for df, in_edges in zip([train_structures_df, test_structures_df], [in_edges_train, in_edges_test]):
        for molecule_index, molecule_df in df.groupby("molecule_index"):
            coords = molecule_df[["x", "y", "z"]].values
            dist = distance_matrix(coords, coords)
            in_edges[molecule_index, :dist.shape[0], :dist.shape[1], 0] = dist


    # Add the bond features 
    for df, in_edges in zip([train_bonds, test_bonds], [in_edges_train, in_edges_test]):
        molecule_indices = df["molecule_index"].values
        atom_indices_0 = df["atom_index_0"].values
        atom_indices_1 = df["atom_index_1"].values
        features = df[['nbond_1', 'nbond_1.5', 'nbond_2', 'nbond_3']].values

        in_edges[molecule_indices, atom_indices_0, atom_indices_1,9:13] = features
        in_edges[molecule_indices, atom_indices_1, atom_indices_0,9:13] = features
        
    # Finally, add the angles features
    for df, in_edges in zip([train_angles, test_angles], [in_edges_train, in_edges_test]):
        molecule_indices = df["molecule_index"].values
        atom_indices_0 = df["atom_index_0"].values
        atom_indices_1 = df["atom_index_1"].values
        features = df[['shortest_path_n_bonds', 'cosinus', 'dihedral']].values

        in_edges[molecule_indices, atom_indices_0, atom_indices_1,13:] = features
        in_edges[molecule_indices, atom_indices_1, atom_indices_0,13:] = features

    return in_edges_train, in_edges_test
    
def make_out_edges(train_df, n_train, max_size):

    out_edges_train = np.zeros((n_train, max_size, max_size), dtype=np.float32)

    molecule_indices = train_df["molecule_index"].values
    atom_indices_0 = train_df["atom_index_0"].values
    atom_indices_1 = train_df["atom_index_1"].values
    scc_values = train_df["scalar_coupling_constant"].values

    out_edges_train[molecule_indices, atom_indices_0, atom_indices_1] = scc_values
    out_edges_train[molecule_indices, atom_indices_1, atom_indices_0] = scc_values

    return out_edges_train

def make_three_bonds_adjacency_matrix(train_bonds, test_bonds, n_train, n_test, max_size):
    """Creates an adjacency matrix that considers an edge if two atoms are fewer than 3 bonds apart."""
    adjacency_train = np.zeros((n_train, max_size, max_size), dtype=bool)
    adjacency_test = np.zeros((n_test, max_size, max_size), dtype=bool)
    
    final_adjacencies = []

    for adjacency, df_bonds in zip([adjacency_train, adjacency_test], [train_bonds, test_bonds]):
        # First iteration
        adjacency[df_bonds["molecule_index"], df_bonds["atom_index_0"], df_bonds["atom_index_1"]] = 1
        adjacency[df_bonds["molecule_index"], df_bonds["atom_index_1"], df_bonds["atom_index_0"]] = 1

        # Two bonds or one bond
        two_bonds_away = np.matmul(adjacency, adjacency)
        combined_adjacency = np.logical_or(adjacency, two_bonds_away)

        # We make sure that an atom is not linked to itself
        molecule_ind, atom_ind = np.arange(len(adjacency))[:, None, None], np.arange(max_size)
        combined_adjacency[molecule_ind, atom_ind, atom_ind] = False

        # Three bonds or less
        three_bonds_away = np.matmul(adjacency, combined_adjacency)
        fully_combined_adjacency = np.logical_or(combined_adjacency, three_bonds_away)
        fully_combined_adjacency[molecule_ind, atom_ind, atom_ind] = False

        final_adjacencies.append(fully_combined_adjacency.astype(np.float32))

    return final_adjacencies

def array_generator():
    print("Generating arrays ...")
    train_df = pd.read_csv(os.path.join(FOLDER_DIR,'train_df.csv'))
    test_df = pd.read_csv(os.path.join(FOLDER_DIR,'test_df.csv'))


    train_structures_df = pd.read_csv(os.path.join(FOLDER_DIR,'train_structures_df.csv'))
    test_structures_df = pd.read_csv(os.path.join(FOLDER_DIR,'test_structures_df.csv'))


    # train_bonds and test_bonds come from BondFeatures.ipynb
    train_bonds = pd.read_csv(os.path.join(FOLDER_DIR,'train_bonds.csv'))
    test_bonds = pd.read_csv(os.path.join(FOLDER_DIR,'test_bonds.csv'))

   
    # train_angles_df and test_angles_df come from make_angles_dataframe.ipynb
    train_angles = pd.read_csv(os.path.join(FOLDER_DIR,'train_angles_df.csv'))
    test_angles = pd.read_csv(os.path.join(FOLDER_DIR,'test_angles_df.csv'))


    train_df["molecule_index"] = pd.factorize(train_df["molecule_name"])[0]
    test_df["molecule_index"] = pd.factorize(test_df["molecule_name"])[0]

    # Normalize scalar_coupling_constant
    minimum = train_df['scalar_coupling_constant'].min()
    maximum = train_df['scalar_coupling_constant'].max()
    scale_mid = (minimum + maximum) / 2
    scale_norm = maximum - minimum
    train_df['scalar_coupling_constant'] = (train_df['scalar_coupling_constant'] - scale_mid) / scale_norm


    train_structures_df["molecule_index"] = pd.factorize(train_structures_df["molecule_name"])[0]
    test_structures_df["molecule_index"] = pd.factorize(test_structures_df["molecule_name"])[0]


    train_bonds["molecule_index"] = pd.factorize(train_bonds["molecule_name"])[0]
    test_bonds["molecule_index"] = pd.factorize(test_bonds["molecule_name"])[0]
    train_bonds[['nbond_1', 'nbond_1.5', 'nbond_2', 'nbond_3']] = pd.get_dummies(train_bonds['nbond'])
    test_bonds[['nbond_1', 'nbond_1.5', 'nbond_2', 'nbond_3']] = pd.get_dummies(test_bonds['nbond'])


    train_angles['shortest_path_n_bonds'] = train_angles['shortest_path_n_bonds'] / 6
    test_angles['shortest_path_n_bonds'] = test_angles['shortest_path_n_bonds'] / 6
    train_angles['dihedral'] = train_angles['dihedral'] / np.pi
    test_angles['dihedral'] = test_angles['dihedral'] / np.pi
    train_angles["molecule_index"] = pd.factorize(train_angles["molecule_name"])[0]
    test_angles["molecule_index"] = pd.factorize(test_angles["molecule_name"])[0]
    train_angles = train_angles.fillna(0)
    test_angles = test_angles.fillna(0)


    max_size_train = max(train_df.groupby('molecule_name')['atom_index_0'].max())
    max_size_test = max(test_df.groupby('molecule_name')['atom_index_0'].max())
    max_size = max(max_size_train, max_size_test) + 1 # We are given indexes so that goes from 0 to max_size_train or max_size_test
    n_train = train_df['molecule_name'].nunique()
    n_test = test_df['molecule_name'].nunique()

    # This means that :
    # nodes_train.size = [nb_molecule_train, max_size, nb_features_nodes] = [68009, 29, 8]
    # nodes_test.size = [nb_molecule_test, max_size, nb_features_nodes] = [17003, 29, 8]
    # in_edges_train.size = [nb_molecule_train, max_size, max_size, nb_features_edges] = [68009, 29, 29, 19]
    # in_edges_test.size = [nb_molecule_test, max_size, max_size, nb_features_edges] = [17003, 29, 29, 19]
    # out_edges_train.size = [nb_molecule_train, max_size, max_size, 1] = [68009, 29, 29, 1]
    # Because the features for the nodes are : the atome, its position (x,y,z).
    # And the features for the edges are : the distance, dist_x, dist_y, dist_z, the type of the coupling, the bond features, and the angle features. 

    nodes_train, nodes_test = make_nodes(train_structures_df, test_structures_df, n_train, n_test, max_size)
    np.save(FOLDER_DIR + "nodes_train", nodes_train)
    np.save(FOLDER_DIR + "nodes_test", nodes_test)
    print("nodes_train and nodes_test created.")

    in_edges_train, in_edges_test = make_in_edges(train_df, test_df, train_structures_df, test_structures_df, train_bonds, test_bonds, train_angles, test_angles, n_train, n_test, max_size)
    np.save(FOLDER_DIR + "in_edges_train", in_edges_train)
    np.save(FOLDER_DIR + "in_edges_test", in_edges_test)
    print("in_edges_train and in_edges_test created.")

    out_edges_train = make_out_edges(train_df, n_train, max_size)
    np.save(FOLDER_DIR + "out_edges_train", out_edges_train)
    print("out_edges_train created.")

    adjacency_train, adjacency_test = make_three_bonds_adjacency_matrix(train_bonds, test_bonds, n_train, n_test, max_size)
    np.save(FOLDER_DIR + "three_bonds_adjacency_train", adjacency_train)
    np.save(FOLDER_DIR + "three_bonds_adjacency_test", adjacency_test)
    print("Adjacency matrices for three-bond connections created.")

    with open(FOLDER_DIR + "normalization_params.json", "w") as f:
        json.dump({"scale_mid":scale_mid, 
                   "scale_norm":scale_norm}, f)
        