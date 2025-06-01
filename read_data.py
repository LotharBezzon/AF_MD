import numpy as np
import os
import pickle

def check_folder_exists(protein):
    """
    Check if the file exists in the current directory.
    """
    return os.path.isfile(f'{protein}/pkl0.pickle')

def read_data(protein, num_models=5):

    if not check_folder_exists(protein):
        raise FileNotFoundError(
            f"Folder {protein}/ doesn't exist or doesn't contain necessary information.\n"
            "Please, rename it correctly or download data from https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb"
        )

    seq_len, seq = np.loadtxt(f'{protein}/seq.txt', dtype=str)
    seq_len = int(seq_len)
    
    paes = []
    dgrams = []

    for i in range(num_models):

        with open(f'{protein}/pkl{i}.pickle', 'rb') as f:

            data = pickle.load(f)

            # Extract pae
            paes.append(np.array(data['predicted_aligned_error']))

            # Extract dgram
            dgrams.append(np.array(data['distogram']['logits']))

            if i == 0:
                # Extract bin edges
                bins = data['distogram']['bin_edges']

    pae = np.mean(paes, axis=0)
    dgram = np.mean(dgrams, axis=0)

    # Get starting positions
    with open(f'{protein}/starting_structure.pdb', 'r') as f:

        lines = f.readlines()
        coords = []

        for line in lines:
            words = line.split()
            if words[0] == 'ATOM':

                if words[3] == 'GLY' and words[2] == 'CA':
                    coords.append([float(words[6]), float(words[7]), float(words[8])])

                if words[2] == 'CB':
                    coords.append([float(words[6]), float(words[7]), float(words[8])])

    coords = np.array(coords)

    return coords, bins, pae, dgram, seq_len, seq



if __name__ == "__main__":
    protein = 'trp_cage'  # Example protein name, change as needed
    coords, bins, pae, dgram, seq_len, seq = read_data(protein)
    for i in range(seq_len):
        for j in range(seq_len):
            dgram[i, j] = dgram[i, j] - np.max(dgram[i, j])
    print(dgram[4,7])
    print(bins)
