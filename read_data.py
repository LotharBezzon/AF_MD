import numpy as np
import os
import json

def get_data(protein, num_models):

    folder = f'fold_{protein}'

    cpms = []
    paes = []
    plddtss = []
    for i in range(num_models):
        json_file = os.path.join(folder, f'fold_{protein}_full_data_{i}.json')
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        cpms.append(np.array(data['contact_probs']))
        paes.append(np.array(data['pae']))
        plddtss.append(np.array(data['atom_plddts']))
    
    cpm = np.mean(cpms, axis=0)
    pae = np.mean(paes, axis=0)
    plddts = np.mean(plddtss, axis=0)

    return cpm, pae, plddts

def get_pos(protein):

    folder = f'fold_{protein}'
    
    json_file = os.path.join(folder, f'fold_{protein}_full_data_0.json')
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    pos = np.array(data['positions'])
    return pos

def read_data(protein, num_models=5):

    folder = f'fold_{protein}'
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist. Download data from https://alphafoldserver.com/")
    
    cpm, pae, plddts = get_data(protein, num_models)

    return cpm, pae, plddts
    
    


if __name__ == "__main__":
    protein = 'chignolin'
    read_data(protein)
