from read_data import read_data
import numpy as np
import pickle
from scipy.special import softmax

protein = 'chignolin'

cpm, pae, plddts = read_data(protein)

with open('chignolin.pickle', 'rb') as f:
    data = pickle.load(f)
    print(softmax(data['distogram']['logits'][0][5]))