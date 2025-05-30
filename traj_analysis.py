import pyemma
import numpy as np
import matplotlib.pyplot as plt
from utils import xyz_to_numpy

protein = 'chignolin'

coords = xyz_to_numpy(f'{protein}.xyz')
data = pyemma.coordinates.source(coords).save    # Write the coordinates in the pyemma format


