from read_data import *
from utils import *
import numpy as np
from scipy.special import softmax
from openmm import app, unit, openmm
from openmm.vec3 import Vec3

protein = 'chignolin'
coords, bins, pae, dgram, seq_len, seq = read_data(protein)

### Set up the OpenMM system
system = openmm.System()
system.setDefaultPeriodicBoxVectors(
    Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10)
) # A sufficiently large box for non-periodic simulations

### Create particles
masses = get_bead_masses_from_sequence(seq)
for i in range(seq_len):
    system.addParticle(masses[i] * unit.amu)

### Define potentials
'''
use CustomNonbondedForce to define the potentials (all beads are connected)
use a 2d tabulated function (actual potential in rows and particle parameters to choose the rigth row)'''
