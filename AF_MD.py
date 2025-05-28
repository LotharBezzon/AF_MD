from read_data import *
from utils import *
import numpy as np
from scipy.special import softmax
from openmm import app, unit
import openmm as mm
from openmm.vec3 import Vec3

protein = 'chignolin'
coords, bins, pae, dgram, seq_len, seq = read_data(protein)

### Set up the OpenMM system
system = mm.System()
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
use a 2d tabulated function (actual potential in rows and particle indexes to choose the rigth row)'''

# Create potentials
values = np.concatenate([dgram[i, j] for i in range(seq_len) for j in range(seq_len)])
potentials = mm.Continuous2DFunction(64, seq_len**2, values, 2+10/64, 22-10/64, 0, seq_len**2)

# Create force field
AF_force = mm.CustomNonbondedForce('U(r, index); index = seq_len * index1 + index2')
AF_force.setCutoffDistance(21.0 * unit.angstroms)   # because the last bin include farther distances and I'm not treating it
AF_force.addPerParticleParameter('index')
AF_force.addGlobalParameter('seq_len', seq_len)
AF_force.addTabulatedFunction('U', potentials)
for i in range(seq_len):
    AF_force.addParticle([i])

system.addForce(AF_force)

### Set up the integrator
temperature = 300.0 * unit.kelvin
friction_coefficient = 1.0 / unit.picosecond
step_size = 2.0 * unit.femtosecond

integrator = mm.LangevinIntegrator(temperature, friction_coefficient, step_size)



