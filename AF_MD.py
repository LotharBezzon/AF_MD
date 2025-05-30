from read_data import *
from utils import *
import numpy as np
from scipy.special import softmax
from openmm import app, unit
import openmm as mm
from openmm.vec3 import Vec3

### Load the data for the protein
# This assumes you have a folder named 'chignolin' with the necessary files.
protein = 'chignolin'
coords, bins, pae, dgram, seq_len, seq = read_data(protein)
for i in range(seq_len):
    for j in range(seq_len):
        dgram[i, j] = dgram[i, j] - np.max(dgram[i, j])  # Needed for numerical stability. Note that evaluated probabilities don't change

### Set up the OpenMM system
system = mm.System()
system.setDefaultPeriodicBoxVectors(
    Vec3(10, 0, 0), Vec3(0, 10, 0), Vec3(0, 0, 10)
) # A sufficiently large box for non-periodic simulations. Length are expressed in nanometers.

### Create particles
masses = get_bead_masses_from_sequence(seq)
for i in range(seq_len):
    system.addParticle(masses[i] * unit.amu)

### Define potentials
'''
use CustomNonbondedForce to define the potentials (all beads are connected!)
use a 2d tabulated function (actual potential in rows and particle indexes to choose the rigth row)'''

# Create potentials
values = np.concatenate([-dgram[i, j] + np.log(np.sum(np.exp(dgram[i,j]))) for i in range(seq_len) for j in range(seq_len)])
potentials = mm.Continuous2DFunction(64, seq_len**2, values, (2+10/64)/10, (22-10/64)/10, 0, seq_len**2)    # lenths are expressed in nanometers, so we need to scale them down to match the OpenMM units

# Create force field
AF_force = mm.CustomNonbondedForce('U(r, index); index = seq_len * index1 + index2')
AF_force.setCutoffDistance(21.0 * unit.angstroms)   # because the last bin include farther distances and I'm not treating it
AF_force.addPerParticleParameter('index')
AF_force.addGlobalParameter('seq_len', seq_len)
AF_force.addTabulatedFunction('U', potentials)
AF_force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)  # Use periodic boundary conditions
print(f'cutoff distance: {AF_force.getCutoffDistance()}')
print(f'force field is periodic: {AF_force.usesPeriodicBoundaryConditions()}')
for i in range(seq_len):
    AF_force.addParticle([i])

system.addForce(AF_force)

### Set up the simulation
# Set up the integrator
temperature = 300.0 * unit.kelvin
friction_coefficient = 1.0 / unit.picosecond
step_size = 0.01 * unit.picosecond

integrator = mm.LangevinIntegrator(temperature, friction_coefficient, step_size)

# Platform selection
platform = mm.Platform.getPlatformByName('Reference') # Use 'Reference' for CPU, 'CUDA' or 'OpenCL' for GPU
# For simple systems, 'Reference' is good for debugging and clarity.

context = mm.Context(system, integrator, platform)
context.setPositions(coords * unit.angstrom)  # Set initial positions
context.setVelocitiesToTemperature(temperature)  # Set initial velocities


### Run the simulation
tot_steps = 10000  # Number of steps to run the simulation
num_steps = 0
while num_steps < tot_steps:
    state = context.getState(energy=True, positions=True)
    write_xyz_file_frame(f'{protein}.xyz', state, seq, seq_len, num_steps)
    num_steps += 100
    integrator.step(100)  # Run 100 steps at a time
    


