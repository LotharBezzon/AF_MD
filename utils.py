from openmm import app
import openmm as mm
import numpy as np
import pyemma
import deeptime

def get_bead_masses_from_sequence(amino_acid_sequence: str):
    """
    Calculates the mass for each bead in a coarse-grained protein model,
    assuming one bead per amino acid residue.

    Args:
        amino_acid_sequence (str): A string representing the amino acid sequence
                                   (e.g., "ARNDCEQGHI"). Uses one-letter codes.

    Returns:
        list: A list of mm.app.element.Element for each
              amino acid in the sequence.
              Returns an empty list if the sequence is empty or contains
              unrecognized amino acid codes.
    """
    # Average molecular weights of common amino acid residues (in Daltons or amu).
    # These values represent the mass of the amino acid after a water molecule
    # (H2O, ~18.015 Da) has been removed during peptide bond formation.
    # Source: Based on standard biochemical tables (e.g., Sigma-Aldrich).
    amino_acid_residue_masses = {
        'A': 71.0788,   # Alanine
        'R': 156.1875,  # Arginine
        'N': 114.1038,  # Asparagine
        'D': 115.0886,  # Aspartic Acid
        'C': 103.1388,  # Cysteine
        'Q': 128.1307,  # Glutamine
        'E': 129.1155,  # Glutamic Acid
        'G': 57.0519,   # Glycine
        'H': 137.1411,  # Histidine
        'I': 113.1594,  # Isoleucine
        'L': 113.1594,  # Leucine
        'K': 128.1741,  # Lysine
        'M': 131.1926,  # Methionine
        'F': 147.1766,  # Phenylalanine
        'P': 97.1167,   # Proline
        'S': 87.0782,   # Serine
        'T': 101.1051,  # Threonine
        'W': 186.2132,  # Tryptophan
        'Y': 163.1760,  # Tyrosine
        'V': 99.1326    # Valine
    }
    
    masses = []
    
    # Convert the sequence to uppercase to handle both 'a' and 'A'
    for aa_code in amino_acid_sequence.upper():
        if aa_code in amino_acid_residue_masses:
            masses.append(amino_acid_residue_masses[aa_code])
        else:
            raise ValueError(
                f"Unrecognized amino acid code '{aa_code}' in sequence. "
                "Please use one-letter codes (e.g., 'ARNDCEQGHI')."
            )
    return masses


def write_xyz_file_frame(filename, state, seq, seq_len, frame_num):
    """
    Writes the coordinates of the system to an XYZ file.

    Args:
        filename (str): The name of the output file.
        state (mm.State): The state containing the coordinates.
        seq (str): The amino acid sequence.
        seq_len (int): The length of the sequence.
    """
    if frame_num == 0:
        writing_type = 'w'
    else:
        writing_type = 'a'

    with open(filename, writing_type) as f:
        f.write(f"{seq_len}\n")
        f.write(f"Frame: {frame_num}\n")
        positions = state.getPositions() * 10 # Convert from nm to Angstroms
        for i in range(seq_len):
            f.write(f"{seq[i]} {positions[i].x:.3f} {positions[i].y:.3f} {positions[i].z:.3f}\n")


def xyz_to_numpy(filename):
    """
    Reads an XYZ file and returns the coordinates as a NumPy array.

    Args:
        filename (str): The name of the XYZ file.

    Returns:
        np.ndarray: An array of shape (T, N, 3) containing the coordinates. T is the number of frames,
        N is the number of atoms, and 3 corresponds to the x, y, z coordinates.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_atoms = int(lines[0].strip())
    num_frames = len(lines) // (num_atoms + 2)  # Each frame has num_atoms + 2 lines
    
    coords = np.empty((num_frames, num_atoms, 3), dtype=np.float32)
    for frame in range(num_frames):
        start_line = frame * (num_atoms + 2) + 2  # Skip the first two lines
        for atom in range(num_atoms):
            line = lines[start_line + atom].strip().split()
            coords[frame, atom] = [float(line[1]), float(line[2]), float(line[3])]

    return coords


def get_inter_bead_distances(coords):
    """
    Calculates the inter-bead distances for a set of coordinates. First neighbours are ignored.

    Args:
        coords (np.ndarray): An array of shape (T, N, 3) containing the coordinates of the beads.

    Returns:
        np.ndarray: A square matrix of shape ((num_beads-1)*(num_beads)/2) containing the distances between the pairs of beads.
    """
    num_timesteps = coords.shape[0]
    num_beads = coords.shape[1]
    distances = np.empty((num_timesteps, int((num_beads-1)*(num_beads)/2)), dtype=np.float32)
    
    for t in range(num_timesteps):
        count = 0
        for i in range(num_beads):
            for j in range(i + 1, num_beads):
                dist = np.linalg.norm(coords[t, i] - coords[t, j])
                distances[t, count] = dist
                count += 1
    
    return distances


def get_per_residue_helicity(traj):
    """
    Calculates the per-residue helicity for a trajectory.

    Args:
        traj (np.ndarray): An array of shape (T, N, 3) containing the coordinates of the beads.

    Returns:
        np.ndarray: An array of shape (N-3,) containing the helicity for each residue trough the trajectory.
    """
    num_timesteps = traj.shape[0]
    num_beads = traj.shape[1]
    
    # Initialize helicity array
    helicity = np.zeros((num_beads), dtype=np.float32)
    
    shifted_traj = np.roll(traj, shift=-3, axis=1)
    helical_distances = np.linalg.norm(traj - shifted_traj, axis=2)[:, :-3]
    helix_mask = np.exp(-(helical_distances - 0.575)**2 / 0.08)
    helicity = np.sum(helix_mask, axis=0) / num_timesteps  # Average over time
    
    
    return helicity, helix_mask
