from openmm import app
import openmm as mm

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