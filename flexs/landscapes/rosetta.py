import flexs
import numpy as np
import pyrosetta as prs

# Initialize pyrosetta and suppress output messages
prs.init("-mute all")

# Some pyrosetta methods take three letter aa representations
# so we need to convert our single letter representations
aa_single_to_three_letter_code = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "B": "ASX",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "Z": "GLX",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


class RosettaFolding(flexs.Landscape):
    """
    This oracle scores sequences using a fixed conformation design energy.
    In this case, both backbone and side chain conformations are fixed (no repacking).

    In this setting, we have a 3-D structure that we'd like to design for
    (given by the PDB file), so we look for sequences that might stably fold to
    the given conformation.

    The best way to query how well a sequence might fold to a given conformation
    is to run a folding simulation, but since that is so computationally
    intense, it is more common to simply calculate the energy of the sequence
    if it was forced to fold into the target 3-D structure.

    This is just a proxy for folding stability, but it is often an effective
    one and is the approach used by RosettaDesign.

    We use Rosetta's centroid energy function instead of the full-atom one since
    it is less sensitive to switching out residues without repacking side-chain conformations.

    """

    def __init__(self, pdb_file, norm_value=3):
        super().__init__(name="RosettaFolding")

        # We will reuse this pose over and over, mutating it to match
        # whatever sequence we are given to measure.
        # This is necessary since sequence identity can only be mutated
        # one residue at a time in Rosetta, because the atom coords of the
        # backbone of the previous residue are copied into the new one.
        self.pose = prs.pose_from_pdb(pdb_file)
        self.wt_pose = self.pose.clone()

        # Change self.pose from full-atom to centroid representation
        to_centroid_mover = prs.SwitchResidueTypeSetMover("centroid")
        to_centroid_mover.apply(self.pose)

        # Use 1 - sigmoid(centroid energy / norm_value) as the fitness score
        self.score_function = prs.create_score_function("cen_std")
        self.norm_value = norm_value

    def _mutate_pose(self, mut_aa, mut_pos):
        """ This method mutates `self.pose` to contain `mut_aa` at `mut_pos`. """

        current_residue = self.pose.residue(
            mut_pos + 1
        )  # + 1 since rosetta is 1-indexed
        conformation = self.pose.conformation()

        # Get ResidueType for new residue
        new_restype = prs.rosetta.core.pose.get_restype_for_pose(
            self.pose, aa_single_to_three_letter_code[mut_aa]
        )

        # Create the new residue using current_residue backbone
        new_res = prs.rosetta.core.conformation.ResidueFactory.create_residue(
            new_restype,
            current_residue,
            conformation,
            preserve_c_beta=False,
            allow_alternate_backbone_matching=True,
        )

        # Make sure we retain as much info from the previous res as possible
        prs.rosetta.core.conformation.copy_residue_coordinates_and_rebuild_missing_atoms(
            current_residue,
            new_res,
            conformation,
            preserve_only_sidechain_dihedrals=False,
        )

        # Replace residue
        self.pose.replace_residue(mut_pos + 1, new_res, orient_backbone=False)

        # Update the coordinates of atoms that depend on polymer bonds
        conformation.rebuild_polymer_bond_dependent_atoms_this_residue_only(mut_pos + 1)

    def get_folding_energy(self, sequence):
        """ Return rosetta folding energy (delta G) of the given sequence in `self.pose`'s conformation. """

        pose_sequence = self.pose.sequence()

        if len(sequence) != len(pose_sequence):
            raise ValueError(
                "`sequence` must be of the same length as original protein in .pdb file"
            )

        # Mutate `self.pose` where necessary to have the same sequence identity as `sequence`
        for i in range(len(sequence)):
            if sequence[i] != pose_sequence[i]:
                self._mutate_pose(sequence[i], i)

        return self.score_function(self.pose)

    def _fitness_function(self, sequences):
        """ Negate and normalize folding energy to get maximization objective """

        energies = np.array([-self.get_folding_energy(seq) for seq in sequences])
        return energies / self.norm_value
