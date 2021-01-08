"""Defines the RosettaFolding landscape and problem registry."""
import os
from typing import Dict

import numpy as np
import torch

import flexs
from flexs.types import SEQUENCES_TYPE

# Pyrosetta is an optional dependency
try:
    import pyrosetta as prs
except ImportError:
    pass

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
    it is less sensitive to switching out residues without repacking side-chain
    conformations.

    We convert these energies to a maximization objective in the 0-1 scale by
    fitness = (-energy - `sigmoid_center`) / `sigmoid_norm_value`.

    Attributes:
        wt_pose: The original PyRosetta pose object from the .pdb file.
            Call `wt_pose.sequence()` to get the wild type sequence.

    """

    def __init__(self, pdb_file: str, sigmoid_center: float, sigmoid_norm_value: float):
        """
        Create a RosettaFolding landscape from a .pdb file with structure.

        Args:
            pdb_file: Path to .pdb file with structure information.
            sigmoid_center: Center of sigmoid function.
            sigmoid_norm_value: 1 / scale of sigmoid function.

        """
        super().__init__(name="RosettaFolding")

        # Inform the user if pyrosetta is not available.
        try:
            prs
        except NameError as e:
            raise ImportError(
                "Error: Pyrosetta not installed. "
                "Source, binary, and conda installations available "
                "at http://www.pyrosetta.org/dow"
            ) from e

        # Initialize pyrosetta and suppress output messages
        prs.init("-mute all")

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
        self.sigmoid_center = sigmoid_center
        self.sigmoid_norm_value = sigmoid_norm_value

    def _mutate_pose(self, mut_aa: str, mut_pos: int):
        """Mutate `self.pose` to contain `mut_aa` at `mut_pos`."""
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

        # Make sure we retain as much info from the previous resdiue as possible
        prs.rosetta.core.conformation.copy_residue_coordinates_and_rebuild_missing_atoms(  # noqa: E501
            current_residue,
            new_res,
            conformation,
            preserve_only_sidechain_dihedrals=False,
        )

        # Replace residue
        self.pose.replace_residue(mut_pos + 1, new_res, orient_backbone=False)

        # Update the coordinates of atoms that depend on polymer bonds
        conformation.rebuild_polymer_bond_dependent_atoms_this_residue_only(mut_pos + 1)

    def get_folding_energy(self, sequence: str):
        """
        Return rosetta folding energy of the given sequence in
        `self.pose`'s conformation.

        """
        pose_sequence = self.pose.sequence()

        if len(sequence) != len(pose_sequence):
            raise ValueError(
                "`sequence` must be of the same length as original protein in .pdb file"
            )

        # Mutate `self.pose` where necessary to have the same sequence identity as
        # `sequence`
        for i, aa in enumerate(sequence):
            if aa != pose_sequence[i]:
                self._mutate_pose(aa, i)

        return self.score_function(self.pose)

    def _fitness_function(self, sequences: SEQUENCES_TYPE) -> np.ndarray:
        """Negate and normalize folding energy to get maximization objective"""
        energies = torch.tensor([self.get_folding_energy(seq) for seq in sequences])
        scaled_energies = (-energies - self.sigmoid_center) / self.sigmoid_norm_value
        return torch.sigmoid(scaled_energies).numpy()


def registry() -> Dict[str, Dict]:
    """
    Return a dictionary of problems of the form:
    `{
        "problem name": {
            "params": ...,
        },
        ...
    }`

    where `flexs.landscapes.RosettaFolding(**problem["params"])` instantiates the
    rosetta folding landscape for the given set of parameters.

    Returns:
        dict: Problems in the registry.

    """
    rosetta_data_dir = os.path.join(os.path.dirname(__file__), "data/rosetta")

    return {
        "3msi": {
            "params": {
                "pdb_file": f"{rosetta_data_dir}/3msi.pdb",
                "sigmoid_center": -3,
                "sigmoid_norm_value": 12,
            },
            "starts": {
                "ed_3_wt": "MAQASVVANQLIPINTHLTLVMMRSEVVTYVHIPAEDIPRLVSMDVNRAVPLGTTLMPDMVKGYAA",  # noqa: E501
                "ed_5_wt": "MAQASVVFNQLIPINTHLTLVMMRFEVVTPVGCPAMDIPRLVSQQVNRAVPLGTTLMPDMVKGYAA",  # noqa: E501
                "ed_7_wt": "WAQRSVVANQLIPINTGLTLVMMRSELVTGVGAPAEDIPRLVSMQVNRAVPLGTTNMPDMVKGYAA",  # noqa: E501
                "ed_12_wt": "RAQESVVANQLIPILTHLTQKMSRRFVVTPVGIPAEDIPRLVNAQVDRAVPLGTTLMPDMDKGYAA",  # noqa: E501
                "ed_27_wt": "MRRYSVIAYQERPINLHSTLTFNRSEVPWPVNRPASDAPRLVSMQNNRSVPLGTKLPEDPVCRYAL",  # noqa: E501
            },
        },
        "3mx7": {
            "params": {
                "pdb_file": f"{rosetta_data_dir}/3mx7.pdb",
                "sigmoid_center": -3,
                "sigmoid_norm_value": 12,
            },
            "starts": {
                "ed_2_wt": "MTDLVAVWDVALSDGHHKIEFEHGTTSGKRVVYVDGKESIRKEWMFKLVGKETFYVGAAKTKATINIDAISGFAYEYTLEINGKSLKKYM",  # noqa: E501
                "ed_5_wt": "MTDLVAVWFYALSDGVHKIEFEHGTTSGKRVVYVDGKEEIRKEWMFKLVGKETFYVGAAKTKATINIWAISGFAIEYTLTINGKSLKKYM",  # noqa: E501
                "ed_7_wt": "MTDLVAYWDVANSDGVHKISFEHGTTSGKRVVYVDGKEEIRKEGMFKLVGRETFYVGAAKTKATINIDAGSGFAYEYTLEINGKVLKKYM",  # noqa: E501
                "ed_13_wt": "VTDKSAVWDVALSDGVHKIEFEHGTTSIKRVVYVQGKEENRKEWQFKGVGKETFYVGAAKRKATINIDAKSGFAYEVTLEINQKSLKQYM",  # noqa: E501
                "ed_29_wt": "STDLVEVMRIACSDGVHKIEFEHGTTSGMRVHYKDLKEEGRKPHRFKLEGNFQWYENCHKTKAIINITAIMGFAYWYFLEWNGKSLKKYM",  # noqa: E501
            },
        },
    }
