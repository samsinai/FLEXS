import numpy as np
import pandas as pd 
import random
import yaml
from glob import glob
import pyrosetta as prs
from meta.model import Ground_truth_oracle
import os
from utils.multi_dimensional_model import Multi_dimensional_model


# Initialize pyrosetta and suppress output messages
prs.init('-mute all')

# Some pyrosetta methods take three letter aa representations
# so we need to convert our single letter representations
aa_single_to_three_letter_code = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'B': 'ASX',
    'C': 'CYS',
    'E': 'GLU',
    'Q': 'GLN',
    'Z': 'GLX',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}

class Protein_landscape_constructor:
    def __init__(self):
        self.loaded_landscapes = []
        self.starting_seqs = {}

    def load_landscapes(self, data_path="../data/Protein_landscapes", landscapes_to_test=[]):
        df_info = pd.read_csv(f"{data_path}/Protein_landscape_id.csv", sep=",")
        start_seqs, target_seqs = {}, {}
        for _, row in df_info.iterrows():
            if row['id'] not in landscapes_to_test:
                continue 

            pdb_file = data_path + '/folding_landscapes/' + row['file_name']
            native_pose = prs.pose_from_pdb(pdb_file)
            seq = native_pose.sequence()

            self.starting_seqs[row['id']] = 'A' * len(seq)
            self.loaded_landscapes.append({
                'landscape_id': row['id'],
                'targets': [seq],
                'pdb_file': [pdb_file]
            })

        print(f"{len(self.loaded_landscapes)} Protein landscapes loaded.")

    def construct_landscape_object(self, landscape_dict):
        landscape_id = landscape_dict['landscape_id']
        landscapes = []
        for pdb_file in landscape_dict["pdb_file"]:
            landscape = Protein_landscape_folding_energy(pdb_file)
            landscapes.append(landscape)

        if len(landscapes) > 1:
            from functools import reduce

            combined_func = lambda z: reduce(
                lambda x, y: x * y, z
            )  # currently multiplicative landscapes
            composite_landscape = Multi_dimensional_model(
                landscapes, combined_func=combined_func
            )

        else:
            return {
                "landscape_id": landscape_id,
                "starting_seqs": self.starting_seqs,
                "landscape_oracle": landscapes[0],
            }

        return {
            "landscape_id": landscape_id,
            "starting_seqs": self.starting_seqs,
            "landscape_oracle": composite_landscape,
        }

    def generate_from_loaded_landscapes(self):

        for landscape in self.loaded_landscapes:

            yield self.construct_landscape_object(landscape)


class Protein_landscape_folding_energy(Ground_truth_oracle):
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

    """

    def __init__(self, pdb_file, norm_value=10000):
        # We will reuse this pose over and over, mutating it to match
        # whatever sequence we are given to measure.
        # This is necessary since sequence identity can only be mutated
        # one residue at a time in Rosetta, because the atom coords of the
        # backbone of the previous residue are copied into the new one.
        self.pose = prs.pose_from_pdb(pdb_file)

        # Use 1 - sigmoid(full-atom energy / norm_value) as the fitness score
        self.scorefxn = prs.get_fa_scorefxn()
        self.norm_value = norm_value

        # Reset every residue in protein to a new one to unpack the backbone
        # and get more constent energies
        for i, wt_aa in enumerate(self.pose.sequence()):
            new_aa = 'A' if wt_aa != 'A' else 'C'
            self._mutate_pose(new_aa, i + 1) # + 1 since rosetta is 1-indexed
            self._mutate_pose(wt_aa, i + 1)

    def _mutate_pose(self,
                     mut_aa,
                     mut_pos,
                     do_not_preserve_atom_coords=False,
                     update_polymer_dependents=True,
                     orient_backbone=False):
        
        current_residue = self.pose.residue(mut_pos)
        conformation = self.pose.conformation()
        
        # Get ResidueType for new residue
        new_restype = prs.rosetta.core.pose.get_restype_for_pose(self.pose,
                                                                 aa_single_to_three_letter_code[mut_aa])

        # Create the new residue and replace it, using current_residue backbone
        new_res = prs.rosetta.core.conformation.ResidueFactory.create_residue(
            new_restype, current_residue, conformation)

        # Make sure we retain as much info from the previous res as possible
        prs.rosetta.core.conformation.copy_residue_coordinates_and_rebuild_missing_atoms(
            current_residue, new_res, conformation, do_not_preserve_atom_coords)
        self.pose.replace_residue(mut_pos, new_res, orient_backbone)

        # Update the coordinates of atoms that depend on polymer bonds
        if update_polymer_dependents:
            conformation.rebuild_polymer_bond_dependent_atoms_this_residue_only(mut_pos)

    def _mutate_pose_and_repack(self,
                                mut_aa,
                                mut_pos,
                                pack_radius=10):

        """
        Replaces the residue at `mutant_position` in `pose` with `mutant_aa`
        and repack any residues within `pack_radius` Angstroms of the mutating
        residue's center (nbr_atom) using `pack_scorefxn`.

        Examples
        --------
        >>> mutate_residue(pose, 30, A)

        See also:
            Pose
            PackRotamersMover
            MutateResidue
            pose_from_sequence

        """

        #### a MutateResidue Mover exists similar to this except it does not pack
        ####    the area around the mutant residue (no pack_radius feature)
        #mutator = MutateResidue( mutant_position , mutant_aa )
        #mutator.apply( test_pose )

        task = prs.standard_packer_task(self.pose)

        # Mutation is performed by using a PackerTask with only the mutant
        #    amino acid available during design.
        # To do this, construct a Vector1 of booleans indicating which amino acid
        #    (by its numerical designation) to allow.
        aa_bool = prs.rosetta.utility.vector1_bool()
        mutant_aa_index = prs.rosetta.core.chemical.aa_from_oneletter_code(mut_aa)
        for i in range(1 , 21):
            aa_bool.append(i == mutant_aa_index)

        # Modify the mutating residue's assignment in the PackerTask using the
        #    Vector1 of booleans across the proteogenic amino acids
        task.nonconst_residue_task(mut_pos).restrict_absent_canonical_aas(aa_bool)

        # Only pack the mutating residue and any within the pack_radius.
        # Prevent residues from packing by setting the per-residue "options" of
        #    the PackerTask.
        center = self.pose.residue(mut_pos).nbr_atom_xyz()
        for i in range(1 , self.pose.total_residue() + 1):
            if center.distance_squared(self.pose.residue(i).nbr_atom_xyz()) > pack_radius**2:
                task.nonconst_residue_task(i).prevent_repacking()

        # Apply the mutation and pack nearby residues
        packer = prs.rosetta.protocols.minimization_packing.PackRotamersMover(self.scorefxn, task)
        packer.apply(self.pose)

    def _sigmoid(self, x):
        return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

    def get_folding_energy(self, sequence):
        pose_sequence = self.pose.sequence()

        if len(sequence) != len(pose_sequence):
            raise ValueError('`sequence` must be of the same length as original protein in .pdb file')

        # Mutate `self.pose` where necessary to have the same sequence identity as `sequence`
        for i in range(len(sequence)):
            if sequence[i] != pose_sequence[i]:
                self._mutate_pose(sequence[i], i + 1)

        return self.scorefxn(self.pose)

    def get_fitness(self, sequence):
        folding_energy = self.get_folding_energy(sequence)
        return 1 - self._sigmoid(folding_energy / self.norm_value)
