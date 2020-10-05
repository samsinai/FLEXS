import shutil
import warnings

import flexs
from flexs.utils import sequence_utils as s_utils


def test_additive_aav_packaging():
    problem = flexs.landscapes.additive_aav_packaging.registry()["heart"]
    landscape = flexs.landscapes.AdditiveAAVPackaging(**problem["params"])

    test_seqs = s_utils.generate_random_sequences(90, 100, s_utils.AAS)
    landscape.get_fitness(test_seqs)


def test_rna():
    # Since ViennaRNA is an optional dependency, only test if installed
    try:
        problem = flexs.landscapes.rna.registry()["C20_L100_RNA1+2"]
        landscape = flexs.landscapes.RNABinding(**problem["params"])

        test_seqs = s_utils.generate_random_sequences(100, 100, s_utils.RNAA)
        landscape.get_fitness(test_seqs)

    except ImportError:
        warnings.warn(
            "Skipping RNABinding landscape test since" "ViennaRNA not installed."
        )


def test_rosetta():
    # Since PyRosetta is an optional dependency, only test if installed
    try:
        problem = flexs.landscapes.rosetta.registry()["3msi"]
        landscape = flexs.landscapes.RosettaFolding(**problem["params"])

        seq_length = len(landscape.wt_pose.sequence())
        test_seqs = s_utils.generate_random_sequences(seq_length, 100, s_utils.AAS)
        landscape.get_fitness(test_seqs)

    except ImportError:
        warnings.warn(
            "Skipping RosettaFolding landscape test since PyRosetta not installed."
        )


def test_tf_binding():
    problem = flexs.landscapes.tf_binding.registry()["SIX6_REF_R1"]
    landscape = flexs.landscapes.TFBinding(**problem["params"])

    test_seqs = s_utils.generate_random_sequences(8, 100, s_utils.DNAA)
    landscape.get_fitness(test_seqs)


# TODO: This test takes too long for github actions. Needs further investigation.
"""
def test_bert_gfp():
    landscape = flexs.landscapes.BertGFPBrightness()

    seq_length = len(landscape.gfp_wt_sequence)
    test_seqs = s_utils.generate_random_sequences(seq_length, 100, s_utils.AAS)
    landscape.get_fitness(test_seqs)

    # Clean up downloaded model
    shutil.rmtree("fluorescence-model")
"""
