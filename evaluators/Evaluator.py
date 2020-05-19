import time
import uuid
from pathlib import Path

from models.Ground_truth_oracles.GFP_landscape_models import \
    GFP_landscape_constructor
from models.Ground_truth_oracles.Protein_landscape_models import \
    Protein_landscape_constructor
from models.Ground_truth_oracles.RNA_landscape_models import \
    RNA_landscape_constructor
from models.Ground_truth_oracles.TF_binding_landscape_models import \
    TF_binding_landscape_constructor
from models.Noisy_models.Ensemble import Ensemble_models
from models.Noisy_models.Neural_network_models import NN_model
from models.Noisy_models.Noisy_abstract_model import (Noisy_abstract_model,
                                                      Null_model)
from utils.model_architectures import (NLNN, SKBR, SKGB, SKGP, SKRF, CNNa,
                                       Linear, SKExtraTrees, SKLasso, SKLinear,
                                       SKNeighbors)

LANDSCAPE_TYPES = {
    "RNA": [2],
    "TF": [],
    "Protein": [2],
    "GFP": [],
}  # ["RNA","TF","GFP","ADDITIVE"]
LANDSCAPE_ALPHABET = {
    "RNA": "UCGA",
    "TF": "TCGA",
    "Protein": "ILVAGMFYWEDQNHCRKSTP",
    "GFP": "ILVAGMFYWEDQNHCRKSTP",
}


class Evaluator:
    '''
    Evaluator for explorers 

    Currently, the evaluator supports transcription factor (TF), RNA, Protein, 
    and Green Fluorescent Protein (GFP) landscapes. 
    '''
    def __init__(
        self,
        explorer,
        landscape_types=LANDSCAPE_TYPES,
        path="./simulations/property_evaluation/",
        ML_ensemble=["CNNa", "CNNa", "CNNa"],
        adaptive_ensemble=True,
    ):
        self.explorer = explorer
        self.path = path
        self.verbose = False
        Path(self.path).mkdir(exist_ok=True)

        self.landscape_types = landscape_types
        self.landscape_generator = {}
        self.ML_ensemble = []
        if ML_ensemble:
            self.ML_ensemble = self.load_ensemble(ML_ensemble)
        self.adaptive = adaptive_ensemble
        self.load_landscapes()

    def load_landscapes(self):
        print(
            f'loading landscapes RNA: {self.landscape_types.get("RNA")}, TF: {self.landscape_types.get("TF")}, '
            f'Protein: {self.landscape_types.get("Protein")}, GFP: {self.landscape_types.get("GFP")}'
        )
        if "RNA" in self.landscape_types:
            RNALconstructor = RNA_landscape_constructor()
            RNALconstructor.load_landscapes(
                "../data/RNA_landscapes/RNA_landscape_config.yaml",
                landscapes_to_test=self.landscape_types["RNA"],
            )
            self.landscape_generator[
                "RNA"
            ] = RNALconstructor.generate_from_loaded_landscapes()

        if "TF" in self.landscape_types:
            TFconstructor = TF_binding_landscape_constructor()
            TFconstructor.load_landscapes(landscapes_to_test=self.landscape_types["TF"])
            self.landscape_generator[
                "TF"
            ] = TFconstructor.generate_from_loaded_landscapes()

        if "Protein" in self.landscape_types:
            Protein_constructor = Protein_landscape_constructor()
            Protein_constructor.load_landscapes(
                landscapes_to_test=self.landscape_types["Protein"]
            )
            self.landscape_generator[
                "Protein"
            ] = Protein_constructor.generate_from_loaded_landscapes()

        if "GFP" in self.landscape_types:
            GFP_constructor = GFP_landscape_constructor()
            GFP_constructor.load_landscapes(
                landscapes_to_test=self.landscape_types["GFP"]
            )
            self.landscape_generator[
                "GFP"
            ] = GFP_constructor.generate_from_loaded_landscapes()

        self.explorer.run_id = str(uuid.uuid1())
        print("Loading complete.")

    @staticmethod
    def load_ensemble(ML_ensemble):
        ensemble = []
        for key in ML_ensemble:
            if key == "LNN":
                ensemble.append(Linear)

            elif key == "NLNN":
                ensemble.append(NLNN)

            elif key == "CNNa":
                ensemble.append(CNNa)

            elif key == "Linear":
                ensemble.append(SKLinear)

            elif key == "Lasso":
                ensemble.append(SKLasso)

            elif key == "RF":
                ensemble.append(SKRF)

            elif key == "GB":
                ensemble.append(SKGB)

            elif key == "NE":
                ensemble.append(SKNeighbors)

            elif key == "BR":
                ensemble.append(SKBR)

            elif key == "ExtraTrees":
                ensemble.append(SKExtraTrees)

            elif key == "GP":
                ensemble.append(SKGP)

        return ensemble

    def run_on_null_model(
        self,
        landscape_oracle,
        Null_args,
        start_seq,
        num_batches=10,
        hot_start=False,
        verbose=False,
        overwrite=False,
    ):

        print("Running null ", Null_args)

        if not self.ML_ensemble:
            noisy_landscape = Null_model(landscape_oracle, **Null_args)
            if hot_start:
                pass
            else:
                noisy_landscape.reset([start_seq])
            self.explorer.set_model(noisy_landscape)
            self.explorer.run(num_batches, overwrite, verbose)

        else:
            nnlandscapes = []
            for _ in range(len(self.ML_ensemble)):
                noisy_landscape = Null_model(landscape_oracle, **Null_args)
                nnlandscapes.append(noisy_landscape)

            nam_ensemble_landscape = Ensemble_models(
                nnlandscapes, adaptive=self.adaptive
            )
            nam_ensemble_landscape.reset()
            if hot_start:
                pass
            else:
                nam_ensemble_landscape.update_model([start_seq])
            self.explorer.set_model(nam_ensemble_landscape)
            self.explorer.run(num_batches, overwrite=overwrite, verbose=verbose)

    def run_on_NAM(
        self,
        landscape_oracle,
        NAM_args,
        start_seq,
        num_batches=10,
        hot_start=False,
        verbose=False,
        overwrite=False,
    ):

        print("Running  NAM", NAM_args)

        if not self.ML_ensemble:
            noisy_landscape = Noisy_abstract_model(landscape_oracle, **NAM_args)
            if hot_start:
                pass
            else:
                noisy_landscape.reset([start_seq])
            self.explorer.set_model(noisy_landscape)
            self.explorer.run(num_batches, overwrite=overwrite, verbose=verbose)
        else:
            nnlandscapes = []
            for _ in range(len(self.ML_ensemble)):
                noisy_landscape = Noisy_abstract_model(landscape_oracle, **NAM_args)
                nnlandscapes.append(noisy_landscape)

            nam_ensemble_landscape = Ensemble_models(
                nnlandscapes, adaptive=self.adaptive
            )
            nam_ensemble_landscape.reset()
            if hot_start:
                pass
            else:
                nam_ensemble_landscape.update_model([start_seq])
            self.explorer.set_model(nam_ensemble_landscape)
            self.explorer.run(num_batches, overwrite=overwrite, verbose=verbose)

    def run_on_NNmodel(
        self,
        landscape_oracle,
        NNM_args,
        start_seq,
        num_batches=10,
        hot_start=False,
        verbose=False,
        overwrite=False,
    ):

        print("Running NN", NNM_args)

        if not self.ML_ensemble:
            nnlandscapes = []

            for arch in [SKLinear, SKRF, NLNN, CNNa]:
                nn_model = arch(len(start_seq), alphabet=self.explorer.alphabet)
                nnlandscape = NN_model(landscape_oracle, nn_model, **NNM_args)

                if hot_start:
                    pass
                else:
                    nnlandscape.update_model([start_seq])

                self.explorer.set_model(nnlandscape)
                self.explorer.run(num_batches, overwrite, verbose)

        else:
            nnlandscapes = []
            for arch in self.ML_ensemble:
                nn_model = arch(len(start_seq), alphabet=self.explorer.alphabet)
                nnlandscape = NN_model(landscape_oracle, nn_model, **NNM_args)
                nnlandscapes.append(nnlandscape)

            nn_ensemble_landscape = Ensemble_models(
                nnlandscapes, adaptive=self.adaptive
            )
            nn_ensemble_landscape.reset()
            if hot_start:
                pass
            else:
                nn_ensemble_landscape.update_model([start_seq])

            self.explorer.set_model(nn_ensemble_landscape)
            self.explorer.run(num_batches, overwrite, verbose)

    def evaluate_for_landscapes(self, property_of_interest_evaluator, num_starts=100):
        for landscape_type in self.landscape_types:
            self.explorer.alphabet = LANDSCAPE_ALPHABET[landscape_type]
            for landscape in self.landscape_generator[landscape_type]:
                oracle = landscape["landscape_oracle"]
                landscape_id = landscape["landscape_id"]
                print(f"Running on {landscape_id}")
                starts_per_landscape = 0
                for starting_seq in landscape["starting_seqs"]:
                    start_seq_id = starting_seq
                    start_seq = landscape["starting_seqs"][starting_seq]
                    property_of_interest_evaluator(
                        oracle, start_seq, landscape_id, start_seq_id
                    )
                    starts_per_landscape += 1
                    if starts_per_landscape >= num_starts:
                        break

    def consistency_robustness_independence(
        self, oracle, start_seq, landscape_id, start_seq_id
    ):
        '''
        Evaluate explorer on NAM model using a variety of noise levels. 
        '''
        Path(self.path + "consistency_robustness_independence/").mkdir(exist_ok=True)
        self.explorer.path = self.path + "consistency_robustness_independence/"

        print(f"start seq {start_seq_id}")

        for ss in [0, 0.5, 0.9, 1]:
            print(f"Evaluating for signal_strength: {ss}")
            landscape_idents = {
                "landscape_id": landscape_id,
                "start_id": start_seq_id,
                "signal_strength": ss,
            }
            self.run_on_NAM(oracle, landscape_idents, start_seq, verbose=True)
        landscape_idents = {"landscape_id": landscape_id, "start_id": start_seq_id}
        self.run_on_NNmodel(oracle, landscape_idents, start_seq, verbose=True)
        self.run_on_null_model(oracle, landscape_idents, start_seq, verbose=True)

    def efficiency(self, oracle, start_seq, landscape_id, start_seq_id):
        Path(self.path + "efficiency/").mkdir(exist_ok=True)
        self.explorer.path = self.path + "efficiency/"

        print(f"start seq {start_seq_id}")

        landscape_idents = {
            "landscape_id": landscape_id,
            "start_id": start_seq_id,
            "signal_strength": 1,
        }
        self.explorer.batch_size = 100
        for virtual_screen in [1, 10, 100, 1000]:
            print(f"Evaluating for virtual_screen: {virtual_screen}")
            self.explorer.virtual_screen = virtual_screen
            self.run_on_NAM(oracle, landscape_idents, start_seq)

    def adaptivity(self, oracle, start_seq, landscape_id, start_seq_id):
        Path(self.path + "adaptivity/").mkdir(exist_ok=True)
        self.explorer.path = self.path + "adaptivity/"

        print(f"start seq {start_seq_id}")
        landscape_idents = {"landscape_id": landscape_id, "start_id": start_seq_id}
        self.ML_ensemble = self.load_ensemble(["Linear", "RF", "CNNa"])
        for num_batches in [1, 10, 100, 1000]:
            print(f"Evaluating for num_batches: {num_batches}")
            self.explorer.batch_size = int(1000 / num_batches)
            self.explorer.virtual_screen = 20
            self.run_on_NNmodel(
                oracle,
                landscape_idents,
                start_seq,
                num_batches=num_batches,
                verbose=True,
            )

    def scalability(self, oracle, start_seq, landscape_id, start_seq_id):
        Path(self.path + "scalability/").mkdir(exist_ok=True)
        self.explorer.path = self.path + "scalability/"

        self.explorer.debug = True
        print(f"start seq {start_seq_id}")
        landscape_idents = {
            "landscape_id": landscape_id,
            "start_id": start_seq_id,
            "signal_strength": 1,
        }
        with open(
            self.explorer.path + f"times{landscape_id}_{self.explorer.run_id}.csv", "a"
        ) as outfile:
            out_string = "landscape_id,batch_size,virtual_screen,time_3xrounds\n"
            outfile.write(out_string)
            for batch_size in [100, 1000]:
                for virtual_screen in [1, 10, 100]:
                    print(
                        f"Evaluating for virtual_screen: {virtual_screen}, batch_size: {batch_size}"
                    )
                    self.explorer.batch_size = batch_size
                    self.explorer.virtual_screen = virtual_screen
                    start = time.time()
                    self.run_on_NAM(oracle, landscape_idents, start_seq, num_batches=3)
                    end = time.time()
                    time_needed = end - start
                    out_string = (
                        f"{landscape_id},{batch_size},{virtual_screen},{time_needed}\n"
                    )
                    print(out_string)
                    outfile.write(out_string)
