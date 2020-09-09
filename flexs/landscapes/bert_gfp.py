import os

import numpy as np
import pandas as pd
import requests
import tape
import torch

import flexs


class BertGFPBrightness(flexs.Landscape):
    def __init__(self, norm_value=1):
        """
        Green fluorescent protein (GFP) lanscape. The oracle used in this lanscape is
        the transformer model from TAPE (https://github.com/songlab-cal/tape).

        To create the transformer model used here, run the command:

        tape-train transformer fluorescence --from_pretrained bert-base --batch_size 128 --gradient_accumulation_steps 10 --data_dir .
        """
        super().__init__(name="GFP")

        # Download GFP model weights and config info
        if not os.path.exists("fluorescence-model"):
            os.mkdir("fluorescence-model")

            # URL for BERT GFP fluorescence model
            gfp_model_path = "https://fluorescence-model.s3.amazonaws.com/fluorescence_transformer_20-05-25-03-49-06_184764/"
            for file_name in [
                "args.json",
                "checkpoint.bin",
                "config.json",
                "pytorch_model.bin",
            ]:
                print("Downloading", file_name)
                response = requests.get(gfp_model_path + file_name)
                with open(f"fluorescence-model/{file_name}", "wb") as f:
                    f.write(response.content)

        # self.GFP_info = dict(zip(self.GFP_df["seq"], self.GFP_df["brightness"]))
        self.tokenizer = tape.TAPETokenizer(vocab="iupac")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = tape.ProteinBertForValuePrediction.from_pretrained(
            "fluorescence-model"
        ).to(self.device)

    def _fitness_function(self, sequences):
        sequences = np.array(sequences)

        encoded_seqs = torch.tensor(
            [self.tokenizer.encode(seq) for seq in sequences]
        ).to(self.device)

        return self.model(encoded_seqs)[0].detach().numpy().astype(float).reshape(-1)
