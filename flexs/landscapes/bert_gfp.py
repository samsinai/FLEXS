"""Defines the BertGFPBrightness landscape."""
import os

import numpy as np
import requests
import tape
import torch

import flexs


class BertGFPBrightness(flexs.Landscape):
    r"""
    Green fluorescent protein (GFP) brightness landscape.

    The oracle used in this lanscape is the transformer model
    from TAPE (https://github.com/songlab-cal/tape).

    To create the transformer model used here, run the command:

        ```tape-train transformer fluorescence --from_pretrained bert-base \
                                               --batch_size 128 \
                                               --gradient_accumulation_steps 10 \
                                               --data_dir .```

    Note that the output of this landscape is not normalized to be between 0 and 1.

    Attributes:
        gfp_wt_sequence (str): Wild-type sequence for jellyfish
            green fluorescence protein.
        starts (dict): A dictionary of starting sequences at different edit distances
            from wild-type with different difficulties of optimization.

    """

    gfp_wt_sequence = (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVT"
        "TLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIE"
        "LKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNT"
        "PIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"
    )

    starts = {
        "ed_10_wt": "MSKGEVLFTGVVPILVEMDGDVNGHKFSVSGEGEGDATYGKLTTKFTCTTGKLPVPWPTKVTTLSYRVQCFSRYPDVMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVQFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNIKRDCMVLLEFVTAAGITHGMDELYK",  # noqa: E501
        "ed_18_wt": "MSKGEHLFTGVVPILVELDGDVNGKKFSVSGEGQGDATYGKLTLKFICTTAKVHVPWCTLVTTLSYGVQCFSRYPDHMKQHDFFKGAMPEGYVQERTIFFKDIGNYKLRAEVKFEGDTLVNRIELKGIDFKEDGNIHGHKLEYNYNSQNVYIMASKQKNGIKVNFKIRLNIEDGSVQLAEHYQVNTPIGDFPVLLPDNHKLSAQSADSKDPNEKRDHMHLLEFVTAVGITHGMDELYK",  # noqa: E501
        "ed_31_wt": "MSKGEELFSGVQPILVELDGCVNGHKFSVSGEGEIDATYGKLTLKFICTTWKLPMPWPCLVTFGSYGVQCFSRYRDHPKQHDFFKSAVPEGYVQERTIFMKDDLLYKTRAEVKFEGLTLVNRIELKGKDFKEDGNILGHKLEYNYNSHCVYPMADWNKNWIKVNSKIRLPIEDGSVILADHYQQNTPIGDQPVLLPENHYLSTQSALSKDPEEKGDLMVLLEFVTAAGITHGMDELYK",  # noqa: E501
    }

    def __init__(self):
        """
        Create GFP landscape.

        Downloads model into `./fluorescence-model` if not already cached there.
        If interrupted during download, may have to delete this folder and try again.
        """
        super().__init__(name="GFP")

        # Download GFP model weights and config info
        if not os.path.exists("fluorescence-model"):
            os.mkdir("fluorescence-model")

            # URL for BERT GFP fluorescence model
            gfp_model_path = "https://fluorescence-model.s3.amazonaws.com/fluorescence_transformer_20-05-25-03-49-06_184764/"  # noqa: E501
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

        self.tokenizer = tape.TAPETokenizer(vocab="iupac")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = tape.ProteinBertForValuePrediction.from_pretrained(
            "fluorescence-model"
        ).to(self.device)

    def _fitness_function(self, sequences):
        sequences = np.array(sequences)
        scores = []

        # Score sequences in batches of size 32
        for subset in np.array_split(sequences, max(1, len(sequences) // 32)):
            encoded_seqs = torch.tensor(
                [self.tokenizer.encode(seq) for seq in subset]
            ).to(self.device)

            scores.append(
                self.model(encoded_seqs)[0].detach().numpy().astype(float).reshape(-1)
            )

        return np.concatenate(scores)
