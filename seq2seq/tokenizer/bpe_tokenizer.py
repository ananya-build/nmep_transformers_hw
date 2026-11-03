from .tokenizer import Tokenizer

import torch
from transformers import AutoTokenizer


class BPETokenizer(Tokenizer):
    def __init__(self, model: str = "bert-base-multilingual-cased", verbose: bool = False):
        """
        Initializes the BPETokenizer class for French to English translation.

        Uses a pretrained BPE tokenizer to encode and decode text.
        """

        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )

        self.vocab = self.tokenizer.get_vocab()

        self.pad_token = self.tokenizer.pad_token
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.encode(
                text, add_special_tokens=False
            ),
            dtype=torch.long,
        )

    def decode(self, tokens: torch.Tensor) -> str:
        return self.tokenizer.decode(tokens.tolist())
