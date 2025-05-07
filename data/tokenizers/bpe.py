from typing import List, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from .base import BaseTokenizer


class BpeTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "[UNK]",
        **trainer_kwargs
    ) -> None:

        self.tokenizer = Tokenizer(BPE(unk_token=unk_token))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = ByteLevelDecoder()
        if special_tokens is None:
            special_tokens = [
                unk_token,
                "[PAD]",
                "[SOS]",
                "[EOS]",
            ]
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            **trainer_kwargs
        )

    def fit(self, texts: List[str]) -> None:
        self.tokenizer.train_from_iterator(texts, self.trainer)

    def encode(self, text: str) -> List[int]:
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)
