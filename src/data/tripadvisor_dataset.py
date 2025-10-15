import re
from collections import Counter
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset
from nltk.corpus import stopwords

from my_tokenizers.base import BaseTokenizer

stop_words = set(stopwords.words("english"))


def extract_keywords(text: str, max_keywords: int = 8) -> str:
    tokens = re.findall(r"\b\w+\b", text.lower())
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    freq = Counter(tokens)
    keywords = [w for w, _ in freq.most_common(max_keywords)]
    return ", ".join(keywords)


def causal_mask(size: int) -> Tensor:
    # To keep only the previous context and mask the word swe haven't seen yet.
    return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int) == 0


class TripAdvisorDataset(Dataset):
    """For generative models."""

    def __init__(
        self,
        texts: List[str],
        ratings: List[int],
        tokenizer: BaseTokenizer,
        max_input_len: int = 32,
        max_target_len: int = 256,
    ) -> None:
        self.texts: List[str] = texts
        self.ratings: List[int] = ratings
        self.tokenizer: BaseTokenizer = tokenizer
        self.max_input_len: int = max_input_len
        self.max_target_len: int = max_target_len

        self.sos: Tensor = torch.tensor(
            [self.tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.eos: Tensor = torch.tensor(
            [self.tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )

        self.pad: Tensor = torch.tensor(
            [self.tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        text = self.texts[idx]
        rating = self.ratings[idx]

        keywords = extract_keywords(text)
        prompt = f"{rating}: {keywords}"

        source_tokens = self.tokenizer.encode(prompt)[: self.max_input_len]
        target_tokens = self.tokenizer.encode(text)[: (self.max_target_len - 1)]

        source_padding_len = self.max_target_len - len(source_tokens) - 2
        target_padding_len = self.max_target_len - len(target_tokens) - 1

        if source_padding_len < 0 or target_padding_len < 0:
            raise ValueError("Sentence length error")

        # Adding SOS and EOS to the encoder input.
        encoder_input: Tensor = torch.cat(
            [
                self.sos,
                torch.tensor(source_tokens, dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * source_padding_len, dtype=torch.int64),
            ]
        )

        # Adding SOS to the decoder input.
        decoder_input: Tensor = torch.cat(
            [
                self.sos,
                torch.tensor(target_tokens, dtype=torch.int64),
                torch.tensor([self.pad] * target_padding_len, dtype=torch.int64),
            ]
        )

        # Adding EOS to the decoder output.
        label: Tensor = torch.cat(
            [
                torch.tensor(target_tokens, dtype=torch.int64),
                self.eos,
                torch.tensor([self.pad] * target_padding_len, dtype=torch.int64),
            ]
        )

        assert encoder_input.size(0) == self.max_target_len
        assert decoder_input.size(0) == self.max_target_len
        assert label.size(0) == self.max_target_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad).unsqueeze(0).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)
            ),  # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label": label,  #  (seq_len)
            "source_text": prompt,
            "target_text": text,
        }
