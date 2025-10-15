import logging
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from .base import BaseTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BpeTokenizer(BaseTokenizer):
    def __init__(
        self,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None,
        unk_token: str = "[UNK]",
        **trainer_kwargs,
    ) -> None:
        self.tokenizer = Tokenizer(BPE(unk_token=unk_token))
        self.tokenizer.pre_tokenizer = ByteLevel()
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
            **trainer_kwargs,
        )

    def fit(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on a list of texts.

        :param List[str] texts: List of input texts to train the tokenizer
        """
        self.tokenizer.train_from_iterator(texts, self.trainer)

        # Add post-processing to handle special tokens properly
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ("[SOS]", self.token_to_id("[SOS]")),
                ("[EOS]", self.token_to_id("[EOS]")),
            ],
        )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a single text string into a list of tokens.

        :param str text: Input text to tokenize
        :return List[str]: List of token strings
        """
        enc = self.tokenizer.encode(text)
        return enc.tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode a single text string into a list of token IDs.

        :param str text: Input text to encode
        :return List[int]: List of token IDs
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs into a single text string.

        :param List[int] tokens: List of token IDs to decode
        :return str: Decoded text string
        """
        # Filter out special tokens
        special_token_ids = [
            self.token_to_id("[PAD]"),
            self.token_to_id("[SOS]"),
            self.token_to_id("[EOS]"),
            self.token_to_id("[UNK]"),
        ]

        # Remove special tokens before decoding
        filtered_tokens = [token for token in tokens if token not in special_token_ids]

        # If no tokens left after filtering, return empty string
        if not filtered_tokens:
            return ""

        # Use the tokenizer's built-in decoder
        try:
            decoded = self.tokenizer.decode(filtered_tokens)
            # Clean up any spacing issues
            decoded = " ".join(decoded.split())
            return decoded
        except Exception as e:
            logger.error(f"Error decoding tokens: {e}")
            logger.error(f"Tokens: {filtered_tokens}")
            return ""

    def token_to_id(self, token: str) -> Optional[int]:
        """
        Convert a token string to its corresponding token ID.

        :param str token: Token string to convert
        :return Optional[int]: Corresponding token ID or None if not found
        """
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """
        Convert a token ID back to its corresponding token string.

        :param int token_id: Token ID to convert
        :return str: Corresponding token string
        """
        return self.tokenizer.id_to_token(token_id)

    def save(self, path: str) -> None:
        """
        Save the tokenizer to a file.

        :param str path: File path to save the tokenizer
        """
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        """
        Load the tokenizer from a file.

        :param str path: File path to load the tokenizer from
        """
        self.tokenizer = Tokenizer.from_file(path)
