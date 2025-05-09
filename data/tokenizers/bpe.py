from typing import List, Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing
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
        #self.tokenizer.decoder = ByteLevelDecoder()
        
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

        # Add post-processing to handle special tokens properly
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ("[SOS]", self.token_to_id("[SOS]")),
                ("[EOS]", self.token_to_id("[EOS]")),
            ],
        )


    def encode(self, text: str) -> List[int]:
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    # def decode(self, tokens: List[int]) -> str:
    #     # Decode with spaces between tokens
    #     decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
    #     return decoded
    
    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into text, properly handling special tokens.
        """
        # Filter out special tokens
        special_token_ids = [
            self.token_to_id("[PAD]"), 
            self.token_to_id("[SOS]"), 
            self.token_to_id("[EOS]"),
            self.token_to_id("[UNK]")
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
            decoded = ' '.join(decoded.split())
            return decoded
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            print(f"Tokens: {filtered_tokens}")
            return ""


    def token_to_id(self, token: str) -> Optional[int]:
        return self.tokenizer.token_to_id(token)
    
    def id_to_token(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id)

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)
