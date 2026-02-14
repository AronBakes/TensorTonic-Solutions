import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        # self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        next_id = 4

        for text in texts:
            for word in text.split():           # or word.lower() if case-insensitive wanted
                if word not in self.word_to_id:
                    self.word_to_id[word] = next_id
                    self.id_to_word[next_id] = word
                    next_id += 1
        
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = []
        for word in text.split():
            token_id = self.word_to_id.get(word, 1)   # 1 = <UNK>
            tokens.append(token_id)
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for token_id in ids:
            word = self.id_to_word.get(token_id, self.unk_token)   # ← use id→word dict
            words.append(word)
        
        return " ".join(words)
