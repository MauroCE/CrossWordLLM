import torch


class ReverseGPTConfig:
    """Settings for ReverseGPT (next-token-prediction)."""
    def __init__(self):
        self.seq_len: int = 10  # length of sequence to order
        self.batch_size: int = 64
        self.context_size: int = 256
        self.n_emb: int = 100  # each head is 100//4 = 25 dimensional, which is smaller than standard
        self.num_layers: int = 4
        self.num_heads: int = 4
        self.dropout_prop: float = 0.2  # 20% of neurons are dropped out
        self.device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.vocabulary_size: int = 28
        self.vocabulary: tuple = (".", ":", "a", "b", "c",
                                  "d", "e", "f", "g", "h",
                                  "i", "j", "k", "l", "m",
                                  "n", "o", "p", "q", "r",
                                  "s", "t", "u", "v", "w",
                                  "x", "y", "z")
        self.str_to_int: dict = {character: integer for integer, character in enumerate(self.vocabulary)}
        self.int_to_str = {integer: character for integer, character in enumerate(self.vocabulary)}
        self.str2int = lambda string: [self.str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([self.int_to_str[integer] for integer in int_list])  # list(int) to str


class MultiTokenConfig:
    """Settings for ReverseGPT (next-token-prediction)."""
    def __init__(self):
        self.seq_len: int = 10  # length of sequence to order
        self.batch_size: int = 64
        self.context_size: int = 256
        self.n_emb: int = 100  # each head is 100//4 = 25 dimensional, which is smaller than standard
        self.num_layers: int = 2
        self.num_heads: int = 4
        self.dropout_prop: float = 0.2  # 20% of neurons are dropped out
        self.device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.vocabulary_size: int = 28
        self.n_future_tokens = 2
        self.return_all_heads = True
        self.vocabulary: tuple = (".", ":", "a", "b", "c",
                                  "d", "e", "f", "g", "h",
                                  "i", "j", "k", "l", "m",
                                  "n", "o", "p", "q", "r",
                                  "s", "t", "u", "v", "w",
                                  "x", "y", "z")
        self.str_to_int: dict = {character: integer for integer, character in enumerate(self.vocabulary)}
        self.int_to_str = {integer: character for integer, character in enumerate(self.vocabulary)}
        self.str2int = lambda string: [self.str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([self.int_to_str[integer] for integer in int_list])  # list(int) to str
