from config import MultiTokenConfig, ReverseGPTConfig
from reversing_model import ReverseGPT, MultiTokenGPT

if __name__ == "__main__":
    alphabet = "abcdefghijklmnopqrstuvwxyz:."
    vocabulary = sorted(list(set(alphabet)))
    vocab_size = len(vocabulary)

    # Config for next token-prediction
    config_nt = ReverseGPTConfig()
    config_nt.device = 'cpu'
    config_nt.seq_len = 10
    config_nt.context_size = 2 * config_nt.seq_len + 1
    config_nt.vocabulary = vocabulary
    config_nt.vocabulary_size = vocab_size

    # Config for multi-token prediction
    config_mt = MultiTokenConfig()
    config_mt.device = 'cpu'
    config_mt.seq_len = 10
    config_mt.context_size = 2 * config_mt.seq_len + 1
    config_mt.vocabulary = vocabulary
    config_mt.vocabulary_size = vocab_size

    next_token = ReverseGPT(config_nt)
    multi_token = MultiTokenGPT(config_mt)
    print("Next token: ", next_token.num_parameters())
    print("Multi token: ", multi_token.num_parameters())



