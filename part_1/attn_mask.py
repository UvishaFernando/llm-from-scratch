import torch

def causal_mask(T: int, device=None):
    """
    Create a no-cheating mask so each token can only look at itself and previous tokens,
    never future tokens
    """
    m = torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, T, T)