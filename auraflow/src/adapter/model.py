import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.up_proj = nn.Linear(rank, out_features, bias=False)

    def forward(self, x):
        return self.up_proj(self.down_proj(x))


class LoRAAttention(Attention):
    def __init__(self, *args, lora_rank=4, **kwargs):
        super().__init__(*args, **kwargs)  # Initialize the base Attention class
        
        # Define LoRA components for query, key, and value
        self.lora_q = LoRA(self.query_dim, self.query_dim, lora_rank)
        self.lora_k = LoRA(self.query_dim, self.query_dim, lora_rank)
        self.lora_v = LoRA(self.query_dim, self.query_dim, lora_rank)

    def forward(self, hidden_states, *args, **kwargs):
        # Add LoRA-enhanced projections
        query = self.query(hidden_states) + self.lora_q(hidden_states)
        key = self.key(hidden_states) + self.lora_k(hidden_states)
        value = self.value(hidden_states) + self.lora_v(hidden_states)

        # Call the superclass forward method with modified query, key, and value
        return super().forward(query, key, value, *args, **kwargs)
