"""
Enhanced Qwen3-Next with DSA support
Supports 3 attention types: full_attention, linear_attention, dsa_attention
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from models.qwen3_next.modular_qwen3_next import (
    Qwen3NextModel,
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextRMSNorm,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
    Qwen3NextRotaryEmbedding,
    Qwen3NextPreTrainedModel,
)
from models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import PreTrainedModel

# Import DeepSeek Sparse Attention
from experiments.exp1_attention_mechanisms.exp1_sparse_vs_classic_attention.sparse_attention import (
    DeepSeekSparseAttention,
)


class EnhancedQwen3NextDecoderLayer(nn.Module):
    """
    Enhanced decoder layer supporting 3 attention types:
    - full_attention: Standard multi-head attention
    - linear_attention: Gated DeltaNet
    - dsa_attention: DeepSeek Sparse Attention
    """
    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        
        # Token mixer - choose based on layer type
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx)
        elif self.layer_type == "dsa_attention":
            # DeepSeek Sparse Attention
            self.dsa_attn = DeepSeekSparseAttention(
                d_model=config.hidden_size,
                n_heads=config.num_attention_heads,
                max_seq_len=config.max_position_embeddings,
                indexer_heads=getattr(config, 'indexer_heads', 4),
                indexer_dim=getattr(config, 'indexer_dim', 64),
                sparse_top_k=getattr(config, 'sparse_top_k', 64),  # Reduced for small models
                dropout=config.attention_dropout,
            )
        else:
            raise ValueError(f"Unknown layer_type: {self.layer_type}")
        
        # MLP (same as Qwen3-Next)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(config)
        else:
            self.mlp = Qwen3NextMLP(config, intermediate_size=config.intermediate_size)
        
        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        
        # Token Mixer - based on layer type
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        elif self.layer_type == "dsa_attention":
            # DSA doesn't use position embeddings (has its own indexing)
            hidden_states, _ = self.dsa_attn(hidden_states)
        
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states
        
        return hidden_states


class EnhancedQwen3NextModel(Qwen3NextPreTrainedModel):
    """
    Enhanced Qwen3-Next model with DSA support
    """
    def __init__(self, config: Qwen3NextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [EnhancedQwen3NextDecoderLayer(config, layer_idx) 
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3NextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if cache_position is None:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        
        return type('ModelOutput', (), {
            'last_hidden_state': hidden_states,
            'past_key_values': past_key_values,
        })()


class EnhancedQwen3NextForCausalLM(PreTrainedModel):
    """
    Enhanced Qwen3-Next for Causal LM with DSA support
    """
    def __init__(self, config):
        super().__init__(config)
        self.model = EnhancedQwen3NextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return type('CausalLMOutput', (), {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.past_key_values,
        })()

