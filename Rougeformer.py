# rougeformer_fixed.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple  # For Python versions < 3.10


# Utils
def to_additive_mask(mask: torch.Tensor, dtype=torch.float32):
    """
    Convert bool or 0/1 mask -> additive mask with finite values.
    Shape preserved.
    """
    if mask.dtype != torch.bool:
        mask = mask != 0
    mask = mask.to(dtype)

    # Valid = 0, invalid = large negative (not -inf)
    additive = torch.where(mask > 0.5,
                           torch.zeros_like(mask, dtype=dtype),
                           torch.full_like(mask, -1e4, dtype=dtype))
    if not torch.isfinite(additive).all():
        print("⚠️ NaN in additive mask")
    return additive




# -------------------------
# RMSNorm
# -------------------------
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Safe RMSNorm with epsilon
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        out = x / norm * self.scale
        if not torch.isfinite(out).all():
            print("⚠️ NaN in RMSNorm")
        return out

# -------------------------
# RoPE utilities
# -------------------------
def sincos_cache(max_seq, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq, device=device).type_as(inv_freq)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    sin = emb.sin()[None, :, :]   # [1, seq, dim]
    cos = emb.cos()[None, :, :]   # [1, seq, dim]
    return sin, cos

def apply_rope(q, k, sin, cos):
    """
    Apply RoPE to q,k expected in [B, H, L, D] layout.
    sin,cos shapes: [1, seq, dim]
    """
    # Expect q: [B, H, L, D]
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError("apply_rope expects q,k with shape [B, H, L, D]")
    B, H, L, D = q.shape
    if L == 0:
        return q, k

    # permute to [B, L, H, D] to align with sin/cos indexing
    q_ = q.permute(0, 2, 1, 3)  # [B, L, H, D]
    k_ = k.permute(0, 2, 1, 3)

    sin_ = sin[:, :L, :].to(q.device).unsqueeze(2)  # [1, L, 1, D]
    cos_ = cos[:, :L, :].to(q.device).unsqueeze(2)  # [1, L, 1, D]

    def rope(x):
        # x: [B, L, H, D]
        if x.numel() == 0:
            return x
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # interleave rotated pairs
        xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
        return x * cos_ + xr * sin_

    q_r = rope(q_)
    k_r = rope(k_)
    # permute back to [B, H, L, D]
    q_r = q_r.permute(0, 2, 1, 3).contiguous()
    k_r = k_r.permute(0, 2, 1, 3).contiguous()
    return q_r, k_r

# -------------------------
# Sliding window mask with globals
# -------------------------
def sliding_window_mask_with_global(seq_len, window, global_positions):
    if seq_len == 0:
        return torch.zeros((0, 0), dtype=torch.bool)
    if seq_len == 1:
        return torch.ones((1, 1), dtype=torch.bool)

    idxs = torch.arange(seq_len)
    diff = (idxs.unsqueeze(0) - idxs.unsqueeze(1)).abs()
    mask = diff <= window

    # Safe iteration over global_positions
    if global_positions is None:
        global_positions = []
    elif isinstance(global_positions, torch.Tensor):
        if global_positions.numel() == 0:
            global_positions = []
        else:
            global_positions = global_positions.flatten().tolist()

    for g in global_positions:
        if isinstance(g, torch.Tensor):
            if g.numel() == 0:
                continue
            g = int(g)
        if 0 <= g < seq_len:
            mask[g, :] = True
            mask[:, g] = True
    return mask

# -------------------------
# Attention: GQA + RoPE + sliding window + global
# -------------------------
class GQA_RoPE_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, kv_groups=1,
                 rotary_max_seq=2048, window=None, dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % kv_groups == 0
        self.hidden = hidden_size
        self.num_heads = num_heads
        self.kv_groups = kv_groups
        self.head_dim = hidden_size // num_heads
        self.heads_per_group = num_heads // kv_groups
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, kv_groups * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, kv_groups * self.head_dim)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.rotary_max_seq = rotary_max_seq
        self._sincos = None
        self._sincos_device = None
        self.window = window

    def _get_sincos(self, device):
        # regenerate if device changed
        if self._sincos is None or self._sincos_device != device:
            self._sincos = sincos_cache(self.rotary_max_seq, self.head_dim, device)
            self._sincos_device = device
        return self._sincos

    def apply_rope(self, q, k):
        """Apply rotary positional embeddings to q and k (delegates to helper)."""
        sin, cos = self._get_sincos(q.device)
        return apply_rope(q, k, sin, cos)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        global_positions: Optional[torch.Tensor] = None,
        return_attn_probs: bool = False,
    ):
        """
        Forward pass for GQA_RoPE_Attention with caching support.
        Args:
            x: [B, L, D] input embeddings
            attn_mask: optional attention mask broadcastable to [B, H, L, S] (additive: -inf for masked)
            past_key_value: (past_k, past_v) cached tensors, each [B, H, L_prev, head_dim]
            use_cache: if True, return present_key_value along with output
            global_positions: optional global attention positions
            return_attn_probs: if True returns (attn_out, attn_weights)
        Returns:
            - attn_out if use_cache=False and not return_attn_probs
            - (attn_out, present_key_value) if use_cache=True and not return_attn_probs
            - (attn_out, attn_weights) if return_attn_probs and not use_cache
            - (attn_out, attn_weights, present_key_value) if both True
        """
        B, L, D = x.size()
        H = self.num_heads
        G = self.kv_groups
        Hd = self.head_dim

        # Project q, k, v
        q = self.q_proj(x)  # [B, L, H*Hd]
        k = self.k_proj(x)  # [B, L, G*Hd]
        v = self.v_proj(x)  # [B, L, G*Hd]

        # Reshape to [B, H, L, Hd] for Q, and [B, G, L, Hd] for K,V
        q = q.view(B, L, H, Hd).transpose(1, 2)  # [B, H, L, Hd]
        k = k.view(B, L, G, Hd).transpose(1, 2)  # [B, G, L, Hd]
        v = v.view(B, L, G, Hd).transpose(1, 2)  # [B, G, L, Hd]

        # Expand K,V to match Q heads (GQA: each group serves multiple query heads)
        k = k.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).contiguous()  # [B, G, H_per_group, L, Hd]
        k = k.view(B, H, L, Hd)  # [B, H, L, Hd]
        v = v.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1).contiguous()
        v = v.view(B, H, L, Hd)  # [B, H, L, Hd]

        # Apply RoPE rotation to q, k
        q, k = self.apply_rope(q, k)

        # Concatenate cached keys/values if provided (both in [B, H, L, Hd] layout)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is not None and past_v is not None:
                # concat along seq_len dim (dim=2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

        # Apply sliding window mask if specified
        if self.window is not None:
            current_seq_len = k.size(2)
            window_mask = sliding_window_mask_with_global(current_seq_len, self.window, global_positions)
            window_mask = window_mask.to(x.device)
            window_attn_mask = torch.zeros_like(window_mask, dtype=torch.float, device=x.device)
            window_attn_mask.masked_fill_(~window_mask, float("-inf"))

            # Expand window mask to [1,1,L, current_seq_len] and combine
            window_attn_mask_expanded = window_attn_mask[-L:, :].unsqueeze(0).unsqueeze(0)
            if attn_mask is None:
                attn_mask = window_attn_mask_expanded
            else:
                attn_mask = attn_mask + window_attn_mask_expanded

        # Scaled dot-product attention
        # q: [B, H, L, Hd], k: [B, H, S, Hd] -> scores [B, H, L, S]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask  # additive mask (-inf where masked)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)  # [B, H, L, Hd]

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)

        # Output projection
        attn_out = self.out_proj(attn_out)

        present_key_value = (k, v) if use_cache else None

        # return variants
        if return_attn_probs and use_cache:
            return attn_out, attn_weights, present_key_value
        if return_attn_probs:
            return attn_out, attn_weights
        if use_cache:
            return attn_out, present_key_value
        return attn_out

# -------------------------
# Encoder Layer
# -------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads=4, kv_groups=1,
                 rotary_max_seq=2048, window=None, dropout=0.1, ff_dropout=0.0):
        super().__init__()
        self.pre_attn_norm = RMSNorm(hidden_size)
        self.attn = GQA_RoPE_Attention(hidden_size, num_heads, kv_groups,
                                       rotary_max_seq, window, dropout)
        self.post_attn_dropout = nn.Dropout(dropout)
        self.pre_ffn_norm = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x, attn_mask=None, global_positions=None, return_attn_probs=False):
        a = self.pre_attn_norm(x)
        # Pass global_positions to attention
        if return_attn_probs:
            out = self.attn(a, attn_mask=attn_mask, global_positions=global_positions, return_attn_probs=True)
            # Unpack depending on whether use_cache was requested by attention (we don't use cache here)
            if isinstance(out, tuple) and len(out) >= 2:
                a, attn_probs = out[0], out[1]
            else:
                a = out
                attn_probs = None
        else:
            a = self.attn(a, attn_mask=attn_mask, global_positions=global_positions)
            attn_probs = None

        x = x + self.post_attn_dropout(a)
        f = self.pre_ffn_norm(x)
        out = x + self.ffn(f)
        if return_attn_probs:
            return out, attn_probs
        else:
            return out

# -------------------------
# Full Model
# -------------------------
class Rougeformer(nn.Module):
    def __init__(self, vocab_size, max_seq=512, num_layers=8, hidden_size=320,
                 intermediate_size=1280, num_heads=8, kv_groups=2,
                 rotary_max_seq=2048, window=64, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden_size
        self.max_seq = max_seq

        # --- ONLY token embeddings ---
        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # --- Transformer layers ---
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, intermediate_size,
                                    num_heads, kv_groups,
                                    rotary_max_seq, window, dropout)
            for _ in range(num_layers)
        ])

        # --- Output head ---
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight  # weight tying

        # Apply weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Apply proper initialization (clean version)."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # better scaling
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)

    def log_training_step(self, loss, step=None):
        """Optional debug logger for training."""
        grad_norm, needs_clip = self.check_gradients(max_norm=1.0)
        msg = f"[Rougeformer] loss={loss.item():.4f}, grad_norm={grad_norm:.4f}, needs_clip={needs_clip}"
        if step is not None:
            msg = f"Step {step}: " + msg
        print(msg)

    def gradient_norm(self):
        """Calculate the norm of all gradients safely (ignores NaNs)."""
        norms = []
        for p in self.parameters():
            if p.grad is not None:
                val = p.grad.data.norm(2).item()
                if not math.isnan(val):
                    norms.append(val ** 2)
        if len(norms) == 0:
            return 0.0
        val = (sum(norms)) ** 0.5
        return val if val > 0 else 1e-8

    def check_gradients(self, max_norm=1.0):
        """Check gradients against a clipping threshold."""
        grad_norm = self.gradient_norm()
        needs_clip = grad_norm > max_norm
        return grad_norm, needs_clip

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_positions=None,
        labels=None,
        output_attentions=False,
        **kwargs,
    ):
        # --- Handle dict or BatchEncoding-like inputs FIRST ---
        if isinstance(input_ids, dict):
            if "input_ids" not in input_ids:
                raise ValueError("Dict passed as input_ids but no 'input_ids' key found.")
            attention_mask = input_ids.get("attention_mask", attention_mask)
            labels = input_ids.get("labels", labels)
            global_positions = input_ids.get("global_positions", global_positions)
            input_ids = input_ids["input_ids"]

        if hasattr(input_ids, "input_ids"):  # HuggingFace BatchEncoding
            attention_mask = getattr(input_ids, "attention_mask", attention_mask)
            labels = getattr(input_ids, "labels", labels)
            global_positions = getattr(input_ids, "global_positions", global_positions)
            input_ids = input_ids.input_ids

        if input_ids is None:
            raise ValueError("`input_ids` cannot be None")

        if not isinstance(input_ids, torch.Tensor):
            raise TypeError(f"`input_ids` must be a torch.Tensor, got {type(input_ids)}")

        # --- Squeeze extra dims if needed ---
        if input_ids.dim() == 3 and input_ids.size(1) == 1:
            input_ids = input_ids.squeeze(1)
        if attention_mask is not None and attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)

        # --- Validate final shape ---
        if input_ids.dim() != 2:
            raise ValueError(f"Expected input_ids to be 2D (batch, seq_len), got {input_ids.shape}")

        b, s = input_ids.shape
        if s > self.max_seq:
            raise ValueError(f"Sequence length {s} exceeds max_seq {self.max_seq}")

        # --- Validate labels ---
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"labels must be a torch.Tensor, got {type(labels)}")
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            if labels.shape != input_ids.shape:
                raise ValueError(f"labels shape {labels.shape} must match input_ids {input_ids.shape}")

        # --- Embed tokens ---
        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)

        all_attentions = [] if output_attentions else None

        # --- Normalize attention mask once and adapt per layer ---
        attn_mask_input = attention_mask

        for layer in self.layers:
            L_layer = x.size(1)
            attn_mask_layer = None

            if attn_mask_input is not None:
                am = attn_mask_input
                if am.dim() == 1:  # [S] -> [B,S]
                    am = am.unsqueeze(0).expand(b, -1)
                elif am.dim() == 3 and am.size(1) == 1:
                    am = am.squeeze(1)

                # Trim or pad so mask length matches current sequence length
                if am.size(1) != L_layer:
                    if am.size(1) > L_layer:
                        am = am[:, :L_layer]
                    else:
                        pad_len = L_layer - am.size(1)
                        pad = torch.ones((am.size(0), pad_len), dtype=am.dtype, device=am.device)
                        am = torch.cat([am, pad], dim=1)

                # Convert to additive mask [B,S] -> [B,1,1,S]
                attn_mask_layer = to_additive_mask(am, dtype=x.dtype).unsqueeze(1).unsqueeze(2)
            else:
                attn_mask_layer = None

            if output_attentions:
                x, attn_probs = layer(x, attn_mask=attn_mask_layer, global_positions=global_positions, return_attn_probs=True)
                all_attentions.append(attn_probs)
            else:
                x = layer(x, attn_mask=attn_mask_layer, global_positions=global_positions)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        # Optional: scale logits for stability
        logits = logits / math.sqrt(self.hidden_size)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        output = (loss, logits) if loss is not None else logits

        if output_attentions:
            return output, all_attentions
        else:
            return output
