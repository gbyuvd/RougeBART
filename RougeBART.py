# RhoBART_Clean_Fixed_Phase3.py - Fixed seq2seq model without MTP
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

# Import from Rougeformer
try:
    from Rougeformer import TransformerEncoderLayer, RMSNorm, GQA_RoPE_Attention
except ImportError:
    raise RuntimeError("Rougeformer.py must be in PYTHONPATH")

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


# Modeling

class TransformerDecoderLayer(nn.Module):
    """Decoder layer with self-attention, cross-attention, and FFN."""
    def __init__(self, hidden_size, intermediate_size, num_heads=8, kv_groups=2, 
                 rotary_max_seq=2048, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Self-attention (causal) - reuse GQA from Rougeformer
        self.self_norm = RMSNorm(hidden_size)
        self.self_attn = GQA_RoPE_Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_groups=kv_groups,
            rotary_max_seq=rotary_max_seq,
            window=None,  # No sliding window for decoder
            dropout=dropout
        )

        # Cross-attention (standard multi-head)
        self.cross_norm = RMSNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # FFN - Fixed dropout application
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
        )

    @staticmethod
    def _causal_mask(seq_len, device):
        """Create causal mask for self-attention - True for valid positions"""
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: [B, L, D] decoder hidden states
            memory: [B, S, D] encoder outputs
            memory_key_padding_mask: [B, S] mask for encoder (True for masked positions)
            past_key_value: (self_k, self_v, cross_k, cross_v)
            use_cache: whether to return cache
            attn_mask: optional causal mask
        """
        B, L, D = x.shape
        
        # Shape assertions for debugging
        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"
        assert memory.dim() == 3, f"Expected 3D memory, got {memory.shape}"
        if memory_key_padding_mask is not None:
            assert memory_key_padding_mask.shape == (B, memory.size(1)), f"Memory mask shape mismatch: {memory_key_padding_mask.shape} vs {(B, memory.size(1))}"

        # 1. Self-attention
        residual = x
        x_norm = self.self_norm(x)

        # Create causal mask if not provided
        if attn_mask is None:
            causal_mask = self._causal_mask(L, x.device)  # bool
            attn_mask_for_self = to_additive_mask(causal_mask, dtype=x.dtype)

        else:
            attn_mask_for_self = attn_mask
            assert attn_mask_for_self.shape == (L, L), f"Self-attention mask shape mismatch: {attn_mask_for_self.shape} vs {(L, L)}"

        present_self_k = present_self_v = None
        out = self.self_attn(
            x_norm,
            attn_mask=attn_mask_for_self,
            past_key_value=past_key_value[:2] if past_key_value is not None else None,
            use_cache=use_cache,
        )

        # out could be: attn_out or (attn_out, present_kv) or (attn_out, attn_weights) etc.
        if isinstance(out, tuple):
            # Various signatures possible: (attn_out, present_kv) or (attn_out, attn_weights, present_kv)
            attn_out = out[0]
            if len(out) == 2 and out[1] is not None:
                # (attn_out, present_kv)
                maybe = out[1]
                if isinstance(maybe, tuple) and len(maybe) == 2 and maybe[0].dim() == 4:
                    present_self_k, present_self_v = maybe
            elif len(out) == 3:
                # (attn_out, attn_weights, present_kv)
                if out[2] is not None:
                    present_kv = out[2]
                    if isinstance(present_kv, tuple) and len(present_kv) == 2:
                        present_self_k, present_self_v = present_kv
        else:
            attn_out = out

        x = residual + attn_out

        # 2. Cross-attention
        residual = x
        x_norm2 = self.cross_norm(x)

        # Validate cross-attention inputs
        assert x_norm2.shape[0] == memory.shape[0], "Batch size mismatch between query and key/value"
        assert x_norm2.shape[2] == memory.shape[2], "Feature dimension mismatch between query and key/value"

        # Cross-attention - use encoder memory directly (no caching needed since it's static)
        attn_output, _ = self.cross_attn(
            query=x_norm2,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
            is_causal=False,
        )

        x = residual + attn_output

        # 3. Feed-forward - Fixed dropout: applied only once after final linear
        residual = x
        x_norm3 = self.ffn_norm(x)
        x = residual + self.ffn(x_norm3)  # FFN already has dropout in intermediate layer

        present_key_value = (
            present_self_k,
            present_self_v,
            None,  # No need to cache cross-attention K,V since encoder outputs are static
            None,
        ) if use_cache else None

        return x, present_key_value


class RougeBART(nn.Module):
    """Clean BART-like encoder-decoder model without MTP."""
    def __init__(
        self,
        vocab_size,
        max_seq=512,
        num_layers=6,
        hidden_size=320,
        intermediate_size=1280,
        num_heads=8,
        kv_groups=2,
        rotary_max_seq=2048,
        window=64,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq = max_seq
        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.decoder_start_token_id = kwargs.get("decoder_start_token_id", 2)



        # Shared embeddings
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.drop = nn.Dropout(dropout)

        # Encoder (bidirectional, from Rougeformer)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                kv_groups=kv_groups,
                rotary_max_seq=rotary_max_seq,
                window=window,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.encoder_norm = RMSNorm(hidden_size)

        # Decoder (causal, with cross-attention)
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                kv_groups=kv_groups,
                rotary_max_seq=rotary_max_seq,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.decoder_norm = RMSNorm(hidden_size)

        # Output head (tied with embeddings)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  # better scaling
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.scale)

    def _shift_right(self, labels):
        """Shift labels right for decoder input."""
        if labels is None:
            return None
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 0] = self.decoder_start_token_id
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted = torch.where(labels == -100, self.pad_token_id, shifted)
        return shifted

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_positions: Optional[torch.Tensor] = None,
    ):
        """Run encoder only."""
        B, L = input_ids.shape

        # --- Embed tokens ---
        x = self.drop(self.embed(input_ids))

        # --- Normalize attention mask once and adapt per layer ---
        attn_mask_input = attention_mask
        for layer in self.encoder_layers:
            L_layer = x.size(1)
            attn_mask_layer = None

            if attn_mask_input is not None:
                am = attn_mask_input
                if am.dim() == 1:  # [S] -> [B,S]
                    am = am.unsqueeze(0).expand(B, -1)
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

            x = layer(x, attn_mask=attn_mask_layer, global_positions=global_positions)

        return self.encoder_norm(x)

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ):
        """Run decoder only."""
        # Shape assertions
        B, L = decoder_input_ids.shape
        assert encoder_hidden_states.shape[0] == B, f"Batch size mismatch: {encoder_hidden_states.shape[0]} vs {B}"
        
        h = self.drop(self.embed(decoder_input_ids))
        
        # Convert encoder attention mask to expected format for cross-attention
        memory_key_padding_mask = None
        if encoder_attention_mask is not None:
            # Convert from attention format (1 for valid, 0 for padding) to mask format (True for masked)
            memory_key_padding_mask = encoder_attention_mask == 0

        next_decoder_cache = [] if use_cache else None
        
        for i, layer in enumerate(self.decoder_layers):
            past_for_layer = None
            if past_key_values is not None and i < len(past_key_values):
                past_for_layer = past_key_values[i]
            
            h, present_kv = layer(
                h,
                encoder_hidden_states,
                memory_key_padding_mask=memory_key_padding_mask,
                past_key_value=past_for_layer,
                use_cache=use_cache,
            )
            
            if use_cache:
                next_decoder_cache.append(present_kv)

        h = self.decoder_norm(h)
        logits = self.lm_head(h)

        return {
            "logits": logits,
            "past_key_values": next_decoder_cache if use_cache else None,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        global_positions=None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        """Full forward pass with optional loss computation."""
        # Handle dict input
        if isinstance(input_ids, dict):
            batch = input_ids
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask", attention_mask)
            decoder_input_ids = batch.get("decoder_input_ids", decoder_input_ids)
            decoder_attention_mask = batch.get("decoder_attention_mask", decoder_attention_mask)
            labels = batch.get("labels", labels)
            global_positions = batch.get("global_positions", global_positions)
            past_key_values = batch.get("past_key_values", past_key_values)
            use_cache = batch.get("use_cache", use_cache)

        if input_ids is None:
            raise ValueError("input_ids required for encoder-decoder model")

        # Shape assertions
        B, L = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (B, L), f"Input attention mask shape mismatch: {attention_mask.shape} vs {(B, L)}"
        
        # 1. Encode
        encoder_hidden_states = self.encode(input_ids, attention_mask, global_positions)

        # 2. Prepare decoder input
        if decoder_input_ids is None:
            if labels is None:
                raise ValueError("Either decoder_input_ids or labels must be provided")
            decoder_input_ids = self._shift_right(labels)

        # Shape assertions for decoder
        B_dec, L_dec = decoder_input_ids.shape
        if decoder_attention_mask is not None:
            assert decoder_attention_mask.shape == (B_dec, L_dec), f"Decoder attention mask shape mismatch: {decoder_attention_mask.shape} vs {(B_dec, L_dec)}"
        
        # 3. Decode
        decoder_outputs = self.decode(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = decoder_outputs["logits"]

        # 4. Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shape assertions for loss computation
            assert logits.shape[:2] == labels.shape, f"Logits and labels shape mismatch: {logits.shape[:2]} vs {labels.shape}"
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            if decoder_attention_mask is not None:
                shift_mask = decoder_attention_mask[:, 1:].contiguous()
                valid_mask = (shift_labels != -100) & (shift_mask.bool())
            else:
                valid_mask = shift_labels != -100

            if valid_mask.sum() > 0:
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))[valid_mask.view(-1)]
                flat_labels = shift_labels.view(-1)[valid_mask.view(-1)]
                loss = F.cross_entropy(flat_logits, flat_labels, reduction='mean')
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": decoder_outputs["past_key_values"],
        }

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Incremental autoregressive generation with KV caching.
        HuggingFace-style signature (supports pad/eos/bos ids).
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Encode source sequence
        encoder_hidden_states = self.encode(input_ids, attention_mask)

        # Pick BOS / decoder start token
        start_id = (
            bos_token_id
            if bos_token_id is not None
            else getattr(self, "decoder_start_token_id", None)
        )
        if start_id is None:
            raise ValueError("decoder_start_token_id or bos_token_id must be set")

        current_input = torch.full(
            (batch_size, 1), start_id, dtype=torch.long, device=device
        )

        generated_tokens = [current_input]
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                decoder_outputs = self.decode(
                    decoder_input_ids=current_input,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = decoder_outputs["past_key_values"]
                logits = decoder_outputs["logits"][:, -1, :] / max(temperature, 1e-6)

                # Top-k filtering
                if top_k is not None:
                    values, indices = torch.topk(logits, k=top_k)
                    full = torch.full_like(logits, float("-inf"))
                    full.scatter_(1, indices, values)
                    logits = full

                # Top-p / nucleus filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)

                    sorted_mask = cumprobs > top_p
                    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                    sorted_mask[..., 0] = False

                    mask_original = torch.zeros_like(sorted_mask, dtype=torch.bool).scatter(
                        1, sorted_indices, sorted_mask
                    )
                    logits = logits.masked_fill(mask_original, float("-inf"))

                probs = F.softmax(logits, dim=-1)

                next_token = (
                    torch.multinomial(probs, num_samples=1)
                    if do_sample
                    else torch.argmax(probs, dim=-1, keepdim=True)
                )

                generated_tokens.append(next_token)
                current_input = next_token

                # Early stop if EOS across batch
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break

        # Concatenate along sequence dimension
        return torch.cat(generated_tokens, dim=1)

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
        """Check for gradient explosion and return if clipping is needed."""
        grad_norm = self.gradient_norm()
        needs_clipping = grad_norm > max_norm
        return grad_norm, needs_clipping
