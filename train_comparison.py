import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import pandas as pd
import selfies as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Union

# Add plotting dependencies
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

from transformers import BartConfig, BartForConditionalGeneration
from RougeBART import RougeBART
# -----------------------
# Tokenizer
# -----------------------
try:
    from FastChemTokenizerHF import FastChemTokenizerSelfies
    tokenizer = FastChemTokenizerSelfies.from_pretrained("../selftok_core")
    print(f"✅ Loaded FastChemTokenizer (vocab_size={len(tokenizer)})")
except Exception as e:
    raise RuntimeError(f"Tokenizer not available: {e}")

# -----------------------
# SELFIES processing (same as before)
# -----------------------
def process_selfies_sentence(selfies_str):
    try:
        selfies_tokens = list(sf.split_selfies(selfies_str))
        joined_tokens = []
        i = 0
        while i < len(selfies_tokens):
            if selfies_tokens[i] == '.' and i + 1 < len(selfies_tokens):
                joined_tokens.append(f".{selfies_tokens[i+1]}")
                i += 2
            else:
                joined_tokens.append(selfies_tokens[i])
                i += 1
        return ' '.join(joined_tokens)
    except Exception as e:
        print(f"SELFIES processing Error: {e}")
        return None

# -----------------------
# Dataset with 10% sampling
# -----------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=128, frac=1.0, seed=42):
        df = pd.read_csv(filepath)
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
            print(f"✅ Loaded {len(df)} samples ({frac*100:.1f}% of original from {filepath})")
        self.source_texts = df['text1'].tolist()
        self.target_texts = df['text2'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        processed_source = process_selfies_sentence(source_text)
        processed_target = process_selfies_sentence(target_text)
        if processed_source is None or processed_target is None:
            return "", ""
        return processed_source, processed_target

# -----------------------
# Collator (same for both models)
# -----------------------
def collate_fn(batch, tokenizer=tokenizer, max_len=128):
    source_texts, target_texts = zip(*batch)
    source_texts = [s if s else "" for s in source_texts]
    target_texts = [t if t else "" for t in target_texts]
    source_enc = tokenizer(source_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    target_enc = tokenizer(target_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return {
        "input_ids": source_enc["input_ids"],
        "attention_mask": source_enc["attention_mask"],
        "labels": target_enc["input_ids"],
    }

# -----------------------
# Model Summary Helper
# -----------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model, name="Model"):
    print(f"\n{'='*50}")
    print(f"{name} Architecture Summary")
    print('='*50)
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Check for encoder/decoder structure
    if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
        # Try different possible layer attribute names
        enc_layers = 'Unknown'
        dec_layers = 'Unknown'
        
        # For BART, layers are typically in 'layers' attribute
        if hasattr(model.encoder, 'block'):
            enc_layers = len(model.encoder.block)
        elif hasattr(model.encoder, 'layers'):
            enc_layers = len(model.encoder.layers)
        elif hasattr(model.encoder, '_modules'):
            enc_layers = len([name for name in model.encoder._modules if 'layer' in name.lower() or name.isdigit()])
        
        if hasattr(model.decoder, 'block'):
            dec_layers = len(model.decoder.block)
        elif hasattr(model.decoder, 'layers'):
            dec_layers = len(model.decoder.layers)
        elif hasattr(model.decoder, '_modules'):
            dec_layers = len([name for name in model.decoder._modules if 'layer' in name.lower() or name.isdigit()])
        
        print(f"Encoder layers: {enc_layers}")
        print(f"Decoder layers: {dec_layers}")
    
    print('='*50 + '\n')


# ------------------------------------------------------------------
# Generic GFLOP calculator
# ------------------------------------------------------------------
def gflops_per_layer(model: Union[BartForConditionalGeneration, RougeBART],
                     batch_size: int = 1,
                     seq_len: int = 128) -> dict:
    """
    Returns a dict with:
        encoder_gflops : float
        decoder_gflops : float   (0.0 for encoder-only models)
    Computed for a single forward pass of *one* layer.
    """
    cfg = model.config if hasattr(model, 'config') else model   # RougeBART stores attrs directly

    # ----- common quantities -----
    def _get(attr, default):
        return getattr(cfg, attr, default)

    hidden       = _get('hidden_size', _get('d_model', 320))
    ff           = _get('intermediate_size', _get('d_ff', 1280))
    heads        = _get('num_heads', _get('num_attention_heads', 8))
    kv_groups    = _get('kv_groups', 1)          # 1 ⇒ full MHA
    head_dim     = hidden // heads
    kv_heads     = heads // kv_groups

    # ---------- self-attention ----------
    # Q-proj, K-proj, V-proj, Out-proj
    # We count matmuls only (bias adds are negligible)
    attn_proj_q  = batch_size * seq_len * (hidden * hidden)               # Q dense
    attn_proj_kv = batch_size * seq_len * (kv_heads * head_dim * hidden)  # K/V dense (GQA)
    attn_out     = batch_size * seq_len * (hidden * hidden)               # out-proj
    # QK^T and AV matmuls inside attention
    qk_dot       = batch_size * heads * seq_len * seq_len * head_dim
    av_dot       = batch_size * heads * seq_len * head_dim * seq_len
    attn_total   = attn_proj_q + 2 * attn_proj_kv + attn_out + qk_dot + av_dot

    # --------------- FFN -----------------
    ffn_total    = batch_size * seq_len * (hidden * ff + ff * hidden)

    # --------------- encoder -------------
    enc_total    = (attn_total + ffn_total) / 1e9   # -> GFLOPs

    # --------------- decoder -------------
    # decoder has *self* attn + *cross* attn + FFN
    cross_attn   = attn_total                         # same sizes, just different tensors
    dec_total    = (2 * attn_total + ffn_total) / 1e9

    # --------------- return --------------
    if isinstance(model, RougeBART):
        return dict(encoder_gflops=enc_total, decoder_gflops=dec_total)
    elif isinstance(model, BartForConditionalGeneration):
        # HF BART shares hyper-params across enc/dec
        return dict(encoder_gflops=enc_total, decoder_gflops=dec_total)
    else:
        raise TypeError("Unknown model class")


# ------------------------------------------------------------------
# Pretty print helper
# ------------------------------------------------------------------
def check_flop_parity(model_a, name_a, model_b, name_b, bsz=1, seq=128):
    a = gflops_per_layer(model_a, bsz, seq)
    b = gflops_per_layer(model_b, bsz, seq)
    print("GFLOPs per layer (B=1, L=128)")
    print("-" * 40)
    print(f"{name_a:<12}  enc: {a['encoder_gflops']:.3f}  dec: {a['decoder_gflops']:.3f}")
    print(f"{name_b:<12}  enc: {b['encoder_gflops']:.3f}  dec: {b['decoder_gflops']:.3f}")
    enc_diff = abs(a['encoder_gflops'] - b['encoder_gflops']) / a['encoder_gflops'] * 100
    dec_diff = abs(a['decoder_gflops'] - b['decoder_gflops']) / a['decoder_gflops'] * 100
    print("-" * 40)
    print(f"encoder Δ = {enc_diff:.2f}%   decoder Δ = {dec_diff:.2f}%")
    
# -----------------------
# Evaluation & Perplexity Functions
# -----------------------
def evaluate_model_with_tracking(model, eval_loader, device, model_name, track_losses=True):
    model.eval()
    eval_losses = []
    eval_pbar = tqdm(eval_loader, total=len(eval_loader), desc=f"[{model_name}] Evaluation")
    
    with torch.no_grad():
        for batch in eval_pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"].item()
            eval_losses.append(loss)
            eval_pbar.set_postfix({"Loss": f"{loss:.4f}"})
    
    avg_loss = np.mean(eval_losses)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity, eval_losses if track_losses else None

def train_model_with_tracking(model, model_name, train_loader, eval_loader, device, num_epochs=1):
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    checkpoint_interval = max(1, total_steps // 4)
    global_step = 0

    train_losses = []
    eval_losses = []
    eval_perplexities = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs}")
        
        for step, batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_train_losses.append(loss_val)
            train_losses.append(loss_val)
            train_pbar.set_postfix({"Loss": f"{loss_val:.4f}"})

            if (global_step + 1) % checkpoint_interval == 0:
                eval_loss, eval_ppl, _ = evaluate_model_with_tracking(model, eval_loader, device, f"{model_name} (Step {global_step+1})", track_losses=False)
                eval_losses.append(eval_loss)
                eval_perplexities.append(eval_ppl)
                print(f"\n[{model_name}] Checkpoint @ Step {global_step+1} | Eval Loss: {eval_loss:.4f} | PPL: {eval_ppl:.2f}")

            global_step += 1

        # End of epoch evaluation
        avg_train_loss = np.mean(epoch_train_losses)
        final_eval_loss, final_eval_ppl, _ = evaluate_model_with_tracking(model, eval_loader, device, f"{model_name} (Epoch {epoch+1})", track_losses=False)
        eval_losses.append(final_eval_loss)
        eval_perplexities.append(final_eval_ppl)
        
        print(f"[{model_name}] Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {final_eval_loss:.4f} | PPL: {final_eval_ppl:.2f}")
    
    return train_losses, eval_losses, eval_perplexities

# -----------------------
# Plotting Functions
# -----------------------
def plot_comparison(rouge_train, rouge_eval, bart_train, bart_eval, metric_name, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(rouge_train, label='RougeBART Train', alpha=0.7)
    plt.plot(rouge_eval, label='RougeBART Eval', linestyle='--', alpha=0.7)
    plt.plot(bart_train, label='BART Train', alpha=0.7)
    plt.plot(bart_eval, label='BART Eval', linestyle='--', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Plot saved: {filename}")

def plot_perplexity_comparison(rouge_ppl, bart_ppl, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(rouge_ppl, label='RougeBART', alpha=0.7)
    plt.plot(bart_ppl, label='BART', alpha=0.7)
    plt.xlabel('Evaluation Step')
    plt.ylabel('Perplexity')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"📊 Perplexity plot saved: {filename}")

# -----------------------
# Main Comparison
# -----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Load 10% data ===
    DATA_FRAC = 0.1
    print("Loading datasets...")
    train_dataset = Seq2SeqDataset("../data/NP_train_seq2seq.csv", tokenizer, frac=DATA_FRAC)
    eval_dataset = Seq2SeqDataset("../data/NP_test_seq2seq.csv", tokenizer, frac=DATA_FRAC)  # or frac=1.0 for full test

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # === Build RougeBART ===
    print("\nBuilding RougeBART...")
    rouge_bart = RougeBART(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.bos_token_id,
        max_seq=128,
        num_layers=6,
        hidden_size=320,
        intermediate_size=1280,
        num_heads=8,
        kv_groups=2,
        rotary_max_seq=2048,
        dropout=0.1,
    ).to(device)

    print_model_summary(rouge_bart, "RougeBART")

    # === Build BART with matched config ===
    print("\nBuilding Custom BART...")
    # ------------------------------------------------------------------
    # Build BART (matched compute)  –  replaces T5 block
    # ------------------------------------------------------------------

    print("\nBuilding matched BART-large clone...")
    bart_config = BartConfig(
        vocab_size=len(tokenizer),
        d_model=320,                     # hidden
        encoder_ffn_dim=1280,            # intermediate
        decoder_ffn_dim=1280,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        max_position_embeddings=128,     # we truncate at 128 anyway
        dropout=0.1,
        attention_dropout=0.1,
        activation_function="gelu",      # same as RougeBART
        use_cache=True,                  # for fair generation later
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        forced_bos_token_id=tokenizer.bos_token_id,  # BART expects this for generation
    )

    bart_model = BartForConditionalGeneration(bart_config).to(device)
    print_model_summary(bart_model, "BART (matched)")

    # === Compare params ===
    rb_params = count_parameters(rouge_bart)
    bart_params = count_parameters(bart_model)
    print(f"\n📊 Parameter Comparison:")
    print(f"RougeBART: {rb_params:,}")
    print(f"BART       : {bart_params:,}")
    print(f"Difference: {abs(rb_params - bart_params):,} ({abs(rb_params - bart_params)/rb_params*100:.2f}%)")
    check_flop_parity(rouge_bart, "RougeBART", bart_model, "BART")
    if abs(rb_params - bart_params) / rb_params > 0.05:
        print("⚠️ Warning: Parameter counts differ by >5%. Consider adjusting BART config.")

    # === Train both with tracking ===
    print("\n🚀 Starting RougeBART Training...")
    rb_train_losses, rb_eval_losses, rb_eval_ppls = train_model_with_tracking(
        rouge_bart, "RougeBART", train_loader, eval_loader, device
    )

    print("\n🚀 Starting BART Training...")
    bart_train_losses, bart_eval_losses, bart_eval_ppls = train_model_with_tracking(
        bart_model, "BART", train_loader, eval_loader, device
    )

    # === Final Evaluation on Test Set ===
    print("\n🔍 Final Evaluation on Test Set...")
    rb_final_loss, rb_final_ppl, _ = evaluate_model_with_tracking(rouge_bart, eval_loader, device, "RougeBART Final", track_losses=False)
    bart_final_loss, bart_final_ppl, _ = evaluate_model_with_tracking(bart_model, eval_loader, device, "BART Final", track_losses=False)

    # === Plotting ===
    print("\n📊 Generating Plots...")
    plot_comparison(
        rb_train_losses, rb_eval_losses, 
        bart_train_losses, bart_eval_losses,
        "Loss", "Train vs Eval Loss Comparison", "loss_comparison.png"
    )
    
    plot_perplexity_comparison(
        rb_eval_ppls, bart_eval_ppls,
        "Perplexity Comparison Over Training", "ppl_comparison.png"
    )

    # === Final Comparison ===
    print("\n" + "="*70)
    print("FINAL RESULTS (10% data, 1 epoch)")
    print("="*70)
    print(f"RougeBART:")
    print(f"  - Final Eval Loss: {rb_final_loss:.4f}")
    print(f"  - Final Perplexity: {rb_final_ppl:.2f}")
    print(f"  - Total Train Steps: {len(rb_train_losses)}")
    print(f"  - Parameters: {rb_params:,}")
    
    print(f"\nBART:")
    print(f"  - Final Eval Loss: {bart_final_loss:.4f}")
    print(f"  - Final Perplexity: {bart_final_ppl:.2f}")
    print(f"  - Total Train Steps: {len(bart_train_losses)}")
    print(f"  - Parameters: {bart_params:,}")
    
    print(f"\n📊 Difference:")
    print(f"  - Loss Difference: {abs(rb_final_loss - bart_final_loss):.4f}")
    print(f"  - PPL Difference: {abs(rb_final_ppl - bart_final_ppl):.2f}")
    print("="*70)

if __name__ == "__main__":
    main()