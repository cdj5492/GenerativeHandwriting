#!/usr/bin/env python
# coding: utf-8
"""
Training script for Transformer handwriting models.
Implements mixed precision, Transformer LR schedule with warm-up, and
proper padding masks.
"""

import os, math, argparse, random, numpy as np
import torch, torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ------------------------------------------------------------------ #
#   MODELS - import the Transformer classes and alias the old names
# ------------------------------------------------------------------ #
from transformer_handwriting_models import HandWritingPredictionNet, HandWritingSynthesisNet

# (everything below still refers to HandWritingPredictionNet / SynthesisNet)

# ------------------------------------------------------------------ #
#   UTILS
# ------------------------------------------------------------------ #
from utils import plot_stroke
from utils.constants import Global
from utils.dataset import HandwritingDataset, MathHandwritingDataset
from utils.model_utils import compute_nll_loss
from utils.data_utils import data_denormalization
from generate import generate_conditional_sequence, generate_unconditional_seq

# ------------------------------------------------------------------ #
#   ARGUMENTS
# ------------------------------------------------------------------ #
def argparser():
    p = argparse.ArgumentParser("Transformer Handwriting Trainer")
    p.add_argument("--d_model",     type=int,   default=512)
    p.add_argument("--nhead",       type=int,   default=8)
    p.add_argument("--nlayers",     type=int,   default=6)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--n_epochs",    type=int,   default=100)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup",      type=int,   default=4000,
                       help="warm-up steps for Transformer schedule")
    p.add_argument("--patience",    type=int,   default=1500)
    p.add_argument("--model_type",  type=str,   default="prediction",
                       choices=["prediction", "synthesis"])
    p.add_argument("--data_path",   type=str,   default="./data/")
    p.add_argument("--save_path",   type=str,   default="./logs/")
    p.add_argument("--text_req",    action="store_true")
    p.add_argument("--data_aug",    action="store_true")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--seed",        type=int,   default=212)
    return p.parse_args()

# ------------------------------------------------------------------ #
#   LR SCHEDULE  (Transformer warm-up+inverse-sqrt decay)
# ------------------------------------------------------------------ #
def transformer_lr_lambda(warmup):
    def f(step):
        step = max(step, 1)
        return (1 / math.sqrt(step)) * min(step / warmup, 1.0)
    return f

# ------------------------------------------------------------------ #
#   ONE EPOCH -mixed precision, masks, clipping
# ------------------------------------------------------------------ #
def run_epoch(model, loader, scaler, optimiser, device, is_train, model_type):
    if is_train:
        model.train()
    else:
        model.eval()

    total, n_tokens = 0.0, 0
    autocast = torch.amp.autocast

    for i, batch in enumerate(loader):
        if model_type == "prediction":
            # batch = (inputs, targets, mask)
            inputs, targets, mask = batch
            text, text_mask = None, None
        else:
            # batch = (inputs, targets, mask, text, text_mask)
            inputs, targets, mask, text, text_mask = batch
            text, text_mask = text.to(device), text_mask.to(device)

        inputs, targets, mask = (
            inputs.to(device), targets.to(device), mask.to(device)
        )

        # Transformer expects True for PAD tokens
        stroke_pad_mask = (mask == 0)

        with torch.no_grad() if not is_train else torch.enable_grad(), \
             autocast(enabled=True, device_type=device.type):
            if model_type == "prediction":
                y_hat = model(inputs, src_key_padding_mask=stroke_pad_mask)
            else:
                y_hat = model(
                    inputs, text,
                    stroke_padding_mask=stroke_pad_mask,
                    text_padding_mask=(text_mask == 0),
                )
            loss = compute_nll_loss(targets, y_hat, mask)

        if is_train:
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # clip *global* norm (safer for attention)
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            if i % 10 == 0:
                batch_size = inputs.shape[0]
                print(
                    "[{:5d}] loss: {:.3f}".format(i + 1, loss / batch_size)
                )

        total += loss.item() * inputs.size(0)
        n_tokens += inputs.size(0)

    return total / n_tokens


# ------------------------------------------------------------------ #
#   MAIN TRAINING LOOP
# ------------------------------------------------------------------ #
def train(model, train_loader, valid_loader, args, device):
    model = model.to(device)
    optimiser = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=1e-2,
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=transformer_lr_lambda(args.warmup)
    )
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    # --- MODIFIED: Construct save path for model file with 't_' prefix ---
    model_save_path = os.path.join(args.save_path,
                                   f"t_best_{args.model_type}.pt")
    # --- END MODIFICATION ---

    os.makedirs(args.save_path, exist_ok=True)

    # --- ADDED: Create specific subdirectories for plots ---
    pred_plot_dir = os.path.join(args.save_path, 't_prediction')
    synth_plot_dir = os.path.join(args.save_path, 't_synthesis')
    os.makedirs(pred_plot_dir, exist_ok=True)
    os.makedirs(synth_plot_dir, exist_ok=True)
    # --- END ADDITION ---

    best_loss, best_epoch = float("inf"), 0

    # if the modelfile doesn't exist, create it (using the new path)
    if not os.path.exists(model_save_path):
        torch.save(model.state_dict(), model_save_path)
        print(f"Initial model state saved to {model_save_path}")
    else:
        print(f"Loading existing model state from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))

    # generate one before training for visualization
    # --- MODIFIED: Use new plot directories and simplified filenames ---
    print("Generating initial sample plot...")
    if args.model_type == "prediction":
        gen_seq = generate_unconditional_seq(
            model_path=model_save_path, seq_len=700, device=device, bias=10.0, style=None, prime=False
        )
        gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)
        initial_plot_path = os.path.join(pred_plot_dir, "initial_prediction.png")
        plot_stroke(
            gen_seq[0],
            save_name=initial_plot_path,
        )
        print(f"Saved initial prediction plot to {initial_plot_path}")
    else: # Synthesis
        gen_seq, phi = generate_conditional_sequence(
            model_path=model_save_path,
            char_seq="3(3b+4)-6=-12",
            device=device,
            dataset=train_loader.dataset,
            bias=10.0,
            prime=False,
            prime_seq=None,
            real_text=None,
            is_map=True,
        )
        gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)
        initial_plot_path = os.path.join(synth_plot_dir, "initial_synthesis.png")
        plot_stroke(
            gen_seq[0],
            save_name=initial_plot_path,
        )
        print(f"Saved initial synthesis plot to {initial_plot_path}")
    # --- END MODIFICATION ---

    print("Starting training loop...")
    for epoch in range(1, args.n_epochs + 1):
        tr_loss = run_epoch(model, train_loader, scaler,
                            optimiser, device, True, args.model_type)
        val_loss = run_epoch(model, valid_loader, scaler,
                             optimiser, device, False, args.model_type)
        scheduler.step()

        print(f"Epoch {epoch:3d}: train={tr_loss:.4f}  valid={val_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # periodic plotting for debug
        # --- MODIFIED: Use new plot directories ---
        if args.model_type == "prediction":
            state_path = model_save_path if os.path.exists(model_save_path) else None
            gen_seq = generate_unconditional_seq(
                model_path = state_path,
                seq_len=700, device=device, bias=10.0,
                style=None, prime=False
            )
            gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)
            plot_stroke(
                gen_seq[0],
                save_name=os.path.join(pred_plot_dir, 'pred_epoch%d.png' % epoch) # Save to t_prediction subdir
            )
        else: # Synthesis
            state_path = model_save_path if os.path.exists(model_save_path) else None
            gen_seq, phi = generate_conditional_sequence(
                model_path=state_path,
                char_seq='3(3b+4)-6=-12',
                device=device, dataset=train_loader.dataset,
                bias=10.0, prime=False,
                prime_seq=None, real_text=None, is_map=True
            )
            # Optional: Save phi plot (attention map)
            # phi_plot_path = os.path.join(synth_plot_dir, 'phi_epoch%d.png' % epoch)
            # plt.figure()
            # plt.imshow(phi, cmap='viridis', aspect='auto')
            # plt.colorbar()
            # plt.savefig(phi_plot_path)
            # plt.close()

            gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)
            plot_stroke(
                gen_seq[0],
                save_name=os.path.join(synth_plot_dir, 'synth_epoch%d.png' % epoch) # Save to t_synthesis subdir
            )
        # --- END MODIFICATION ---

        if val_loss < best_loss:
            # --- MODIFIED: Save using the new model path ---
            torch.save(model.state_dict(), model_save_path)
            # --- END MODIFICATION ---
            best_loss, best_epoch = val_loss, epoch
            print(f"  > saved new best model (epoch {epoch}) to {model_save_path}")

    print(f"Training done. Best epoch = {best_epoch} (valid={best_loss:.4f})")


# ------------------------------------------------------------------ #
#   ENTRY-POINT
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    args = argparser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # ------------------------ DATA ---------------------------------- #
    print("Loading datasets...")
    # train_ds = MathHandwritingDataset(
    train_ds = HandwritingDataset(
        args.data_path, split="train",
        text_req=args.text_req, debug=args.debug, data_aug=args.data_aug
    )
    # valid_ds = MathHandwritingDataset(
    valid_ds = HandwritingDataset(
        args.data_path, split="valid",
        text_req=args.text_req, debug=args.debug, data_aug=args.data_aug
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  pin_memory=True, drop_last=True, num_workers=4) # Added num_workers
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                              shuffle=False, pin_memory=True, drop_last=True, num_workers=4) # Added num_workers
    print("Datasets loaded.")

    # ------------------------ MODEL --------------------------------- #
    print(f"Building {args.model_type} model...")
    if args.model_type == "prediction":
        model = HandWritingPredictionNet(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.nlayers,
        )
    else: # Synthesis
        model = HandWritingSynthesisNet(
            vocab_size=train_ds.vocab_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.nlayers,
            num_decoder_layers=args.nlayers,
        )
    print("Model built.")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params/1e6:.2f}M trainable parameters.")


    # ------------------ TRAIN --------------------------------------- #
    train(model, train_loader, valid_loader, args, device)