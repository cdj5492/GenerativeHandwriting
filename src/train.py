import torch
import math
import os
import torch.nn as nn
import numpy as np
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data
from torch.utils.data import DataLoader, Subset
from torch.distributions import bernoulli, uniform
import torch.nn.functional as F
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import HandWritingPredictionNet, HandWritingSynthesisNet
# from transformer_handwriting_models import HandWritingPredictionNet, HandWritingSynthesisNet
from utils import plot_stroke, visualize_mdn_overlay
from utils.constants import Global
from utils.dataset import HandwritingDataset, MathHandwritingDataset
from utils.model_utils import compute_nll_loss
from utils.data_utils import data_denormalization
from generate import generate_conditional_sequence, generate_unconditional_seq


def argparser():

    parser = argparse.ArgumentParser(description="PyTorch Handwriting Synthesis Model")
    parser.add_argument("--hidden_size", type=int, default=400)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--model_type", type=str, default="prediction")
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--save_path", type=str, default="./logs/")
    parser.add_argument("--text_req", action="store_true")
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=212, help="random seed")
    parser.add_argument("--char_seq", type=str, default="3(3b+4)-6=-12")
    args = parser.parse_args()

    return args


def train_epoch(model, optimizer, epoch, train_loader, device, model_type):
    avg_loss = 0.0
    model.train()
    for i, mini_batch in enumerate(train_loader):
        if model_type == "prediction":
            inputs, targets, mask = mini_batch
        else:
            inputs, targets, mask, text, text_mask = mini_batch
            text = text.to(device)
            text_mask = text_mask.to(device)

        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        batch_size = inputs.shape[0]

        optimizer.zero_grad()

        if model_type == "prediction":
            initial_hidden = model.init_hidden(batch_size, device)
            y_hat, state = model.forward(inputs, initial_hidden)
        else:
            initial_hidden, initial_context = model.init_hidden(batch_size, device) # Get initial context
            # Pass the initial context for the start of the sequence
            y_hat, final_states, _ = model.forward( # We might not need the last context here
                inputs, text, text_mask, initial_hidden, initial_context
            )

        loss = compute_nll_loss(targets, y_hat, mask)

        # Output gradient clipping
        y_hat.register_hook(lambda grad: torch.clamp(grad, -100, 100))

        loss.backward()

        # LSTM params gradient clipping
        if model_type == "prediction":
            nn.utils.clip_grad_value_(model.parameters(), 10)
        else:
            nn.utils.clip_grad_value_(model.embedding.parameters(), 10) # Clip embedding
            nn.utils.clip_grad_value_(model.lstm_1.parameters(), 10)
            nn.utils.clip_grad_value_(model.lstm_2.parameters(), 10)
            nn.utils.clip_grad_value_(model.lstm_3.parameters(), 10)
            # nn.utils.clip_grad_value_(model.transformer_encoder.parameters(), 10) # Clip transformer
            # Note: Clipping output_layer gradients might also be useful
            nn.utils.clip_grad_value_(model.output_layer.parameters(), 10)

        optimizer.step()
        avg_loss += loss.item()

        # print every 10 mini-batches
        if i % 10 == 0:
            print(
                "[{:d}, {:5d}] loss: {:.3f}".format(epoch + 1, i + 1, loss / batch_size)
            )
    avg_loss /= len(train_loader.dataset)

    return avg_loss


def validation(model, valid_loader, device, epoch, model_type):
    avg_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, mini_batch in enumerate(valid_loader):
            if model_type == "prediction":
                inputs, targets, mask = mini_batch
            else:
                inputs, targets, mask, text, text_mask = mini_batch
                text = text.to(device)
                text_mask = text_mask.to(device)

            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            batch_size = inputs.shape[0]

            if model_type == "prediction":
                initial_hidden = model.init_hidden(batch_size, device)
                y_hat, state = model.forward(inputs, initial_hidden)
            else:
                initial_hidden, initial_context = model.init_hidden(
                    batch_size, device
                )
                # Pass the initial context for the start of the sequence
                y_hat, final_states, _ = model.forward( # Might not need last context here
                    inputs, text, text_mask, initial_hidden, initial_context
                )

            loss = compute_nll_loss(targets, y_hat, mask)
            avg_loss += loss.item()

            # print every 10 mini-batches
            if i % 10 == 0:
                print(
                    "[{:d}, {:5d}] loss: {:.3f}".format(
                        epoch + 1, i + 1, loss / batch_size
                    )
                )

    avg_loss /= len(valid_loader.dataset)

    return avg_loss


def train(
    model,
    train_loader,
    valid_loader,
    batch_size,
    n_epochs,
    lr,
    patience,
    step_size,
    device,
    model_type,
    save_path,
    char_seq=None,
):
    model_path = save_path + "best_model_" + model_type + ".pt"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)

    best_loss = math.inf
    best_epoch = 0
    k = 0

    # if the modelfile doesn't exist, create it
    if not os.path.exists(model_path):
        print("creating new model")
        torch.save(model.state_dict(), model_path)
    else:
        print("loading existing model")
        model.load_state_dict(torch.load(model_path))

    # if the synthesis or prediction folders don't exist, create them
    if not os.path.exists(save_path + "/" + model_type):
        os.makedirs(save_path + "/" + model_type)

    # generate one before training for visualization
    if model_type == "prediction":
        gen_seq = generate_unconditional_seq(
            model_path, 700, device, bias=10.0, style=None, prime=False, model_arch='lstm'
        )
    else:
        gen_seq, phi = generate_conditional_sequence(
            model_path=model_path,
            char_seq=char_seq,
            device=device,
            dataset=train_loader.dataset,
            bias=10.0,
            prime=False,
            prime_seq=None,
            real_text=None,
            is_map=False,
            model_arch='lstm',
        )

    # denormalize the generated offsets using train set mean and std
    # gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

    # plot the sequence
    plot_stroke(
        gen_seq[0],
        save_name=save_path + "/" + model_type + "/" + model_type + "_seq_" + str(best_epoch) + ".png",
    )

    # load the train and valid losses if they exist
    if os.path.exists(save_path + "/" + model_type + "/" + model_type + "_train_losses.npy"):
        train_losses = list(np.load(save_path + "/" + model_type + "/" + model_type + "_train_losses.npy"))
        valid_losses = list(np.load(save_path + "/" + model_type + "/" + model_type + "_valid_losses.npy"))
    else:
        train_losses = []
        valid_losses = []
    
    start_epoch = len(train_losses)

    for epoch in range(start_epoch, n_epochs):
        print("training.....")
        train_loss = train_epoch(
            model, optimizer, epoch, train_loader, device, model_type
        )

        print("validation....")
        valid_loss = validation(model, valid_loader, device, epoch, model_type)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # save the losses to a file so they could be re-loaded
        np.save(save_path + "/" + model_type + "/" + model_type + "_train_losses.npy", train_losses)
        np.save(save_path + "/" + model_type + "/" + model_type + "_valid_losses.npy", valid_losses)

        print("Epoch {}: Train: avg. loss: {:.3f}".format(epoch + 1, train_loss))
        print("Epoch {}: Valid: avg. loss: {:.3f}".format(epoch + 1, valid_loss))

        # save a continuiously updating loss graph
        plt.plot(train_losses, label="train")
        plt.plot(valid_losses, label="valid")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss vs. Epoch")
        plt.savefig(save_path + "/" + model_type + "/" + model_type + "_loss.png")
        plt.close()

        if step_size != -1:
            scheduler.step()

        # if True:
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = epoch + 1
            print("Saving best model at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), model_path)
            if model_type == "prediction":
                gen_seq = generate_unconditional_seq(
                    model_path, 700, device, bias=10.0, style=None, prime=False, model_arch='lstm'
                )

            else:
                gen_seq, phi = generate_conditional_sequence(
                    model_path=model_path,
                    char_seq=char_seq,
                    device=device,
                    dataset=train_loader.dataset,
                    bias=10.0,
                    prime=False,
                    prime_seq=None,
                    real_text=None,
                    is_map=True,
                    model_arch='lstm',
                )

                # tokenize char_seq
                char_seq = list(char_seq)
                token_seq = train_dataset.char_to_idx(char_seq)
                # convert back
                token_char_seq = train_dataset.idx_to_char(token_seq)
                print("token_char_seq: ", token_char_seq)

                plt.imshow(phi, cmap="viridis", aspect="auto")
                plt.colorbar()
                plt.xlabel("time steps")
                plt.yticks(
                    np.arange(phi.shape[0]),
                    token_char_seq,
                    rotation="horizontal",
                )
                plt.margins(0.2)
                plt.subplots_adjust(bottom=0.15)
                plt.savefig(save_path + "/" + model_type + "/" + "heat_map" + str(best_epoch) + ".png")
                plt.close()
            # denormalize the generated offsets using train set mean and std
            gen_seq = data_denormalization(Global.train_mean, Global.train_std, gen_seq)

            # plot the sequence
            plot_stroke(
                gen_seq[0],
                save_name=save_path + "/" + model_type + "/" + model_type + "_seq_" + str(best_epoch) + ".png",
            )

            # if model_type == "prediction":
            #     # run the model once to get the full y_hat tensor
            #     seq_tensor = torch.from_numpy(gen_seq).float().to(device)
            #     with torch.no_grad():
            #         init_h = model.init_hidden(1, device)
            #         y_hat_full, _ = model.forward(seq_tensor, init_h)

            #     visualize_mdn_overlay(
            #         gen_seq[0],
            #         y_hat_full,
            #         save_path + "/" + model_type + "/" + "mdn_overlay_" + str(best_epoch) + ".png",
            #     )

            k = 0
        elif k > patience:
            print("Best model was saved at epoch: {}".format(best_epoch))
            print("Early stopping at epoch {}".format(epoch))
            break
        else:
            k += 1


if __name__ == "__main__":

    args = argparser()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Arguments: {}".format(args))
    model_type = args.model_type
    batch_size = args.batch_size
    n_epochs = args.n_epochs

    # Load the data and text
    train_dataset = MathHandwritingDataset(
    # train_dataset = HandwritingDataset(
        args.data_path,
        split="train",
        text_req=args.text_req,
        debug=args.debug,
        data_aug=args.data_aug,
    )
    valid_dataset = MathHandwritingDataset(
    # valid_dataset = HandwritingDataset(
        args.data_path,
        split="valid",
        text_req=args.text_req,
        debug=args.debug,
        data_aug=args.data_aug,
    )

    # cut it down by a bunch for testing
    # train_dataset = Subset(train_dataset, random.sample(range(len(train_dataset)), int(0.2 * len(train_dataset))))

    # train_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    sample_batch = next(iter(train_loader))
    single_sample = sample_batch[0]
    with open(args.save_path + "sample.txt", "w") as f:
        sample_shape = single_sample.shape
        f.write("Single sample shape: {}\n".format(sample_shape))
        for i, stroke in enumerate(single_sample):
            f.write("Stroke {}:\n".format(i))
            for j, point in enumerate(stroke):
                f.write("\tPoint {}: {}\n".format(j, point))

    if model_type == "prediction":
        model = HandWritingPredictionNet(
            n_layers=3, output_size=121, input_size=3
        )
    elif model_type == "synthesis":
        model = HandWritingSynthesisNet(
            hidden_size=args.hidden_size,
            n_layers=3,
            output_size=121,
            vocab_size=train_dataset.vocab_size,
        )
    train(
        model,
        train_loader,
        valid_loader,
        batch_size,
        n_epochs,
        args.lr,
        args.patience,
        args.step_size,
        device,
        model_type,
        args.save_path,
        args.char_seq,
    )
