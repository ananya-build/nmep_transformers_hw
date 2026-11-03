import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from seq2seq.transformer.transformer import Decoder
from seq2seq.data.screenplay import ScreenplayDataset, collate_fn, tokenizer

run = wandb.init(
    entity="<INSERT ENTITY HERE>",
    project="transformer",
    config={
        "learning_rate": 0.00005,
        "architecture": "transformer-lm-gpt",
        "dataset": "screenplay",
        "epochs": 100,
    },
)


def decode(model, src_sentence, max_len=100, device="cpu"):
    model.eval()
    tgt_tokens = [tokenizer.bos_token_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(tgt_tensor)

        next_token_logits = output[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def save_checkpoint(epoch: int, model, optimizer, scheduler, latest=True):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    if latest:
        torch.save(checkpoint, "screenplay_lm_gpt_latest.pt")
    else:
        torch.save(checkpoint, f"screenplay_lm_gpt_{epoch}.pt")


def make_pad_mask(q, k):
    # k: (B, T_k)
    # returns: (B, 1, 1, T_k)
    pad_mask = k.eq(tokenizer.pad_token_id).unsqueeze(1).unsqueeze(1)
    return pad_mask


def make_no_peak_mask(q, k, device=0):
    # Create a look-ahead mask to prevent attending to future tokens
    len_q, len_k = q.size(1), k.size(1)
    mask = torch.triu(
        torch.ones(len_q, len_k, device=device, dtype=torch.bool), diagonal=1
    )
    return mask


def train_lm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    block_size = 512  # each sample has a fixed length of block_size input tokens
    batch_size = 16

    data_path = Path("data/lm/")
    dataset = ScreenplayDataset(data_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 4 * embedding_dim  # standard practice
    qk_length = embedding_dim // num_heads # standard practice (note that qk_length is a per-head dqk)
    value_length = embedding_dim // num_heads # standard practice
    max_length = block_size  # no point in having max_length > block_size, because all samples are block_size length
    dropout = 0.1
    epochs = 100

    warmup_steps = 4000
    base_lr = 3e-4

    def lr_lambda(step):
        if step == 0:
            step = 1  # avoid div by zero
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return (warmup_steps**0.5) / (step**0.5)

    model = Decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        qk_length=qk_length,
        max_length=max_length,
        value_length=value_length,
        dropout=dropout,
    ).to(device)

    # TODO: loss shouldn't include pad tokens, so it should ignore pad token ids
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    global_step = 0

    # train over all epochs, checkpointing every 25 epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        data_tqdm = tqdm(dataloader)
        for i, batch in enumerate(data_tqdm):
            try:
                batch = batch.to(device)

                batch_input = batch[:, :-1]
                batch_output = batch[:, 1:]

                optimizer.zero_grad()

                trg_pad_mask = make_pad_mask(batch_input, batch_input)
                trg_no_peak_mask = make_no_peak_mask(batch_input, batch_input)

                trg_mask = trg_pad_mask | trg_no_peak_mask

                output = model(batch_input, tgt_mask=trg_mask)

                loss = criterion(
                    output.reshape(-1, vocab_size), batch_output.reshape(-1)
                )

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping

                optimizer.step()
                scheduler.step()

                global_step += 1
                total_loss += loss.item()

                data_tqdm.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    norm=f"{norm}"
                )

                run.log(
                    {
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "norm": norm
                    }
                )
            except Exception as e:
                print(e)

            if global_step % 5000 == 0 and i > 0:
                print("Saving checkpoint...")
                save_checkpoint(epoch, model, optimizer, scheduler)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss - {avg_loss}")

        if (epoch + 1) % 25 == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, latest=False)


if __name__ == "__main__":
    train_lm()
