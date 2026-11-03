from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from seq2seq.transformer.transformer import Transformer
from seq2seq.data.fr_en import FrEnDataset, collate_fn, tokenizer

run = wandb.init(
    entity="<INSERT ENTITY HERE>",
    project="transformer",
    config={
        "learning_rate": 0.00005,
        "architecture": "transformer",
        "dataset": "fr-en-euro",
        "epochs": 10,
    },
)


def decode(model, src_sentence, max_len=100, device="cpu"):
    model.eval()
    src_tensor = tokenizer.encode(src_sentence).to(device)

    tgt_tokens = [tokenizer.bos_token_id]

    for _ in tqdm(range(max_len)):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(src_tensor.unsqueeze(0), tgt_tensor)

        next_token_logits = output[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(next_token_probs, num_samples=1).item()

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
        torch.save(checkpoint, "fr_en_euro_latest.pt")
    else:
        torch.save(checkpoint, f"fr_en_euro_{epoch}.pt")


def train_nmt():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path("data/nmt/europarl/")
    dataset = FrEnDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 4 * embedding_dim  # standard practice
    qk_length = embedding_dim // num_heads # standard practice (note that qk_length is a per-head dqk)
    value_length = embedding_dim // num_heads # standard practice
    max_length = 200
    dropout = 0.1
    epochs = 1

    warmup_steps = 4000
    base_lr = 5e-5

    def lr_lambda(step):
        if step == 0:
            step = 1  # avoid div by zero
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return (warmup_steps**0.5) / (step**0.5)

    model = Transformer(
        pad_idx=tokenizer.pad_token_id,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        qk_length=qk_length,
        max_length=max_length,
        value_length=value_length,
        dropout=dropout,
        device=device,
    ).to(device)

    # TODO: loss shouldn't include pad tokens, so it should ignore pad token ids
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, betas=[0.9, 0.98], eps=1e-9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # train over all epochs, checkpointing every 25 epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        data_tqdm = tqdm(dataloader)
        for i, (src, tgt) in enumerate(data_tqdm):
            try:
                src, tgt = src.to(device), tgt.to(device)

                tgt_input = tgt[:, :-1]

                # TODO: if the input is up to the second-last token,
                # what should the output be?
                tgt_output = tgt[:, 1:]

                optimizer.zero_grad()

                output = model(src, tgt_input)

                loss = criterion(output.reshape(-1, vocab_size), tgt_output.reshape(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                data_tqdm.set_postfix({"loss": loss})
                run.log({"loss": loss})
            except Exception as e:
                print(e)
                continue

            if i % 1000 == 0:
                print("Saving checkpoint...")
                save_checkpoint(epoch, model, optimizer, scheduler)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: Loss - {avg_loss}")

    model.eval()
    test_sentences = [
        "où est le restaurant?",
        "je suis un étudiant.",
        "quel temps fait-il aujourd'hui?",
    ]

    for sentence in test_sentences:
        translation = decode(model, sentence, max_len=max_length, device=device)
        print(f"French: {sentence}")
        print(f"English: {translation}")
        print("-" * 20)


if __name__ == "__main__":
    train_nmt()
