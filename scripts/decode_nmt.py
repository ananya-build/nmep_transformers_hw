from tqdm import tqdm

import torch

from seq2seq.transformer.transformer import Transformer
from seq2seq.data.fr_en import tokenizer


def decode(model, src_sentence, max_len=100, device="cpu", mode="top_p"):
    model.eval()
    # wrap the source in BOS/EOS the same way FrEnDataset does during training
    src_tensor = torch.cat(
        [
            torch.tensor([tokenizer.bos_token_id]),
            tokenizer.encode(src_sentence),
            torch.tensor([tokenizer.eos_token_id]),
        ]
    ).to(device)

    tgt_tokens = [tokenizer.bos_token_id]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_tokens]).to(device)
        with torch.no_grad():
            output = model(src_tensor.unsqueeze(0), tgt_tensor)

        next_token_logits = output[0, -1, :]

        if mode == "top_k":
            # top-k
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, 20)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('inf')

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        elif mode == "top_p":
            # top-p
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        elif mode == "greedy":
            next_token = torch.argmax(next_token_logits).item()

        if next_token == tokenizer.eos_token_id:
            break

        tgt_tokens.append(next_token)

    return tokenizer.decode(torch.tensor(tgt_tokens))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration
    vocab_size = len(tokenizer.vocab)
    num_layers = 6
    num_heads = 8
    embedding_dim = 512
    ffn_hidden_dim = 4 * embedding_dim
    qk_length = embedding_dim // num_heads
    value_length = embedding_dim // num_heads
    max_length = 200
    dropout = 0.1

    # Instantiate the model
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

    # Load the trained model weights
    model_path = "fr_en_euro_latest.pt"
    try:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)["model"]
        )
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure the model file exists and the path is correct.")
        return

    model.eval()

    # Sentences to translate (from data/nmt/en-fr-small.csv)
    fr_sentences = [
        "Le Parlement européen salue les décisions prises par la Commission européenne, telles que présentées dans ce rapport, y compris celle qui exige, dans un cas précis, le remboursement des sommes allouées et applique donc l'article 88 du traité CECA.",
        "On sait que jusqu'à présent, le Conseil a refusé d'adopter un tel règlement.",
        "Il ne faudrait pas que le nouveau modèle expérimenté à l' heure actuelle par la Commission ait pour conséquence un pur processus de nationalisation, qui annulerait les effets obtenus par notre politique de concurrence.",
        "Si nous voulons qu'une culture juridique existe en Europe, il va sans dire que le droit ne peut être appliqué par la seule Commission, par des organes centraux, mais qu'il doit aussi l'être par les autorités nationales, par les tribunaux nationaux.",
        "Il importe d'assurer la cohérence entre les objectifs de la politique agricole commune et ceux de la politique environnementale, notamment en matière de biodiversité.",
    "La mise en œuvre des mesures transitoires doit être rigoureusement surveillée afin de garantir une concurrence loyale et non faussée sur le marché intérieur.",
    "Toute modification du cadre financier pluriannuel requiert l'accord unanime du Conseil, après consultation et approbation du Parlement européen.",
    "Nous devons renforcer la coopération avec les pays tiers pour relever les défis migratoires et œuvrer conjointement à la stabilité régionale.",
    ]

    en_sentences = [
        "The European Parliament welcomes the decisions taken by the Commission, as set out in this report, including the one which, in a specific case, demands the repayment of the sums allocated and therefore applies Article 88 of the ECSC Treaty.",
        "We know that the Council has so far refused to adopt such a regulation.",
        "The new model currently being tested by the Commission should not result in a pure process of nationalisation which would undo the effects achieved by our competition policy.",
        "If we want a legal culture to exist in Europe, it goes without saying that the law cannot be applied only by the Commission, by central bodies, but must also be applied by the national authorities, by the national courts.",
        "It is important to ensure coherence between the objectives of the Common Agricultural Policy and those of environmental policy, particularly concerning biodiversity.",
    "The implementation of transitional measures must be rigorously monitored to guarantee fair and undistorted competition within the internal market.",
    "Any modification to the multiannual financial framework requires the unanimous agreement of the Council, following consultation and approval by the European Parliament.",
    "We must strengthen cooperation with third countries to address migratory challenges and work jointly towards regional stability.",
    ]

    for fr_sentence, en_sentence in zip(fr_sentences, en_sentences):
        translation = decode(model, fr_sentence, max_len=max_length, device=device)
        print(f"French: {fr_sentence}")
        print(f"Ground Truth English: {en_sentence}")
        print(f"Model Translation: {translation}")
        print("-" * 20)


if __name__ == "__main__":
    main()
