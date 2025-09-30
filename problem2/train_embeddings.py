import argparse, json, os, re, time
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if len(w) >= 2]
    return words

def build_vocab(texts, k = 5000):
    cnt = Counter()
    for t in texts:
        cnt.update(clean_text(t))
    most = cnt.most_common(k)
    vocab_to_idx = {w:i+1 for i,(w,_) in enumerate(most)}
    idx_to_vocab = {str(i):w for w,i in vocab_to_idx.items()}
    return vocab_to_idx, idx_to_vocab, sum(cnt.values())

def texts_to_bow(texts, vocab_to_idx, vocab_size):
    bows = torch.zeros((len(texts), vocab_size), dtype=torch.float32)
    for i, t in enumerate(texts):
        for w in clean_text(t):
            j = vocab_to_idx.get(w, 0)
            if j>0:
                bows[i, j-1] += 1.0
    bows.clamp_(max=1.0)
    return bows


class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_abstracts(papers_path):
    with open(papers_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids, texts = [], []
    for p in data:
        ids.append(p.get("arxiv_id") or p.get("id") or str(len(ids)))
        texts.append(p.get("abstract",""))
    return ids, texts

def save_model(model, vocab_to_idx, cfg, outdir):
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": vocab_to_idx,
        "model_config": cfg
    }, os.path.join(outdir, "model.pth"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("output_dir")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--vocab_size", type=int, default=5000)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--embed_dim", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading abstracts from {args.input_json}")
    ids, texts = load_abstracts(args.input_json)
    print(f"Found {len(texts)} abstracts.")

    print("Building vocabulary")
    v2i, i2v, total_tokens = build_vocab(texts, args.vocab_size)
    vocab_size = len(v2i)
    print(f"Vocabulary size: {vocab_size} (from {total_tokens} tokens)")

    print("Encoding abstracts to BoW")
    X = texts_to_bow(texts, v2i, vocab_size)
    ds = TensorDataset(X, X)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = TextAutoencoder(vocab_size, args.hidden_dim, args.embed_dim)
    total_params = count_params(model)
    arch = f"{vocab_size} -> {args.hidden_dim} -> {args.embed_dim} -> {args.hidden_dim} -> {vocab_size}"
    if total_params > 2000000:
        raise SystemExit("Parameter budget exceeded.")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print("Training autoencoder")
    start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for ep in range(1, args.epochs+1):
        model.train()
        loss_sum = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            y_hat, _ = model(xb)
            loss = loss_fn(y_hat, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
        avg = loss_sum / len(ds)
        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            print(f"Epoch {ep}/{args.epochs}, Loss: {avg:.4f}")
    end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    model.eval()
    with torch.no_grad():
        _, Z = model(X)
    embeddings = []
    for i, arxid in enumerate(ids):
        embeddings.append({
            "arxiv_id": arxid,
            "embedding": Z[i].tolist(),
            "reconstruction_loss": float(((model.decoder(Z[i]) - X[i]).abs()).mean().item())
        })


    save_model(model, v2i, {
        "vocab_size": vocab_size,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embed_dim
    }, args.output_dir)

    with open(os.path.join(args.output_dir, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "vocab_to_idx": v2i,
            "idx_to_vocab": i2v,
            "vocab_size": len(v2i),
            "total_words": total_tokens
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump({
            "start_time": start,
            "end_time": end,
            "epochs": args.epochs,
            "final_loss": float(avg),
            "total_parameters": total_params,
            "papers_processed": len(texts),
            "embedding_dimension": args.embed_dim
        }, f, ensure_ascii=False, indent=2)

    print("Training complete. Files saved to", args.output_dir)

if __name__ == "__main__":
    main()
