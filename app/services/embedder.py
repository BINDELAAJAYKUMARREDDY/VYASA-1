from typing import List

import torch
from transformers import AutoModel, AutoTokenizer


class LocalTextEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        if device == "cuda":
            # Simple speed boost for indexing/inference on GPU.
            self.model = self.model.half()
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}
            with torch.no_grad():
                model_out = self.model(**toks)
                emb = _mean_pool(model_out.last_hidden_state, toks["attention_mask"])
                if normalize_embeddings:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.extend(emb.detach().cpu().tolist())
        return out


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = torch.sum(masked, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
