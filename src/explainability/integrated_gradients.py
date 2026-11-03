"""Token attributions for the DistilBERT classifier via Layer Integrated
Gradients (captum) over the word-embedding layer.

Why this method: it is gradient-exact for the model we actually deploy, needs no
surrogate model (unlike LIME) and no background dataset (unlike kernel SHAP), and
runs on CPU inside a free Space if the number of interpolation steps is kept
small. What it gives you is the sensitivity of one logit to each token embedding
along a straight path from a baseline of [PAD] tokens — not the model's reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TokenAttribution:
    token: str
    score: float      # signed, normalised to [-1, 1] by the largest magnitude
    raw: float


@dataclass
class Explanation:
    label: str
    confidence: float
    tokens: list[TokenAttribution]
    steps: int
    truncated: bool
    method: str = "Layer Integrated Gradients (captum) over the embedding layer"
    caveat: str = (
        "Attributions show which tokens the prediction is locally sensitive to. "
        "They are an approximation of influence, not an account of the model's "
        "reasoning, and they do not establish that a prediction is correct."
    )

    def top_positive(self, k: int = 10) -> list[TokenAttribution]:
        return sorted(self.tokens, key=lambda t: t.score, reverse=True)[:k]

    def highlight_pairs(self) -> list[tuple[str, float]]:
        """Shape Gradio's HighlightedText expects: (token, score)."""
        return [(t.token + " ", t.score) for t in self.tokens]


class IntegratedGradientsExplainer:
    def __init__(self, predictor, steps: int = 16, max_chars: int = 1200) -> None:
        """`steps` trades fidelity for latency; 16 keeps a CPU explanation inside
        a couple of seconds for a 256-token complaint."""
        import torch
        from captum.attr import LayerIntegratedGradients

        self.predictor = predictor
        self.steps = steps
        self.max_chars = max_chars
        self._torch = torch

        model = predictor.model
        self.embedding_layer = model.get_input_embeddings()

        def forward(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask).logits

        self.lig = LayerIntegratedGradients(forward, self.embedding_layer)

    def explain(self, text: str, target: int | None = None) -> Explanation:
        torch = self._torch
        tok = self.predictor.tokenizer
        truncated = len(text) > self.max_chars
        text = text[: self.max_chars]

        enc = tok(
            text,
            truncation=True,
            max_length=self.predictor.max_len,
            return_tensors="pt",
        ).to(self.predictor.device)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            logits = self.predictor.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            probs = torch.softmax(logits.float(), dim=-1)[0]
        if target is None:
            target = int(probs.argmax().item())

        # Baseline: everything replaced by [PAD], special tokens kept in place,
        # so the attribution is "this token vs. nothing" rather than "vs. [CLS]".
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        baseline = torch.full_like(input_ids, pad_id)
        special = torch.tensor(
            tok.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True),
            device=input_ids.device,
            dtype=torch.bool,
        )
        baseline[0][special] = input_ids[0][special]

        attributions = self.lig.attribute(
            inputs=input_ids,
            baselines=baseline,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=self.steps,
            internal_batch_size=max(self.steps // 4, 1),
        )
        # Sum over the embedding dimension -> one signed score per token.
        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        tokens = tok.convert_ids_to_tokens(input_ids[0].tolist())
        keep = [
            i
            for i, t in enumerate(tokens)
            if t not in (tok.cls_token, tok.sep_token, tok.pad_token)
        ]
        raw = scores[keep]
        scale = float(np.abs(raw).max()) or 1.0

        # Re-join WordPiece continuations so the UI shows words, not "##ing".
        merged: list[TokenAttribution] = []
        for idx, pos in enumerate(keep):
            piece = tokens[pos]
            value = float(raw[idx])
            if piece.startswith("##") and merged:
                merged[-1] = TokenAttribution(
                    token=merged[-1].token + piece[2:],
                    score=merged[-1].score + value / scale,
                    raw=merged[-1].raw + value,
                )
            else:
                merged.append(
                    TokenAttribution(token=piece, score=value / scale, raw=value)
                )
        # Re-normalise after merging so the colour scale still spans [-1, 1].
        scale2 = max((abs(t.score) for t in merged), default=1.0) or 1.0
        merged = [
            TokenAttribution(token=t.token, score=t.score / scale2, raw=t.raw) for t in merged
        ]

        return Explanation(
            label=self.predictor.labels[target],
            confidence=float(probs[target].item()),
            tokens=merged,
            steps=self.steps,
            truncated=truncated,
        )
