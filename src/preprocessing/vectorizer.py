"""Word-index vectoriser for the recurrent models.

Equivalent in behaviour to the Keras Tokenizer the referenced project used
(top-N vocabulary, OOV token, post padding/truncation) but with no TensorFlow
dependency and an explicit JSON on-disk format. The vocabulary is fitted on the
training split only.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PAD, OOV = "<PAD>", "<OOV>"
_TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Light tokenisation only. The source narratives are already lowercased and
    stripped of punctuation, so anything heavier would just discard signal."""
    return _TOKEN.findall(str(text).lower())


class WordVectorizer:
    def __init__(self, vocab_size: int = 20000, max_len: int = 200) -> None:
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word_index: dict[str, int] = {}

    def fit(self, texts: Iterable[str]) -> "WordVectorizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        # index 0 = PAD, 1 = OOV, then the most frequent words.
        self.word_index = {PAD: 0, OOV: 1}
        for word, _ in counts.most_common(max(self.vocab_size - 2, 0)):
            self.word_index[word] = len(self.word_index)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_len), dtype=np.int64)
        oov = self.word_index.get(OOV, 1)
        for row, text in enumerate(texts):
            ids = [self.word_index.get(tok, oov) for tok in tokenize(text)][: self.max_len]
            out[row, : len(ids)] = ids  # post padding, post truncation
        return out

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def effective_vocab_size(self) -> int:
        return len(self.word_index)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(
                {
                    "vocab_size": self.vocab_size,
                    "max_len": self.max_len,
                    "word_index": self.word_index,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordVectorizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(vocab_size=blob["vocab_size"], max_len=blob["max_len"])
        obj.word_index = blob["word_index"]
        return obj
