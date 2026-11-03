"""CSV batch classification with validation, limits and no persistence.

Uploaded complaints are held in memory for the life of one request and written
only to a temporary file the user downloads; nothing is logged or stored.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LIKELY_TEXT_COLUMNS = (
    "narrative", "complaint", "complaint_text", "text", "description",
    "consumer_complaint_narrative", "body", "message",
)


class BatchError(ValueError):
    """Raised with a message intended to be shown to the user as-is."""


@dataclass
class BatchResult:
    frame: pd.DataFrame
    n_input_rows: int
    n_classified: int
    n_skipped: int
    n_review: int
    text_column: str
    warnings: list[str] = field(default_factory=list)

    def category_summary(self) -> pd.DataFrame:
        counts = self.frame["predicted_category"].value_counts()
        return pd.DataFrame(
            {
                "category": counts.index,
                "complaints": counts.to_numpy(),
                "share": (counts / max(len(self.frame), 1)).round(4).to_numpy(),
            }
        )


def detect_text_column(df: pd.DataFrame) -> str:
    lowered = {c.lower().strip(): c for c in df.columns}
    for candidate in LIKELY_TEXT_COLUMNS:
        if candidate in lowered:
            return lowered[candidate]
    # Fall back to the object column with the longest average content.
    object_cols = [c for c in df.columns if df[c].dtype == object]
    if not object_cols:
        raise BatchError(
            "No text column found. Include a column named 'narrative', "
            "'complaint' or 'text'."
        )
    lengths = {c: df[c].astype(str).str.len().mean() for c in object_cols}
    best = max(lengths, key=lengths.get)
    if lengths[best] < 20:
        raise BatchError(
            "No column looks like complaint text (the longest text column "
            f"averages {lengths[best]:.0f} characters)."
        )
    return best


def read_csv_safely(path: str | Path, max_mb: float, max_rows: int) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    warnings: list[str] = []
    if not path.exists():
        raise BatchError("The uploaded file could not be read. Please try again.")
    size_mb = path.stat().st_size / 1e6
    if size_mb > max_mb:
        raise BatchError(f"File is {size_mb:.1f} MB; the limit is {max_mb:.0f} MB.")
    if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
        raise BatchError(f"Expected a .csv file, got '{path.suffix or 'no extension'}'.")

    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding, on_bad_lines="skip", engine="python")
            if encoding != "utf-8":
                warnings.append(f"File was not valid UTF-8; read it as {encoding}.")
            break
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise BatchError("The file is empty.") from exc
        except csv.Error as exc:
            raise BatchError(f"The file is not valid CSV: {exc}") from exc
    else:
        raise BatchError(
            "Could not decode the file. Save it as UTF-8 CSV and try again."
        ) from last_error

    if df.empty or not len(df.columns):
        raise BatchError("The file has no rows.")
    if len(df) > max_rows:
        warnings.append(f"File had {len(df):,} rows; only the first {max_rows:,} were classified.")
        df = df.head(max_rows)
    return df, warnings


def classify_frame(
    df: pd.DataFrame,
    predictor,
    text_column: str | None = None,
    batch_size: int = 32,
    warnings: list[str] | None = None,
) -> BatchResult:
    warnings = list(warnings or [])
    n_input = len(df)
    column = text_column or detect_text_column(df)
    if column not in df.columns:
        raise BatchError(f"Column '{column}' is not in the file: {list(df.columns)}")

    texts = df[column].astype(str).str.strip()
    valid = texts.str.len() >= 3
    n_skipped = int((~valid).sum())
    if n_skipped:
        warnings.append(f"{n_skipped:,} row(s) had empty or too-short text and were skipped.")
    if not valid.any():
        raise BatchError("No row contained usable complaint text.")

    subset = df.loc[valid].copy()
    predictions = predictor.predict_batch(texts[valid].tolist(), batch_size=batch_size)

    subset["predicted_category"] = [p.display for p in predictions]
    subset["predicted_label"] = [p.label for p in predictions]
    subset["confidence"] = [round(p.confidence, 4) for p in predictions]
    subset["routing"] = [
        "Manual review recommended" if p.needs_review else "Automated classification available"
        for p in predictions
    ]
    top3 = []
    for p in predictions:
        top3.append("; ".join(f"{n}:{v:.2f}" for n, v in p.top_k(3)))
    subset["top_3"] = top3

    return BatchResult(
        frame=subset,
        n_input_rows=n_input,
        n_classified=int(valid.sum()),
        n_skipped=n_skipped,
        n_review=int(sum(p.needs_review for p in predictions)),
        text_column=column,
        warnings=warnings,
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
