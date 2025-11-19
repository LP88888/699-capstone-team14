import json

import pandas as pd
from IPython.display import display, HTML
from html import escape


def preview_side_by_side(df_wide: pd.DataFrame, text_col: str, n: int = 8):
    """Simple tabular 'original vs cleaned' preview."""
    cols = [text_col, "NER_raw", "NER_clean"] + (
        ["Ingredients"] if "Ingredients" in df_wide.columns else []
    )
    display(df_wide.loc[:, cols].head(n))


def _render_marked(text: str, spans: list[dict]) -> str:
    """Mark entities inline; tooltip shows norm/canonical/id for quick QA."""
    spans = sorted(spans, key=lambda r: r["start"])
    pos = 0
    out = []
    for r in spans:
        out.append(escape(text[pos: r["start"]]))
        frag = escape(text[r["start"]: r["end"]])
        tip = (
            f'norm="{r["norm"]}" | canonical="{r["canonical"]}" | '
            f'id={r["id"] if r["id"] is not None else "-"}'
        )
        out.append(f'<mark title="{escape(tip)}">{frag}</mark>')
        pos = r["end"]
    out.append(escape(text[pos:]))
    return "".join(out)


def html_preview(df_wide: pd.DataFrame, text_col: str, n: int = 8):
    """Inline HTML with highlighted entities and cleaned list below."""
    rows = []
    for _, row in df_wide.head(n).iterrows():
        spans = json.loads(row["spans_json"])
        marked = _render_marked(row[text_col], spans)
        cleaned = ", ".join(row.get("NER_clean") or [])
        rows.append(
            f"""
        <div class="one">
          <div class="orig">{marked}</div>
          <div class="clean"><strong>NER_clean:</strong> {escape(cleaned)}</div>
        </div>
        """
        )
    style = """
    <style>
      .one{border:1px solid #ddd; padding:10px; margin:8px 0; border-radius:6px;}
      .orig{margin-bottom:6px; line-height:1.5}
      mark{padding:0 2px; border-radius:3px}
      .clean{font-family:monospace}
    </style>
    """
    display(HTML(style + "\n".join(rows)))


def describe_predictions(df_wide: pd.DataFrame, top_k: int = 20):
    """Small summary to sanity-check output distribution."""
    s = df_wide["NER_clean"].explode().value_counts().head(top_k)
    print(
        f"Rows: {len(df_wide):,} | "
        f"rows with ≥1 pred: {(df_wide['NER_clean'].map(bool)).sum():,}"
    )
    print(
        f"Mean #unique preds/row: {df_wide['NER_clean'].map(len).mean():.2f}"
    )
    display(s.to_frame("freq"))


__all__ = ["preview_side_by_side", "html_preview", "describe_predictions"]
