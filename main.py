"""
NoteLM — NotebookLM-grade research assistant
FastAPI + Gemini 1.5 Flash backend
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
from docx import Document
from gtts import gTTS
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import pdfplumber
from pptx import Presentation as PptxPresentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt, Emu
from google import genai

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
UPLOADS    = BASE_DIR / "uploads"
OUTPUTS    = BASE_DIR / "outputs"
STATIC     = BASE_DIR / "static"
DATA_FILE  = BASE_DIR / "sources.json"

for d in (UPLOADS, OUTPUTS, STATIC):
    d.mkdir(exist_ok=True)

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyBvnlnYaex2EecyzuJ9N3t1yInFII4U_zw"
)
_client = genai.Client(api_key=GEMINI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _load() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            pass
    return {}

def _save(db: dict) -> None:
    DATA_FILE.write_text(json.dumps(db, indent=2, default=str))

sources_db: dict = _load()

# ──────────────────────────────────────────────────────────────────────────────
# Text extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return "\n\n".join(pages)

def extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_txt(path: Path) -> str:
    return path.read_text(errors="ignore")

async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", r.text)
        text = re.sub(r"\s+", " ", text)
        return text[:60000]

# ──────────────────────────────────────────────────────────────────────────────
# Gemini helpers
# ──────────────────────────────────────────────────────────────────────────────

def _gemini(prompt: str) -> str:
    resp = _client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    return resp.text.strip()

def _gemini_json(prompt: str) -> dict | list:
    raw = _gemini(prompt)
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())

# ──────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyze(content: str, name: str, lang: str = 'ru') -> dict:
    lang_hint = "Russian" if lang == "ru" else "English"
    prompt = f"""You are a research analyst. Analyze the document titled "{name}".
IMPORTANT: All content in your response must be in {lang_hint} language.
Return ONLY a valid JSON object (no markdown) with exactly these fields:

{{
  "summary": "Comprehensive 3-4 paragraph executive summary",
  "key_points": ["concise key point 1", "key point 2", ...],
  "topics": [
    {{
      "title": "Topic Name",
      "description": "2-3 sentence description",
      "points": ["sub-point 1", "sub-point 2"]
    }}
  ],
  "notable_quotes": ["exact quote 1", ...],
  "entities": {{
    "people": ["name1", ...],
    "organizations": ["org1", ...],
    "places": ["place1", ...],
    "dates": ["date1", ...]
  }},
  "sentiment": "positive|negative|neutral|mixed",
  "complexity": "beginner|intermediate|advanced",
  "word_count": 1234,
  "language": "English"
}}

Document content (first 28000 chars):
{content[:28000]}"""

    try:
        return _gemini_json(prompt)
    except Exception as e:
        return {
            "summary": "Analysis failed. Raw content preview: " + content[:300],
            "key_points": [],
            "topics": [],
            "notable_quotes": [],
            "entities": {"people": [], "organizations": [], "places": [], "dates": []},
            "sentiment": "neutral",
            "complexity": "intermediate",
            "word_count": len(content.split()),
            "language": "English",
            "_error": str(e),
        }

# ──────────────────────────────────────────────────────────────────────────────
# Output generators
# ──────────────────────────────────────────────────────────────────────────────

# Colour palette
_DARK  = "#0F172A"
_MID   = "#334155"
_ACCENT= "#6366F1"
_LIGHT = "#F8FAFC"
_MUTED = "#94A3B8"

def gen_presentation(source: dict, out: Path, lang: str = 'ru') -> None:
    a = source.get("analysis") or {}
    prs = PptxPresentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    titles = {
        "ru": {
            "overview": "AI-Generated Обзор исследования",
            "summary": "Резюме",
            "key_points": "Ключевые моменты",
            "entities": "Ключевые сущности",
            "quotes": "Примечательные цитаты",
        },
        "en": {
            "overview": "AI-Generated Research Overview",
            "summary": "Summary",
            "key_points": "Key Points",
            "entities": "Key Entities",
            "quotes": "Notable Quotes",
        }
    }
    t = titles.get(lang, titles["ru"])

    def _blank():
        blank = prs.slide_layouts[6]  # completely blank
        return prs.slides.add_slide(blank)

    # Named color → hex map for convenience
    _NAMED_COLORS = {
        "white": "#FFFFFF",
        "black": "#000000",
        "red":   "#FF0000",
        "green": "#00FF00",
        "blue":  "#0000FF",
    }

    def _parse_color(color: str):
        """Return (r, g, b) tuple from a hex color string or CSS named color."""
        c = _NAMED_COLORS.get(str(color).lower(), color)
        c = c.lstrip('#')
        if len(c) == 3:
            c = c[0]*2 + c[1]*2 + c[2]*2
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    def _bg(slide, color: str):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        r, g, b = _parse_color(color)
        fill.fore_color.rgb = RGBColor(r, g, b)

    def _txt(slide, text, left, top, width, height,
             bold=False, size=18, color="#0F172A", align="LEFT"):
        from pptx.enum.text import PP_ALIGN
        box = slide.shapes.add_textbox(
            Inches(left), Inches(top), Inches(width), Inches(height)
        )
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.bold = bold
        p.font.size = Pt(size)
        r, g, b = _parse_color(color)
        p.font.color.rgb = RGBColor(r, g, b)
        p.alignment = {"LEFT": PP_ALIGN.LEFT, "CENTER": PP_ALIGN.CENTER,
                       "RIGHT": PP_ALIGN.RIGHT}.get(align, PP_ALIGN.LEFT)
        return box

    def _rect(slide, left, top, width, height, color):
        r, g, b = _parse_color(color)
        shape = slide.shapes.add_shape(
            1,  # MSO_SHAPE_TYPE.RECTANGLE
            Inches(left), Inches(top), Inches(width), Inches(height)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(r, g, b)
        shape.line.fill.background()
        return shape

    # ── 1. Title slide
    slide = _blank()
    _bg(slide, _DARK)
    _rect(slide, 0, 3.2, 13.33, 0.06, _ACCENT)
    _txt(slide, source.get("name", "Untitled"),
         0.8, 1.2, 11.7, 2.0, bold=True, size=40, color=_LIGHT, align="LEFT")
    _txt(slide, t["overview"],
         0.8, 3.5, 10, 0.6, size=20, color=_MUTED)
    _txt(slide, datetime.now().strftime("%B %d, %Y"),
         0.8, 4.3, 6, 0.5, size=16, color=_MUTED)

    # ── 2. Summary
    slide = _blank()
    _bg(slide, _LIGHT)
    _rect(slide, 0, 0, 13.33, 1.1, _ACCENT)
    _txt(slide, t["summary"], 0.6, 0.2, 12, 0.7, bold=True, size=28, color="white")
    summary = a.get("summary", "")
    # wrap long summary
    wrapped = textwrap.fill(summary, 110)[:900]
    _txt(slide, wrapped, 0.6, 1.3, 12.1, 5.8, size=17, color=_DARK)

    # ── 3. Key Points
    kps = a.get("key_points", [])
    if kps:
        slide = _blank()
        _bg(slide, _LIGHT)
        _rect(slide, 0, 0, 13.33, 1.1, "#4F46E5")
        _txt(slide, t["key_points"], 0.6, 0.2, 12, 0.7, bold=True, size=28, color="white")
        for i, pt in enumerate(kps[:7]):
            y = 1.35 + i * 0.78
            _rect(slide, 0.5, y, 0.35, 0.35, _ACCENT)
            _txt(slide, str(i + 1), 0.52, y - 0.01, 0.3, 0.4,
                 bold=True, size=14, color="white", align="CENTER")
            _txt(slide, pt, 1.05, y - 0.04, 11.7, 0.55, size=16, color=_DARK)

    # ── 4-N. Topic slides
    for topic in a.get("topics", [])[:7]:
        slide = _blank()
        _bg(slide, _LIGHT)
        _rect(slide, 0, 0, 0.18, 7.5, _ACCENT)
        _txt(slide, topic.get("title", ""), 0.45, 0.3, 12.4, 0.9,
             bold=True, size=26, color=_DARK)
        _rect(slide, 0.45, 1.25, 12.4, 0.04, "#E2E8F0")
        desc = topic.get("description", "")
        _txt(slide, desc, 0.45, 1.45, 12.2, 1.2, size=17, color=_MID)
        for j, pt in enumerate(topic.get("points", [])[:5]):
            _rect(slide, 0.45, 2.85 + j * 0.72, 0.06, 0.32, _ACCENT)
            _txt(slide, pt, 0.7, 2.82 + j * 0.72, 12.1, 0.55, size=16, color=_DARK)

    # ── Entities slide
    ents = a.get("entities", {})
    if any(ents.values()):
        slide = _blank()
        _bg(slide, _DARK)
        _txt(slide, t["entities"], 0.6, 0.25, 12, 0.8,
             bold=True, size=28, color=_LIGHT)
        cols = [("People", ents.get("people", []), "#818CF8"),
                ("Organizations", ents.get("organizations", []), "#34D399"),
                ("Places", ents.get("places", []), "#FB923C"),
                ("Dates", ents.get("dates", []), "#F472B6")]
        for ci, (label, items, col) in enumerate(cols):
            x = 0.5 + ci * 3.2
            _rect(slide, x, 1.2, 2.9, 0.5, col)
            _txt(slide, label, x + 0.1, 1.25, 2.7, 0.4,
                 bold=True, size=14, color="white")
            body = "\n".join(f"• {it}" for it in items[:6]) or "—"
            _txt(slide, body, x + 0.1, 1.85, 2.8, 4.8, size=14, color=_MUTED)

    # ── Quotes slide
    quotes = a.get("notable_quotes", [])
    if quotes:
        slide = _blank()
        _bg(slide, "#1E293B")
        _txt(slide, t["quotes"], 0.6, 0.2, 12, 0.8,
             bold=True, size=28, color=_LIGHT)
        for qi, q in enumerate(quotes[:4]):
            y = 1.3 + qi * 1.42
            _rect(slide, 0.4, y, 0.06, 1.0, _ACCENT)
            _txt(slide, f'"{q}"', 0.7, y, 11.9, 1.1,
                 size=16, color="#CBD5E1")

    prs.save(str(out))


def gen_audio(source: dict, out: Path, lang: str = 'ru') -> None:
    a = source.get("analysis") or {}
    summary = a.get("summary", "No summary available.")
    kps = a.get("key_points", [])
    topics = a.get("topics", [])

    if lang == "ru":
        script = (
            f"Добро пожаловать в аудиообзор документа {source.get('name', 'этот документ')}. "
            f"{summary} "
        )
        if kps:
            script += "Ключевые выводы: " + ". ".join(kps[:5]) + ". "
        if topics:
            script += "Основные темы: " + ", ".join(
                t["title"] for t in topics[:4]
            ) + ". "
        script += "Этот обзор создан NoteLM."
    else:
        script = (
            f"Welcome to the audio overview of {source.get('name', 'this document')}. "
            f"{summary} "
        )
        if kps:
            script += "Key takeaways include: " + ". ".join(kps[:5]) + ". "
        if topics:
            script += "The main topics covered are: " + ", ".join(
                t["title"] for t in topics[:4]
            ) + ". "
        script += "This overview was generated by NoteLM."

    tts_lang = "ru" if lang == "ru" else "en"
    tts = gTTS(text=script, lang=tts_lang, slow=False)
    tts.save(str(out))


def gen_infographic(source: dict, out: Path) -> None:
    """Matplotlib infographic: summary stats + bar chart of topics + sentiment."""
    a = source.get("analysis") or {}
    topics = a.get("topics", [])
    kps    = a.get("key_points", [])
    ents   = a.get("entities", {})

    fig = plt.figure(figsize=(16, 10), facecolor=_DARK)
    fig.patch.set_facecolor(_DARK)

    # ── Grid layout: 2 rows, 3 cols
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)

    # ── Title bar
    fig.text(0.5, 0.95, source.get("name", "Document Overview"),
             ha="center", va="top", fontsize=20, fontweight="bold",
             color="white", fontfamily="DejaVu Sans")
    fig.text(0.5, 0.91, "AI-Generated Infographic · NoteLM",
             ha="center", va="top", fontsize=11, color=_MUTED)

    accent_colors = ["#6366F1", "#8B5CF6", "#EC4899", "#F59E0B",
                     "#10B981", "#3B82F6", "#EF4444"]

    # ── 1. Topic bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1E293B")
    if topics:
        labels = [t["title"][:22] for t in topics[:7]]
        # score = number of sub-points as proxy for depth
        vals   = [max(len(t.get("points", [])), 1) for t in topics[:7]]
        bars   = ax1.barh(labels[::-1], vals[::-1],
                          color=accent_colors[:len(labels)], height=0.55)
        for bar, val in zip(bars, vals[::-1]):
            ax1.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     str(val), va="center", color="white", fontsize=10)
        ax1.set_xlabel("Sub-points", color=_MUTED, fontsize=10)
        ax1.set_title("Topics by Depth", color="white", fontsize=13, pad=8)
    else:
        ax1.text(0.5, 0.5, "No topics extracted", ha="center",
                 color=_MUTED, transform=ax1.transAxes, fontsize=12)
        ax1.set_title("Topics", color="white", fontsize=13)
    ax1.tick_params(colors="white", labelsize=9)
    ax1.spines[:].set_color("#334155")
    ax1.set_xlim(0, max((max([len(t.get("points",[])) for t in topics[:7]], default=1) + 2), 5))

    # ── 2. Sentiment gauge (pie-like)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#1E293B")
    sentiment = a.get("sentiment", "neutral")
    sent_map = {
        "positive": ([0.8, 0.1, 0.1], ["#10B981", "#334155", "#EF4444"],
                     "Mostly Positive"),
        "negative": ([0.1, 0.1, 0.8], ["#10B981", "#334155", "#EF4444"],
                     "Mostly Negative"),
        "mixed":    ([0.4, 0.2, 0.4], ["#10B981", "#334155", "#EF4444"],
                     "Mixed"),
        "neutral":  ([0.1, 0.8, 0.1], ["#10B981", "#F59E0B", "#EF4444"],
                     "Neutral"),
    }
    sizes, colors, label = sent_map.get(sentiment, sent_map["neutral"])
    wedges, _ = ax2.pie(sizes, colors=colors, startangle=90,
                        wedgeprops=dict(width=0.5, edgecolor=_DARK))
    ax2.text(0, 0, label, ha="center", va="center",
             color="white", fontsize=9, fontweight="bold")
    ax2.set_title("Sentiment", color="white", fontsize=13, pad=8)

    # ── 3. Entity counts
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1E293B")
    ent_labels = ["People", "Orgs", "Places", "Dates"]
    ent_vals   = [
        len(ents.get("people", [])),
        len(ents.get("organizations", [])),
        len(ents.get("places", [])),
        len(ents.get("dates", [])),
    ]
    ent_cols = ["#818CF8", "#34D399", "#FB923C", "#F472B6"]
    ax3.bar(ent_labels, ent_vals, color=ent_cols, width=0.55)
    ax3.set_title("Entities Detected", color="white", fontsize=13, pad=8)
    ax3.tick_params(colors="white", labelsize=9)
    ax3.spines[:].set_color("#334155")
    for i, v in enumerate(ent_vals):
        ax3.text(i, v + 0.1, str(v), ha="center", color="white", fontsize=10)
    ax3.set_ylim(0, max(max(ent_vals) + 2, 5))

    # ── 4. Key points word-length chart
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1E293B")
    if kps:
        lens = [len(k.split()) for k in kps[:10]]
        ax4.barh(range(len(lens)), lens,
                 color=["#6366F1" if l > 10 else "#8B5CF6" for l in lens],
                 height=0.6)
        ax4.set_yticks(range(len(lens)))
        ax4.set_yticklabels([f"#{i+1}" for i in range(len(lens))],
                            color="white", fontsize=9)
        ax4.set_xlabel("Words", color=_MUTED, fontsize=9)
    ax4.set_title("Key Point Lengths", color="white", fontsize=13, pad=8)
    ax4.tick_params(axis="x", colors="white", labelsize=9)
    ax4.spines[:].set_color("#334155")

    # ── 5. Stats card
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1E293B")
    ax5.axis("off")
    stats = [
        ("Word Count",  f"{a.get('word_count', 0):,}"),
        ("Topics",      str(len(topics))),
        ("Key Points",  str(len(kps))),
        ("Complexity",  a.get("complexity", "—").title()),
        ("Language",    a.get("language", "—")),
    ]
    ax5.set_title("Document Stats", color="white", fontsize=13, pad=8)
    for si, (lbl, val) in enumerate(stats):
        y = 0.85 - si * 0.18
        ax5.text(0.05, y, lbl, transform=ax5.transAxes,
                 color=_MUTED, fontsize=11)
        ax5.text(0.95, y, val, transform=ax5.transAxes,
                 color="white", fontsize=12, fontweight="bold", ha="right")
        ax5.plot([0.05, 0.95], [y - 0.04, y - 0.04],
                 color="#334155", linewidth=0.5, transform=ax5.transAxes,
                 clip_on=False)

    plt.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=_DARK, edgecolor="none")
    plt.close(fig)


def gen_mindmap(source: dict, out: Path) -> None:
    """NetworkX + matplotlib mind map of topics and sub-points."""
    a = source.get("analysis") or {}
    topics = a.get("topics", [])
    name   = source.get("name", "Source")[:30]

    G = nx.Graph()
    center = name
    G.add_node(center, level=0)
    topic_nodes = []
    sub_nodes   = []

    for topic in topics[:8]:
        tname = topic["title"][:28]
        G.add_node(tname, level=1)
        G.add_edge(center, tname)
        topic_nodes.append(tname)
        for pt in topic.get("points", [])[:4]:
            pname = pt[:30]
            if pname in G.nodes:
                pname = pname + " "  # dedupe
            G.add_node(pname, level=2)
            G.add_edge(tname, pname)
            sub_nodes.append(pname)

    fig, ax = plt.subplots(figsize=(18, 12), facecolor=_DARK)
    ax.set_facecolor(_DARK)
    ax.axis("off")

    pos = nx.spring_layout(G, k=2.8, seed=42, iterations=80)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="#334155", width=1.5, alpha=0.7)
    # Center node
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[center],
                           node_color=_ACCENT, node_size=3200, alpha=0.95)
    # Topic nodes
    if topic_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=topic_nodes,
                               node_color="#4F46E5", node_size=1600, alpha=0.9)
    # Sub-point nodes
    if sub_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sub_nodes,
                               node_color="#1E293B", node_size=700, alpha=0.85)

    # Labels
    center_labels = {center: textwrap.fill(center, 14)}
    topic_labels  = {n: textwrap.fill(n, 14) for n in topic_nodes}
    sub_labels    = {n: textwrap.fill(n.strip(), 16) for n in sub_nodes}

    nx.draw_networkx_labels(G, pos, labels=center_labels, ax=ax,
                            font_size=11, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels=topic_labels, ax=ax,
                            font_size=9, font_color="#C7D2FE")
    nx.draw_networkx_labels(G, pos, labels=sub_labels, ax=ax,
                            font_size=7.5, font_color=_MUTED)

    fig.text(0.5, 0.97, f"Mind Map — {name}", ha="center",
             fontsize=16, fontweight="bold", color="white")
    fig.text(0.5, 0.935, "Generated by NoteLM", ha="center",
             fontsize=10, color=_MUTED)

    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=_DARK, edgecolor="none")
    plt.close(fig)


def gen_study_guide(source: dict, lang: str = 'ru') -> str:
    """Ask Gemini to produce a markdown study guide."""
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""Create a comprehensive study guide for the document "{source.get('name','')}".
IMPORTANT: Generate ALL content in {lang_label} language.

Document summary: {a.get('summary', '')}

Full content excerpt:
{content}

The study guide should include:
1. **Overview** — What this document is about
2. **Core Concepts** — Explained clearly
3. **Key Terms & Definitions** — Important vocabulary
4. **Concept Summary Table** — Two-column markdown table: Concept | Explanation
5. **Critical Questions** — 10 thought-provoking questions
6. **Review Questions** — 10 factual Q&A pairs
7. **Practice Activities** — 3 hands-on exercises

Format as clean markdown. Be thorough and educational."""
    return _gemini(prompt)


def gen_faq(source: dict, lang: str = 'ru') -> str:
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""Generate a comprehensive FAQ document for "{source.get('name','')}".
IMPORTANT: Generate ALL content in {lang_label} language.

Summary: {a.get('summary', '')}
Content: {content}

Create 15-20 natural FAQ pairs covering:
- Basic understanding questions
- Clarification questions  
- Application questions
- Edge case questions

Format as markdown with ## Q: and **A:** pairs."""
    return _gemini(prompt)


def gen_briefing(source: dict, lang: str = 'ru') -> str:
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""Write an executive briefing document for "{source.get('name','')}".
IMPORTANT: Generate ALL content in {lang_label} language.

Include:
# Executive Briefing: {source.get('name','')}

## Bottom Line Up Front (BLUF)
One-sentence critical takeaway.

## Situation
Current context and background (2-3 paragraphs).

## Key Findings
Numbered list of 5-8 most important findings.

## Implications
What this means and why it matters.

## Recommendations
3-5 actionable recommendations.

## Supporting Data
Key statistics, quotes, or evidence.

## Risk Factors
Potential concerns or caveats.

Content: {content}
Summary: {a.get('summary', '')}"""
    return _gemini(prompt)


def gen_timeline(source: dict, lang: str = 'ru') -> str:
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    dates = a.get("entities", {}).get("dates", [])
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""Extract and reconstruct a chronological timeline from "{source.get('name','')}".
IMPORTANT: Generate ALL content in {lang_label} language.

Known dates found: {', '.join(dates) if dates else 'scan content for any dates/periods'}.

Content: {content}

Format as a markdown timeline:
## Timeline

| Date/Period | Event/Development |
|------------|------------------|
| ...        | ...              |

Include ALL dates, periods, and sequential events you can identify.
After the table, add a **Narrative Summary** paragraph explaining the chronology."""
    return _gemini(prompt)


def gen_glossary(source: dict, lang: str = 'ru') -> str:
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""Extract all technical terms, jargon, and important concepts from "{source.get('name','')}".
IMPORTANT: Generate ALL content in {lang_label} language.

Content: {content}

Create a comprehensive glossary:
# Glossary — {source.get('name','')}

For each term provide:
**Term** — Clear, concise definition in plain language. Note context if relevant.

List terms alphabetically. Include 15-30 terms minimum."""
    return _gemini(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# Background task
# ──────────────────────────────────────────────────────────────────────────────

async def process_source(sid: str, file_path: Optional[Path], url: Optional[str]):
    sources_db[sid]["status"] = "processing"
    _save(sources_db)
    try:
        if url:
            content = await fetch_url(url)
        elif file_path:
            ext = file_path.suffix.lower()
            if ext == ".pdf":
                content = extract_pdf(file_path)
            elif ext in (".docx", ".doc"):
                content = extract_docx(file_path)
            else:
                content = extract_txt(file_path)
        else:
            content = ""

        sources_db[sid]["content_preview"] = content[:600]
        sources_db[sid]["content"] = content
        sources_db[sid]["analysis"] = analyze(content, sources_db[sid]["name"])
        sources_db[sid]["status"] = "ready"
    except Exception as e:
        sources_db[sid]["status"] = "error"
        sources_db[sid]["error"] = str(e)
    _save(sources_db)


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="NoteLM", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static",  StaticFiles(directory=str(STATIC)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")

# ── Source CRUD ───────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC / "index.html"))

@app.get("/api/sources")
async def list_sources():
    return [
        {k: v for k, v in s.items() if k != "content"}
        for s in sources_db.values()
    ]

@app.post("/api/sources/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
):
    sid = uuid.uuid4().hex[:10]
    try:
        if file and file.filename:
            safe = Path(file.filename).name
            if not safe or safe.startswith('.'):
                raise HTTPException(400, "Invalid filename")
            fp = UPLOADS / f"{sid}_{safe}"
            content = await file.read()
            if len(content) > 50_000_000:
                raise HTTPException(400, "File too large (max 50MB)")
            fp.write_bytes(content)
            sources_db[sid] = dict(id=sid, name=safe, type="file",
                                   file_path=str(fp), status="pending",
                                   created_at=datetime.now().isoformat())
            _save(sources_db)
            background_tasks.add_task(process_source, sid, fp, None)
            return {"id": sid, "status": "pending", "name": safe}
        elif url:
            if not url.startswith(('http://', 'https://')):
                raise HTTPException(400, "URL must start with http:// or https://")
            sources_db[sid] = dict(id=sid, name=url[:80], type="url",
                                   url=url, status="pending",
                                   created_at=datetime.now().isoformat())
            _save(sources_db)
            background_tasks.add_task(process_source, sid, None, url)
            return {"id": sid, "status": "pending", "name": url[:80]}
        else:
            raise HTTPException(400, "Provide a file or URL")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/api/sources/{sid}")
async def get_source(sid: str):
    s = sources_db.get(sid)
    if not s:
        raise HTTPException(404, "Not found")
    return {k: v for k, v in s.items() if k != "content"}

@app.delete("/api/sources/{sid}")
async def delete_source(sid: str):
    s = sources_db.pop(sid, None)
    if not s:
        raise HTTPException(404, "Not found")
    _save(sources_db)
    fp = s.get("file_path")
    if fp and Path(fp).exists():
        Path(fp).unlink(missing_ok=True)
    return {"deleted": sid}

# ── Generate endpoints ────────────────────────────────────────────────────────

def _require_ready(sid: str) -> dict:
    s = sources_db.get(sid)
    if not s:
        raise HTTPException(404, "Source not found")
    if s.get("status") != "ready":
        raise HTTPException(400, f"Source status is '{s.get('status')}', not ready")
    return s


def _merge_sources(sids: list[str]) -> dict:
    """Merge multiple ready sources into a single virtual source dict for generation."""
    sources_list = []
    for sid in sids:
        s = sources_db.get(sid)
        if s and s.get("status") == "ready":
            sources_list.append(s)
    if not sources_list:
        raise HTTPException(400, "No ready sources found for the given IDs")
    if len(sources_list) == 1:
        return sources_list[0]

    # Merge content and analysis from all sources
    combined_name = " + ".join(s.get("name", "Untitled") for s in sources_list[:3])
    if len(sources_list) > 3:
        combined_name += f" (+{len(sources_list)-3} more)"

    combined_content = "\n\n".join(
        f"=== {s.get('name', 'Source')} ===\n{s.get('content','')[:8000]}"
        for s in sources_list
    )

    # Merge analysis fields
    merged_analysis: dict = {
        "summary": " | ".join(
            filter(None, ((s.get("analysis", {}) or {}).get("summary", "") for s in sources_list))
        )[:2000],
        "key_points": [],
        "topics": [],
        "notable_quotes": [],
        "entities": {"people": [], "organizations": [], "places": [], "dates": []},
        "sentiment": "mixed",
        "complexity": "intermediate",
        "word_count": sum((s.get("analysis", {}) or {}).get("word_count", 0) for s in sources_list),
        "language": "English",
    }
    for s in sources_list:
        a = s.get("analysis", {}) or {}
        merged_analysis["key_points"].extend(a.get("key_points", []))
        merged_analysis["topics"].extend(a.get("topics", []))
        merged_analysis["notable_quotes"].extend(a.get("notable_quotes", []))
        for field in ("people", "organizations", "places", "dates"):
            merged_analysis["entities"][field].extend(
                (a.get("entities") or {}).get(field, [])
            )
    # Deduplicate
    for field in ("key_points", "notable_quotes"):
        seen: set = set()
        deduped = []
        for item in merged_analysis[field]:
            key = item[:60] if item else ""
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        merged_analysis[field] = deduped[:20]
    merged_analysis["topics"] = merged_analysis["topics"][:12]
    for field in ("people", "organizations", "places", "dates"):
        seen = set()
        deduped = []
        for item in merged_analysis["entities"][field]:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        merged_analysis["entities"][field] = deduped[:15]

    return {
        "id": "merged_" + "_".join(sids[:3]),
        "name": combined_name,
        "type": "merged",
        "status": "ready",
        "content": combined_content,
        "analysis": merged_analysis,
    }

@app.post("/api/sources/{sid}/generate/presentation")
async def gen_pptx_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        out = OUTPUTS / f"{sid}_presentation.pptx"
        await asyncio.get_event_loop().run_in_executor(None, gen_presentation, s, out, lang)
        url = f"/outputs/{sid}_presentation.pptx"
        sources_db[sid]["pptx_url"] = url
        _save(sources_db)
        return {"url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate presentation: {str(e)}")

@app.post("/api/sources/{sid}/generate/audio")
async def gen_audio_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        out = OUTPUTS / f"{sid}_audio.mp3"
        await asyncio.get_event_loop().run_in_executor(None, gen_audio, s, out, lang)
        url = f"/outputs/{sid}_audio.mp3"
        sources_db[sid]["audio_url"] = url
        _save(sources_db)
        return {"url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate audio: {str(e)}")

@app.post("/api/sources/{sid}/generate/infographic")
async def gen_infographic_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        out = OUTPUTS / f"{sid}_infographic.png"
        await asyncio.get_event_loop().run_in_executor(None, gen_infographic, s, out)
        url = f"/outputs/{sid}_infographic.png"
        sources_db[sid]["infographic_url"] = url
        _save(sources_db)
        return {"url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate infographic: {str(e)}")

@app.post("/api/sources/{sid}/generate/mindmap")
async def gen_mindmap_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        out = OUTPUTS / f"{sid}_mindmap.png"
        await asyncio.get_event_loop().run_in_executor(None, gen_mindmap, s, out)
        url = f"/outputs/{sid}_mindmap.png"
        sources_db[sid]["mindmap_url"] = url
        _save(sources_db)
        return {"url": url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate mindmap: {str(e)}")

@app.post("/api/sources/{sid}/generate/studyguide")
async def gen_study_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_study_guide, s, lang)
        sources_db[sid]["study_guide"] = text
        _save(sources_db)
        return {"content": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate study guide: {str(e)}")

@app.post("/api/sources/{sid}/generate/faq")
async def gen_faq_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_faq, s, lang)
        sources_db[sid]["faq"] = text
        _save(sources_db)
        return {"content": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate FAQ: {str(e)}")

@app.post("/api/sources/{sid}/generate/briefing")
async def gen_briefing_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_briefing, s, lang)
        sources_db[sid]["briefing"] = text
        _save(sources_db)
        return {"content": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate briefing: {str(e)}")

@app.post("/api/sources/{sid}/generate/timeline")
async def gen_timeline_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_timeline, s, lang)
        sources_db[sid]["timeline"] = text
        _save(sources_db)
        return {"content": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate timeline: {str(e)}")

@app.post("/api/sources/{sid}/generate/glossary")
async def gen_glossary_ep(sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready(sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_glossary, s, lang)
        sources_db[sid]["glossary"] = text
        _save(sources_db)
        return {"content": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate glossary: {str(e)}")

# ── Multi-source generate endpoint ───────────────────────────────────────────

@app.post("/api/generate/{gen_type}")
async def multi_generate_ep(gen_type: str, payload: dict):
    """Generate content from one or more sources identified by source_ids list."""
    try:
        source_ids = payload.get("source_ids", [])
        lang = payload.get("lang", "ru")
        if not source_ids:
            raise HTTPException(400, "No source_ids provided")

        # Merge sources (or use single source if only one)
        merged = _merge_sources(source_ids)
        key = "_".join(sorted(source_ids))[:40]

        if gen_type == "presentation":
            out = OUTPUTS / f"{key}_presentation.pptx"
            await asyncio.get_event_loop().run_in_executor(None, gen_presentation, merged, out, lang)
            url = f"/outputs/{key}_presentation.pptx"
            return {"url": url}

        elif gen_type == "audio":
            out = OUTPUTS / f"{key}_audio.mp3"
            await asyncio.get_event_loop().run_in_executor(None, gen_audio, merged, out, lang)
            url = f"/outputs/{key}_audio.mp3"
            return {"url": url}

        elif gen_type == "infographic":
            out = OUTPUTS / f"{key}_infographic.png"
            await asyncio.get_event_loop().run_in_executor(None, gen_infographic, merged, out)
            url = f"/outputs/{key}_infographic.png"
            return {"url": url}

        elif gen_type == "mindmap":
            out = OUTPUTS / f"{key}_mindmap.png"
            await asyncio.get_event_loop().run_in_executor(None, gen_mindmap, merged, out)
            url = f"/outputs/{key}_mindmap.png"
            return {"url": url}

        elif gen_type == "studyguide":
            text = await asyncio.get_event_loop().run_in_executor(None, gen_study_guide, merged, lang)
            return {"content": text}

        elif gen_type == "faq":
            text = await asyncio.get_event_loop().run_in_executor(None, gen_faq, merged, lang)
            return {"content": text}

        elif gen_type == "briefing":
            text = await asyncio.get_event_loop().run_in_executor(None, gen_briefing, merged, lang)
            return {"content": text}

        elif gen_type == "timeline":
            text = await asyncio.get_event_loop().run_in_executor(None, gen_timeline, merged, lang)
            return {"content": text}

        elif gen_type == "glossary":
            text = await asyncio.get_event_loop().run_in_executor(None, gen_glossary, merged, lang)
            return {"content": text}

        else:
            raise HTTPException(400, f"Unknown generation type: {gen_type}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to generate {gen_type}: {str(e)}")


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(payload: dict):
    try:
        question    = payload.get("question", "").strip()
        source_ids  = payload.get("source_ids", [])
        chat_history = payload.get("history", [])

        if not question:
            raise HTTPException(400, "No question provided")

        context_parts = []
        for sid in source_ids:
            s = sources_db.get(sid)
            if not s:
                continue
            if s.get("status") != "ready":
                continue
            content = s.get("content", "")
            a = s.get("analysis", {}) or {}
            part = f"=== SOURCE: {s['name']} ===\n"
            if content:
                part += content[:10000]
            else:
                part += f"Summary: {a.get('summary','')}\n"
                part += "Key points: " + "; ".join(a.get("key_points", [])) + "\n"
            context_parts.append(part)

        history_text = ""
        for msg in chat_history[-6:]:
            role = msg.get("role", "user")
            history_text += f"\n{role.upper()}: {msg.get('content','')}"

        if not context_parts:
            system = "You are NoteLM, a helpful AI research assistant. Answer based on general knowledge if no sources are provided."
            prompt = f"{system}\n{history_text}\nUSER: {question}\nASSISTANT:"
        else:
            context = "\n\n".join(context_parts)
            prompt = f"""You are NoteLM, an expert research assistant. Answer questions based on the provided sources.
Be precise, cite sources by name, and be thorough. If the answer is not in the sources, say so.

SOURCES:
{context[:28000]}

CONVERSATION HISTORY:{history_text}

USER: {question}
ASSISTANT:"""

        resp = await asyncio.get_event_loop().run_in_executor(None, _gemini, prompt)
        return {"answer": resp}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "sources": len(sources_db)}

@app.on_event("startup")
async def startup_event():
    for sid, s in sources_db.items():
        if s.get("status") == "pending":
            s["status"] = "processing"
            _save(sources_db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)
