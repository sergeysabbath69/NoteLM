"""
NoteLM — NotebookLM-grade research assistant
FastAPI + Gemini backend  
Multi-notebook version with sharing
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from pptx.util import Inches, Pt
from google import genai

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent
UPLOADS   = BASE_DIR / "uploads"
OUTPUTS   = BASE_DIR / "outputs"
STATIC    = BASE_DIR / "static"
DATA_DIR  = BASE_DIR / "data"

for d in (UPLOADS, OUTPUTS, STATIC, DATA_DIR):
    d.mkdir(exist_ok=True)

NOTEBOOKS_FILE   = DATA_DIR / "notebooks.json"
SOURCES_FILE     = DATA_DIR / "sources.json"
OLD_SOURCES_FILE = BASE_DIR / "sources.json"  # legacy migration

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyBvnlnYaex2EecyzuJ9N3t1yInFII4U_zw"
)
_client = genai.Client(api_key=GEMINI_API_KEY)

# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path, default=None):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default if default is not None else {}

def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))

notebooks_db: dict = {}
sources_db: dict   = {}

def _load_dbs():
    global notebooks_db, sources_db
    notebooks_db = _load_json(NOTEBOOKS_FILE, {})
    sources_db   = _load_json(SOURCES_FILE, {})

def _save_notebooks():
    _save_json(NOTEBOOKS_FILE, notebooks_db)

def _save_sources():
    _save_json(SOURCES_FILE, sources_db)

def _save_all():
    _save_notebooks()
    _save_sources()

# ── Migration from old flat sources.json ──────────────────────────────────────
def _migrate():
    if notebooks_db:
        return
    old = _load_json(OLD_SOURCES_FILE, {})
    if not old:
        return
    nid = "nb_" + uuid.uuid4().hex[:10]
    notebooks_db[nid] = {
        "id": nid,
        "name": "My Research",
        "description": "Migrated from previous version",
        "created_at": datetime.now().isoformat(),
        "sources": list(old.keys()),
        "share_token": None,
        "shared_at": None,
    }
    sources_db.update(old)
    _save_all()

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
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw.strip())

# ──────────────────────────────────────────────────────────────────────────────
# Analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyze(content: str, name: str, lang: str = 'ru') -> dict:
    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    lang_hint = "Russian" if lang == "ru" else "English"

    prompt = f"""{lang_instruction}

You are a research analyst. Analyze the document titled "{name}".
Provide detailed, substantive content.
Minimum 5 key points. Minimum 3 topics with 3+ sub-points each.
Return ONLY a valid JSON object (no markdown fences) with exactly these fields:

{{
  "summary": "Comprehensive 3-4 paragraph executive summary",
  "key_points": ["concise key point 1", "key point 2", "..."],
  "topics": [
    {{
      "title": "Topic Name",
      "description": "2-3 sentence description",
      "points": ["sub-point 1", "sub-point 2", "sub-point 3"]
    }}
  ],
  "notable_quotes": ["exact quote 1", "..."],
  "entities": {{
    "people": ["name1", "..."],
    "organizations": ["org1", "..."],
    "places": ["place1", "..."],
    "dates": ["date1", "..."]
  }},
  "sentiment": "positive|negative|neutral|mixed",
  "complexity": "beginner|intermediate|advanced",
  "word_count": 1234,
  "language": "{lang_hint}"
}}

Document content (first 28000 chars):
{content[:28000]}"""

    try:
        return _gemini_json(prompt)
    except Exception as e:
        return {
            "summary": "Analysis failed. Content preview: " + content[:300],
            "key_points": [],
            "topics": [],
            "notable_quotes": [],
            "entities": {"people": [], "organizations": [], "places": [], "dates": []},
            "sentiment": "neutral",
            "complexity": "intermediate",
            "word_count": len(content.split()),
            "language": lang_hint,
            "_error": str(e),
        }

# ──────────────────────────────────────────────────────────────────────────────
# Output generators
# ──────────────────────────────────────────────────────────────────────────────

_DARK   = "#0F172A"
_MID    = "#334155"
_ACCENT = "#6366F1"
_LIGHT  = "#F8FAFC"
_MUTED  = "#94A3B8"


def gen_presentation(source: dict, out: Path, lang: str = 'ru') -> None:
    a = source.get("analysis") or {}
    prs = PptxPresentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    titles = {
        "ru": {"overview": "AI-Generated Обзор исследования", "summary": "Резюме",
               "key_points": "Ключевые моменты", "entities": "Ключевые сущности",
               "quotes": "Примечательные цитаты"},
        "en": {"overview": "AI-Generated Research Overview", "summary": "Summary",
               "key_points": "Key Points", "entities": "Key Entities",
               "quotes": "Notable Quotes"},
    }
    t = titles.get(lang, titles["ru"])

    def _blank():
        return prs.slides.add_slide(prs.slide_layouts[6])

    def _parse_color(color: str):
        named = {"white": "#FFFFFF", "black": "#000000"}
        c = named.get(str(color).lower(), color).lstrip('#')
        if len(c) == 3:
            c = c[0]*2 + c[1]*2 + c[2]*2
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    def _bg(slide, color: str):
        bg = slide.background; fill = bg.fill; fill.solid()
        r, g, b = _parse_color(color)
        fill.fore_color.rgb = RGBColor(r, g, b)

    def _txt(slide, text, left, top, width, height,
             bold=False, size=18, color="#0F172A", align="LEFT"):
        from pptx.enum.text import PP_ALIGN
        box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = text
        p.font.bold = bold; p.font.size = Pt(size)
        r, g, b = _parse_color(color)
        p.font.color.rgb = RGBColor(r, g, b)
        p.alignment = {"LEFT": PP_ALIGN.LEFT, "CENTER": PP_ALIGN.CENTER, "RIGHT": PP_ALIGN.RIGHT}.get(align, PP_ALIGN.LEFT)
        return box

    def _rect(slide, left, top, width, height, color):
        r, g, b = _parse_color(color)
        shape = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor(r, g, b)
        shape.line.fill.background()
        return shape

    slide = _blank(); _bg(slide, _DARK)
    _rect(slide, 0, 3.2, 13.33, 0.06, _ACCENT)
    _txt(slide, source.get("name", "Untitled"), 0.8, 1.2, 11.7, 2.0, bold=True, size=40, color=_LIGHT)
    _txt(slide, t["overview"], 0.8, 3.5, 10, 0.6, size=20, color=_MUTED)
    _txt(slide, datetime.now().strftime("%B %d, %Y"), 0.8, 4.3, 6, 0.5, size=16, color=_MUTED)

    slide = _blank(); _bg(slide, _LIGHT)
    _rect(slide, 0, 0, 13.33, 1.1, _ACCENT)
    _txt(slide, t["summary"], 0.6, 0.2, 12, 0.7, bold=True, size=28, color="white")
    _txt(slide, textwrap.fill(a.get("summary", ""), 110)[:900], 0.6, 1.3, 12.1, 5.8, size=17, color=_DARK)

    kps = a.get("key_points", [])
    if kps:
        slide = _blank(); _bg(slide, _LIGHT)
        _rect(slide, 0, 0, 13.33, 1.1, "#4F46E5")
        _txt(slide, t["key_points"], 0.6, 0.2, 12, 0.7, bold=True, size=28, color="white")
        for i, pt in enumerate(kps[:7]):
            y = 1.35 + i * 0.78
            _rect(slide, 0.5, y, 0.35, 0.35, _ACCENT)
            _txt(slide, str(i+1), 0.52, y-0.01, 0.3, 0.4, bold=True, size=14, color="white", align="CENTER")
            _txt(slide, pt, 1.05, y-0.04, 11.7, 0.55, size=16, color=_DARK)

    for topic in a.get("topics", [])[:7]:
        slide = _blank(); _bg(slide, _LIGHT)
        _rect(slide, 0, 0, 0.18, 7.5, _ACCENT)
        _txt(slide, topic.get("title", ""), 0.45, 0.3, 12.4, 0.9, bold=True, size=26, color=_DARK)
        _rect(slide, 0.45, 1.25, 12.4, 0.04, "#E2E8F0")
        _txt(slide, topic.get("description", ""), 0.45, 1.45, 12.2, 1.2, size=17, color=_MID)
        for j, pt in enumerate(topic.get("points", [])[:5]):
            _rect(slide, 0.45, 2.85+j*0.72, 0.06, 0.32, _ACCENT)
            _txt(slide, pt, 0.7, 2.82+j*0.72, 12.1, 0.55, size=16, color=_DARK)

    ents = a.get("entities", {})
    if any(ents.values()):
        slide = _blank(); _bg(slide, _DARK)
        _txt(slide, t["entities"], 0.6, 0.25, 12, 0.8, bold=True, size=28, color=_LIGHT)
        for ci, (label, items, col) in enumerate([
            ("People", ents.get("people",[]), "#818CF8"),
            ("Organizations", ents.get("organizations",[]), "#34D399"),
            ("Places", ents.get("places",[]), "#FB923C"),
            ("Dates", ents.get("dates",[]), "#F472B6"),
        ]):
            x = 0.5 + ci * 3.2
            _rect(slide, x, 1.2, 2.9, 0.5, col)
            _txt(slide, label, x+0.1, 1.25, 2.7, 0.4, bold=True, size=14, color="white")
            _txt(slide, "\n".join(f"• {it}" for it in items[:6]) or "—",
                 x+0.1, 1.85, 2.8, 4.8, size=14, color=_MUTED)

    quotes = a.get("notable_quotes", [])
    if quotes:
        slide = _blank(); _bg(slide, "#1E293B")
        _txt(slide, t["quotes"], 0.6, 0.2, 12, 0.8, bold=True, size=28, color=_LIGHT)
        for qi, q in enumerate(quotes[:4]):
            y = 1.3 + qi * 1.42
            _rect(slide, 0.4, y, 0.06, 1.0, _ACCENT)
            _txt(slide, f'"{q}"', 0.7, y, 11.9, 1.1, size=16, color="#CBD5E1")

    prs.save(str(out))


def gen_audio(source: dict, out: Path, lang: str = 'ru') -> None:
    """Generate a natural, conversational audio overview using Gemini for the script."""
    a = source.get("analysis") or {}
    content = source.get("content", "")[:15000]
    name = source.get("name", "this document")

    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    lang_label = "Russian" if lang == "ru" else "English"

    prompt = f"""{lang_instruction}

You are a professional podcast host. Write a natural, conversational audio script
reviewing the document "{name}". The script should be 3-5 minutes long when read aloud
(approximately 500-750 words). Do NOT include any stage directions, sound effects,
or formatting marks. Write only the spoken words.

Start with a friendly introduction, cover the main topics and key insights,
include interesting details and context, and end with a thoughtful conclusion.
Use natural speech patterns, transitions like "Now let's talk about...", "What's interesting here is...",
"One key point to understand is...". Make it engaging and informative.

Generate ONLY the script text in {lang_label}, nothing else.

Document summary: {a.get('summary', '')}
Key points: {', '.join(a.get('key_points', [])[:8])}
Main topics: {', '.join(t.get('title','') for t in a.get('topics',[])[:5])}
Content excerpt: {content[:5000]}"""

    try:
        script = _gemini(prompt)
    except Exception:
        # Fallback script
        kps = a.get("key_points", [])
        topics = a.get("topics", [])
        if lang == "ru":
            script = (
                f"Добро пожаловать в аудиообзор документа «{name}». "
                f"{a.get('summary', 'Этот документ содержит важную информацию.')} "
            )
            if kps:
                script += "Среди ключевых выводов стоит отметить: " + ". ".join(kps[:6]) + ". "
            if topics:
                script += "Документ охватывает следующие темы: " + ", ".join(
                    t["title"] for t in topics[:5]) + ". "
            script += "Надеемся, этот обзор был полезен. Спасибо за внимание."
        else:
            script = (
                f"Welcome to the audio overview of '{name}'. "
                f"{a.get('summary', 'This document contains important information.')} "
            )
            if kps:
                script += "Key takeaways include: " + ". ".join(kps[:6]) + ". "
            if topics:
                script += "The main topics covered are: " + ", ".join(
                    t["title"] for t in topics[:5]) + ". "
            script += "We hope this overview was helpful. Thank you for listening."

    tts_lang = "ru" if lang == "ru" else "en"
    tts = gTTS(text=script, lang=tts_lang, slow=False)
    tts.save(str(out))


def gen_infographic(source: dict, out: Path) -> None:
    a = source.get("analysis") or {}
    topics = a.get("topics", [])
    kps    = a.get("key_points", [])
    ents   = a.get("entities", {})

    fig = plt.figure(figsize=(16, 10), facecolor=_DARK)
    fig.patch.set_facecolor(_DARK)
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)
    fig.text(0.5, 0.95, source.get("name", "Document Overview"),
             ha="center", va="top", fontsize=20, fontweight="bold", color="white")
    fig.text(0.5, 0.91, "AI-Generated Infographic · NoteLM",
             ha="center", va="top", fontsize=11, color=_MUTED)

    accent_colors = ["#6366F1","#8B5CF6","#EC4899","#F59E0B","#10B981","#3B82F6","#EF4444"]

    ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor("#1E293B")
    if topics:
        labels = [t["title"][:22] for t in topics[:7]]
        vals   = [max(len(t.get("points",[])), 1) for t in topics[:7]]
        bars   = ax1.barh(labels[::-1], vals[::-1], color=accent_colors[:len(labels)], height=0.55)
        for bar, val in zip(bars, vals[::-1]):
            ax1.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                     str(val), va="center", color="white", fontsize=10)
        ax1.set_xlabel("Sub-points", color=_MUTED, fontsize=10)
        ax1.set_title("Topics by Depth", color="white", fontsize=13, pad=8)
        ax1.set_xlim(0, max(max(vals)+2, 5))
    else:
        ax1.text(0.5, 0.5, "No topics extracted", ha="center", color=_MUTED,
                 transform=ax1.transAxes, fontsize=12)
        ax1.set_title("Topics", color="white", fontsize=13)
    ax1.tick_params(colors="white", labelsize=9)
    ax1.spines[:].set_color("#334155")

    ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor("#1E293B")
    sentiment = a.get("sentiment", "neutral")
    sent_map = {
        "positive": ([0.8,0.1,0.1], ["#10B981","#334155","#EF4444"], "Mostly Positive"),
        "negative": ([0.1,0.1,0.8], ["#10B981","#334155","#EF4444"], "Mostly Negative"),
        "mixed":    ([0.4,0.2,0.4], ["#10B981","#334155","#EF4444"], "Mixed"),
        "neutral":  ([0.1,0.8,0.1], ["#10B981","#F59E0B","#EF4444"], "Neutral"),
    }
    sizes, colors, label = sent_map.get(sentiment, sent_map["neutral"])
    ax2.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.5, edgecolor=_DARK))
    ax2.text(0, 0, label, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax2.set_title("Sentiment", color="white", fontsize=13, pad=8)

    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor("#1E293B")
    ent_labels = ["People","Orgs","Places","Dates"]
    ent_vals   = [len(ents.get("people",[])), len(ents.get("organizations",[])),
                  len(ents.get("places",[])), len(ents.get("dates",[]))]
    ent_cols = ["#818CF8","#34D399","#FB923C","#F472B6"]
    ax3.bar(ent_labels, ent_vals, color=ent_cols, width=0.55)
    ax3.set_title("Entities Detected", color="white", fontsize=13, pad=8)
    ax3.tick_params(colors="white", labelsize=9); ax3.spines[:].set_color("#334155")
    for i, v in enumerate(ent_vals):
        ax3.text(i, v+0.1, str(v), ha="center", color="white", fontsize=10)
    ax3.set_ylim(0, max(max(ent_vals)+2, 5))

    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor("#1E293B")
    if kps:
        lens = [len(k.split()) for k in kps[:10]]
        ax4.barh(range(len(lens)), lens,
                 color=["#6366F1" if l > 10 else "#8B5CF6" for l in lens], height=0.6)
        ax4.set_yticks(range(len(lens)))
        ax4.set_yticklabels([f"#{i+1}" for i in range(len(lens))], color="white", fontsize=9)
        ax4.set_xlabel("Words", color=_MUTED, fontsize=9)
    ax4.set_title("Key Point Lengths", color="white", fontsize=13, pad=8)
    ax4.tick_params(axis="x", colors="white", labelsize=9)
    ax4.spines[:].set_color("#334155")

    ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor("#1E293B"); ax5.axis("off")
    stats = [("Word Count", f"{a.get('word_count',0):,}"), ("Topics", str(len(topics))),
             ("Key Points", str(len(kps))), ("Complexity", a.get("complexity","—").title()),
             ("Language", a.get("language","—"))]
    ax5.set_title("Document Stats", color="white", fontsize=13, pad=8)
    for si, (lbl, val) in enumerate(stats):
        y = 0.85 - si * 0.18
        ax5.text(0.05, y, lbl, transform=ax5.transAxes, color=_MUTED, fontsize=11)
        ax5.text(0.95, y, val, transform=ax5.transAxes, color="white", fontsize=12,
                 fontweight="bold", ha="right")
        ax5.plot([0.05,0.95], [y-0.04,y-0.04], color="#334155", linewidth=0.5,
                 transform=ax5.transAxes, clip_on=False)

    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=_DARK, edgecolor="none")
    plt.close(fig)


def gen_mindmap(source: dict, out: Path) -> None:
    a = source.get("analysis") or {}
    topics = a.get("topics", [])
    name   = source.get("name", "Source")[:30]

    G = nx.Graph()
    center = name; G.add_node(center, level=0)
    topic_nodes = []; sub_nodes = []

    for topic in topics[:8]:
        tname = topic["title"][:28]; G.add_node(tname, level=1)
        G.add_edge(center, tname); topic_nodes.append(tname)
        for pt in topic.get("points", [])[:4]:
            pname = pt[:30]
            if pname in G.nodes: pname = pname + " "
            G.add_node(pname, level=2); G.add_edge(tname, pname); sub_nodes.append(pname)

    fig, ax = plt.subplots(figsize=(18, 12), facecolor=_DARK)
    ax.set_facecolor(_DARK); ax.axis("off")
    pos = nx.spring_layout(G, k=2.8, seed=42, iterations=80)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#334155", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[center],
                           node_color=_ACCENT, node_size=3200, alpha=0.95)
    if topic_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=topic_nodes,
                               node_color="#4F46E5", node_size=1600, alpha=0.9)
    if sub_nodes:
        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=sub_nodes,
                               node_color="#1E293B", node_size=700, alpha=0.85)
    nx.draw_networkx_labels(G, pos, labels={center: textwrap.fill(center, 14)}, ax=ax,
                            font_size=11, font_color="white", font_weight="bold")
    nx.draw_networkx_labels(G, pos, labels={n: textwrap.fill(n, 14) for n in topic_nodes},
                            ax=ax, font_size=9, font_color="#C7D2FE")
    nx.draw_networkx_labels(G, pos, labels={n: textwrap.fill(n.strip(), 16) for n in sub_nodes},
                            ax=ax, font_size=7.5, font_color=_MUTED)
    fig.text(0.5, 0.97, f"Mind Map — {name}", ha="center",
             fontsize=16, fontweight="bold", color="white")
    fig.text(0.5, 0.935, "Generated by NoteLM", ha="center", fontsize=10, color=_MUTED)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=_DARK, edgecolor="none")
    plt.close(fig)


def gen_study_guide(source: dict, lang: str = 'ru') -> str:
    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""{lang_instruction}

Create a COMPREHENSIVE study guide for the document "{source.get('name','')}".
Generate ALL content in {lang_label} language.
Include 10+ review questions with detailed answers. Be thorough and educational.

Document summary: {a.get('summary', '')}

Full content excerpt:
{content}

The study guide MUST include all these sections:
1. **Overview** — What this document is about (2-3 paragraphs)
2. **Core Concepts** — All major concepts explained clearly with examples (at least 5)
3. **Key Terms & Definitions** — Important vocabulary (15+ terms, each with full definition)
4. **Concept Summary Table** — Two-column markdown table: Concept | Explanation (10+ rows)
5. **Critical Questions** — 10 thought-provoking questions with detailed answers (3-5 sentences each)
6. **Review Questions** — 10 factual Q&A pairs with thorough answers
7. **Practice Activities** — 3 hands-on exercises with step-by-step instructions

Format as clean markdown. Be thorough, detailed, and educational."""
    return _gemini(prompt)


def gen_faq(source: dict, lang: str = 'ru') -> str:
    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""{lang_instruction}

Generate 15-20 FAQ pairs for "{source.get('name','')}".
Generate ALL content in {lang_label} language.
Each answer MUST be 2-4 sentences minimum — substantive and informative.

Summary: {a.get('summary', '')}
Content: {content}

Create 15-20 natural FAQ pairs covering:
- Basic understanding questions (what, who, when, where)
- Clarification questions (how, why)
- Application questions (practical use cases)
- Edge case questions (limitations, exceptions)

Format as markdown:
## Q: [Question]
**A:** [Detailed answer, 2-4 sentences minimum]

Generate all 15-20 pairs. Do not skip any."""
    return _gemini(prompt)


def gen_briefing(source: dict, lang: str = 'ru') -> str:
    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""{lang_instruction}

Write a thorough executive briefing for "{source.get('name','')}".
Generate ALL content in {lang_label} language.
The briefing MUST be at least 500 words total. Be substantive and detailed.

# Executive Briefing: {source.get('name','')}

## Bottom Line Up Front (BLUF)
2-3 sentence critical takeaway.

## Situation
Current context and background (3-4 paragraphs, thorough).

## Key Findings
Numbered list of 7-10 most important findings, each with explanation (2-3 sentences per finding).

## Implications
What this means and why it matters (2-3 paragraphs with specific details).

## Recommendations
5-7 actionable recommendations, each with reasoning and expected outcome.

## Supporting Data
Key statistics, quotes, or evidence with context and interpretation.

## Risk Factors
4-6 potential concerns or caveats, each explained clearly.

Content: {content}
Summary: {a.get('summary', '')}"""
    return _gemini(prompt)


def gen_timeline(source: dict, lang: str = 'ru') -> str:
    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    a = source.get("analysis") or {}
    content = source.get("content", "")[:20000]
    dates = a.get("entities", {}).get("dates", [])
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""{lang_instruction}

Extract and reconstruct a chronological timeline from "{source.get('name','')}".
Generate ALL content in {lang_label} language.

Known dates found: {', '.join(dates) if dates else 'scan content for any dates/periods'}.

Content: {content}

Format as a markdown timeline:
## Timeline

| Date/Period | Event/Development |
|------------|------------------|
| ...        | ...              |

Include ALL dates, periods, and sequential events you can identify.
After the table, add a **Narrative Summary** paragraph explaining the chronology in detail."""
    return _gemini(prompt)


def gen_glossary(source: dict, lang: str = 'ru') -> str:
    lang_instruction = (
        "RESPOND ENTIRELY IN RUSSIAN LANGUAGE. НЕ ИСПОЛЬЗУЙ АНГЛИЙСКИЙ ЯЗЫК."
        if lang == "ru" else "RESPOND ENTIRELY IN ENGLISH LANGUAGE."
    )
    content = source.get("content", "")[:20000]
    lang_label = "Russian" if lang == "ru" else "English"
    prompt = f"""{lang_instruction}

Extract all technical terms, jargon, and important concepts from "{source.get('name','')}".
Generate ALL content in {lang_label} language. Include 15-30 terms minimum.

Content: {content}

Create a comprehensive glossary:
# Glossary — {source.get('name','')}

For each term provide:
**Term** — Clear, concise definition in plain language. Note context and importance if relevant.

List terms alphabetically. Include at minimum 15-30 terms."""
    return _gemini(prompt)


# ──────────────────────────────────────────────────────────────────────────────
# Merge sources helper
# ──────────────────────────────────────────────────────────────────────────────

def _merge_sources(sids: list[str]) -> dict:
    sources_list = [s for sid in sids if (s := sources_db.get(sid)) and s.get("status") == "ready"]
    if not sources_list:
        raise HTTPException(400, "No ready sources found")
    if len(sources_list) == 1:
        return sources_list[0]

    combined_name = " + ".join(s.get("name","Untitled") for s in sources_list[:3])
    if len(sources_list) > 3:
        combined_name += f" (+{len(sources_list)-3} more)"

    combined_content = "\n\n".join(
        f"=== {s.get('name','Source')} ===\n{s.get('content','')[:8000]}"
        for s in sources_list
    )
    merged_analysis: dict = {
        "summary": " | ".join(filter(None, ((s.get("analysis",{}) or {}).get("summary","") for s in sources_list)))[:2000],
        "key_points": [], "topics": [], "notable_quotes": [],
        "entities": {"people":[],"organizations":[],"places":[],"dates":[]},
        "sentiment": "mixed", "complexity": "intermediate",
        "word_count": sum((s.get("analysis",{}) or {}).get("word_count",0) for s in sources_list),
        "language": "English",
    }
    for s in sources_list:
        a = s.get("analysis",{}) or {}
        merged_analysis["key_points"].extend(a.get("key_points",[]))
        merged_analysis["topics"].extend(a.get("topics",[]))
        merged_analysis["notable_quotes"].extend(a.get("notable_quotes",[]))
        for f in ("people","organizations","places","dates"):
            merged_analysis["entities"][f].extend((a.get("entities") or {}).get(f,[]))

    for f in ("key_points","notable_quotes"):
        seen: set = set(); deduped = []
        for item in merged_analysis[f]:
            k = item[:60] if item else ""
            if k not in seen: seen.add(k); deduped.append(item)
        merged_analysis[f] = deduped[:20]
    merged_analysis["topics"] = merged_analysis["topics"][:12]
    for f in ("people","organizations","places","dates"):
        seen = set(); deduped = []
        for item in merged_analysis["entities"][f]:
            if item not in seen: seen.add(item); deduped.append(item)
        merged_analysis["entities"][f] = deduped[:15]

    return {"id":"merged_"+"_".join(sids[:3]),"name":combined_name,"type":"merged",
            "status":"ready","content":combined_content,"analysis":merged_analysis}


# ──────────────────────────────────────────────────────────────────────────────
# Background processing
# ──────────────────────────────────────────────────────────────────────────────

async def process_source(sid: str, file_path: Optional[Path], url: Optional[str], lang: str = 'ru'):
    sources_db[sid]["status"] = "processing"
    _save_sources()
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
        sources_db[sid]["analysis"] = analyze(content, sources_db[sid]["name"], lang)
        sources_db[sid]["status"] = "ready"
    except Exception as e:
        sources_db[sid]["status"] = "error"
        sources_db[sid]["error"] = str(e)
    _save_sources()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="NoteLM", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.mount("/static",  StaticFiles(directory=str(STATIC)),  name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS)), name="outputs")


# ──────────────────────────────────────────────────────────────────────────────
# Static routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC / "index.html"))

@app.get("/share/{token}")
async def share_page(token: str):
    """Serve SPA for public share view."""
    return FileResponse(str(STATIC / "index.html"))


# ──────────────────────────────────────────────────────────────────────────────
# Notebook CRUD
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/notebooks")
async def list_notebooks():
    result = []
    for nb in notebooks_db.values():
        src_count = len(nb.get("sources", []))
        result.append({**{k: v for k, v in nb.items() if k != "sources"},
                       "source_count": src_count, "sources": nb.get("sources", [])})
    # Sort by created_at desc
    result.sort(key=lambda x: x.get("created_at",""), reverse=True)
    return result

@app.post("/api/notebooks")
async def create_notebook(payload: dict):
    nid = "nb_" + uuid.uuid4().hex[:10]
    name = payload.get("name", "").strip() or "New Notebook"
    description = payload.get("description", "").strip()
    nb = {
        "id": nid,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "sources": [],
        "share_token": None,
        "shared_at": None,
    }
    notebooks_db[nid] = nb
    _save_notebooks()
    return {**nb, "source_count": 0}

@app.get("/api/notebooks/{nid}")
async def get_notebook(nid: str):
    nb = notebooks_db.get(nid)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    return {**nb, "source_count": len(nb.get("sources", []))}

@app.put("/api/notebooks/{nid}")
async def update_notebook(nid: str, payload: dict):
    nb = notebooks_db.get(nid)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    if "name" in payload:
        nb["name"] = payload["name"].strip() or nb["name"]
    if "description" in payload:
        nb["description"] = payload["description"].strip()
    _save_notebooks()
    return {**nb, "source_count": len(nb.get("sources", []))}

@app.delete("/api/notebooks/{nid}")
async def delete_notebook(nid: str):
    nb = notebooks_db.pop(nid, None)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    # Delete all sources in notebook
    for sid in nb.get("sources", []):
        s = sources_db.pop(sid, None)
        if s:
            fp = s.get("file_path")
            if fp and Path(fp).exists():
                Path(fp).unlink(missing_ok=True)
    _save_all()
    return {"deleted": nid}


# ──────────────────────────────────────────────────────────────────────────────
# Source CRUD (notebook-scoped)
# ──────────────────────────────────────────────────────────────────────────────

def _get_notebook(nid: str) -> dict:
    nb = notebooks_db.get(nid)
    if not nb:
        raise HTTPException(404, "Notebook not found")
    return nb

@app.get("/api/notebooks/{nid}/sources")
async def list_notebook_sources(nid: str):
    nb = _get_notebook(nid)
    result = []
    for sid in nb.get("sources", []):
        s = sources_db.get(sid)
        if s:
            result.append({k: v for k, v in s.items() if k != "content"})
    return result

@app.post("/api/notebooks/{nid}/sources/upload")
async def upload_to_notebook(
    nid: str,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    lang: str = Form('ru'),
):
    nb = _get_notebook(nid)
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
                                   notebook_id=nid,
                                   created_at=datetime.now().isoformat())
            nb["sources"].append(sid)
            _save_all()
            background_tasks.add_task(process_source, sid, fp, None, lang)
            return {"id": sid, "status": "pending", "name": safe}
        elif url:
            if not url.startswith(('http://','https://')):
                raise HTTPException(400, "URL must start with http:// or https://")
            sources_db[sid] = dict(id=sid, name=url[:80], type="url",
                                   url=url, status="pending",
                                   notebook_id=nid,
                                   created_at=datetime.now().isoformat())
            nb["sources"].append(sid)
            _save_all()
            background_tasks.add_task(process_source, sid, None, url, lang)
            return {"id": sid, "status": "pending", "name": url[:80]}
        else:
            raise HTTPException(400, "Provide a file or URL")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

@app.get("/api/notebooks/{nid}/sources/{sid}")
async def get_notebook_source(nid: str, sid: str):
    nb = _get_notebook(nid)
    if sid not in nb.get("sources", []):
        raise HTTPException(404, "Source not in notebook")
    s = sources_db.get(sid)
    if not s:
        raise HTTPException(404, "Source not found")
    return {k: v for k, v in s.items() if k != "content"}

@app.delete("/api/notebooks/{nid}/sources/{sid}")
async def delete_notebook_source(nid: str, sid: str):
    nb = _get_notebook(nid)
    if sid not in nb.get("sources", []):
        raise HTTPException(404, "Source not in notebook")
    nb["sources"].remove(sid)
    s = sources_db.pop(sid, None)
    _save_all()
    if s:
        fp = s.get("file_path")
        if fp and Path(fp).exists():
            Path(fp).unlink(missing_ok=True)
    return {"deleted": sid}


# ──────────────────────────────────────────────────────────────────────────────
# Share endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/notebooks/{nid}/share")
async def share_notebook(nid: str, request=None):
    nb = _get_notebook(nid)
    if not nb.get("share_token"):
        token = "tok_" + secrets.token_urlsafe(16)
        nb["share_token"] = token
        nb["shared_at"] = datetime.now().isoformat()
        _save_notebooks()
    token = nb["share_token"]
    return {"token": token, "url": f"/share/{token}"}

@app.delete("/api/notebooks/{nid}/share")
async def revoke_share(nid: str):
    nb = _get_notebook(nid)
    nb["share_token"] = None
    nb["shared_at"] = None
    _save_notebooks()
    return {"revoked": True}

@app.get("/api/share/{token}")
async def get_shared_notebook(token: str):
    """Public readonly view of a shared notebook."""
    nb = next((n for n in notebooks_db.values() if n.get("share_token") == token), None)
    if not nb:
        raise HTTPException(404, "Share link not found or revoked")
    sources = []
    for sid in nb.get("sources", []):
        s = sources_db.get(sid)
        if s:
            sources.append({k: v for k, v in s.items() if k not in ("content", "file_path")})
    return {
        "id": nb["id"],
        "name": nb["name"],
        "description": nb["description"],
        "created_at": nb["created_at"],
        "sources": sources,
        "token": token,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Generation endpoints (notebook-scoped single source)
# ──────────────────────────────────────────────────────────────────────────────

def _require_ready_in_notebook(nid: str, sid: str) -> dict:
    nb = _get_notebook(nid)
    if sid not in nb.get("sources", []):
        raise HTTPException(404, "Source not in notebook")
    s = sources_db.get(sid)
    if not s:
        raise HTTPException(404, "Source not found")
    if s.get("status") != "ready":
        raise HTTPException(400, f"Source status is '{s.get('status')}', not ready")
    return s


@app.post("/api/notebooks/{nid}/sources/{sid}/generate/presentation")
async def nb_gen_pptx(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        out = OUTPUTS / f"{sid}_presentation.pptx"
        await asyncio.get_event_loop().run_in_executor(None, gen_presentation, s, out, lang)
        url = f"/outputs/{sid}_presentation.pptx"
        sources_db[sid]["pptx_url"] = url; _save_sources()
        return {"url": url}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/audio")
async def nb_gen_audio(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        out = OUTPUTS / f"{sid}_audio.mp3"
        await asyncio.get_event_loop().run_in_executor(None, gen_audio, s, out, lang)
        url = f"/outputs/{sid}_audio.mp3"
        sources_db[sid]["audio_url"] = url; _save_sources()
        return {"url": url}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/infographic")
async def nb_gen_infographic(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        out = OUTPUTS / f"{sid}_infographic.png"
        await asyncio.get_event_loop().run_in_executor(None, gen_infographic, s, out)
        url = f"/outputs/{sid}_infographic.png"
        sources_db[sid]["infographic_url"] = url; _save_sources()
        return {"url": url}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/mindmap")
async def nb_gen_mindmap(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        out = OUTPUTS / f"{sid}_mindmap.png"
        await asyncio.get_event_loop().run_in_executor(None, gen_mindmap, s, out)
        url = f"/outputs/{sid}_mindmap.png"
        sources_db[sid]["mindmap_url"] = url; _save_sources()
        return {"url": url}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/studyguide")
async def nb_gen_study(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_study_guide, s, lang)
        sources_db[sid]["study_guide"] = text; _save_sources()
        return {"content": text}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/faq")
async def nb_gen_faq(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_faq, s, lang)
        sources_db[sid]["faq"] = text; _save_sources()
        return {"content": text}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/briefing")
async def nb_gen_briefing(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_briefing, s, lang)
        sources_db[sid]["briefing"] = text; _save_sources()
        return {"content": text}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/timeline")
async def nb_gen_timeline(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_timeline, s, lang)
        sources_db[sid]["timeline"] = text; _save_sources()
        return {"content": text}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))

@app.post("/api/notebooks/{nid}/sources/{sid}/generate/glossary")
async def nb_gen_glossary(nid: str, sid: str, lang: str = Query('ru')):
    try:
        s = _require_ready_in_notebook(nid, sid)
        text = await asyncio.get_event_loop().run_in_executor(None, gen_glossary, s, lang)
        sources_db[sid]["glossary"] = text; _save_sources()
        return {"content": text}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


# ── Multi-source generation (notebook-scoped) ─────────────────────────────────

@app.post("/api/notebooks/{nid}/generate/{gen_type}")
async def nb_multi_generate(nid: str, gen_type: str, payload: dict):
    try:
        nb = _get_notebook(nid)
        source_ids = payload.get("source_ids", [])
        lang = payload.get("lang", "ru")
        # Validate all sids belong to notebook
        valid_sids = [sid for sid in source_ids if sid in nb.get("sources", [])]
        if not valid_sids:
            valid_sids = nb.get("sources", [])
        if not valid_sids:
            raise HTTPException(400, "No sources in notebook")

        merged = _merge_sources(valid_sids)
        key = nid + "_" + "_".join(sorted(valid_sids))[:30]

        if gen_type == "presentation":
            out = OUTPUTS / f"{key}_presentation.pptx"
            await asyncio.get_event_loop().run_in_executor(None, gen_presentation, merged, out, lang)
            return {"url": f"/outputs/{key}_presentation.pptx"}
        elif gen_type == "audio":
            out = OUTPUTS / f"{key}_audio.mp3"
            await asyncio.get_event_loop().run_in_executor(None, gen_audio, merged, out, lang)
            return {"url": f"/outputs/{key}_audio.mp3"}
        elif gen_type == "infographic":
            out = OUTPUTS / f"{key}_infographic.png"
            await asyncio.get_event_loop().run_in_executor(None, gen_infographic, merged, out)
            return {"url": f"/outputs/{key}_infographic.png"}
        elif gen_type == "mindmap":
            out = OUTPUTS / f"{key}_mindmap.png"
            await asyncio.get_event_loop().run_in_executor(None, gen_mindmap, merged, out)
            return {"url": f"/outputs/{key}_mindmap.png"}
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
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


# ── Legacy endpoints (kept for compatibility) ─────────────────────────────────

@app.get("/api/sources")
async def list_sources_legacy():
    return [{k: v for k, v in s.items() if k != "content"} for s in sources_db.values()]

@app.post("/api/sources/upload")
async def upload_legacy(background_tasks: BackgroundTasks,
                        file: Optional[UploadFile] = File(None),
                        url: Optional[str] = Form(None)):
    # Find or create default notebook
    if not notebooks_db:
        nid = "nb_" + uuid.uuid4().hex[:10]
        notebooks_db[nid] = {"id": nid, "name": "My Research", "description": "",
                              "created_at": datetime.now().isoformat(),
                              "sources": [], "share_token": None, "shared_at": None}
        _save_notebooks()
    nid = next(iter(notebooks_db))
    return await upload_to_notebook(nid, background_tasks, file, url)

@app.get("/api/sources/{sid}")
async def get_source_legacy(sid: str):
    s = sources_db.get(sid)
    if not s: raise HTTPException(404, "Not found")
    return {k: v for k, v in s.items() if k != "content"}

@app.delete("/api/sources/{sid}")
async def delete_source_legacy(sid: str):
    s = sources_db.pop(sid, None)
    if not s: raise HTTPException(404, "Not found")
    # Remove from notebook
    for nb in notebooks_db.values():
        if sid in nb.get("sources", []):
            nb["sources"].remove(sid)
    _save_all()
    fp = s.get("file_path")
    if fp and Path(fp).exists():
        Path(fp).unlink(missing_ok=True)
    return {"deleted": sid}

@app.post("/api/generate/{gen_type}")
async def multi_generate_legacy(gen_type: str, payload: dict):
    source_ids = payload.get("source_ids", [])
    lang = payload.get("lang", "ru")
    if not source_ids:
        raise HTTPException(400, "No source_ids provided")
    merged = _merge_sources(source_ids)
    key = "_".join(sorted(source_ids))[:40]

    if gen_type == "presentation":
        out = OUTPUTS / f"{key}_presentation.pptx"
        await asyncio.get_event_loop().run_in_executor(None, gen_presentation, merged, out, lang)
        return {"url": f"/outputs/{key}_presentation.pptx"}
    elif gen_type == "audio":
        out = OUTPUTS / f"{key}_audio.mp3"
        await asyncio.get_event_loop().run_in_executor(None, gen_audio, merged, out, lang)
        return {"url": f"/outputs/{key}_audio.mp3"}
    elif gen_type == "infographic":
        out = OUTPUTS / f"{key}_infographic.png"
        await asyncio.get_event_loop().run_in_executor(None, gen_infographic, merged, out)
        return {"url": f"/outputs/{key}_infographic.png"}
    elif gen_type == "mindmap":
        out = OUTPUTS / f"{key}_mindmap.png"
        await asyncio.get_event_loop().run_in_executor(None, gen_mindmap, merged, out)
        return {"url": f"/outputs/{key}_mindmap.png"}
    elif gen_type == "studyguide":
        return {"content": await asyncio.get_event_loop().run_in_executor(None, gen_study_guide, merged, lang)}
    elif gen_type == "faq":
        return {"content": await asyncio.get_event_loop().run_in_executor(None, gen_faq, merged, lang)}
    elif gen_type == "briefing":
        return {"content": await asyncio.get_event_loop().run_in_executor(None, gen_briefing, merged, lang)}
    elif gen_type == "timeline":
        return {"content": await asyncio.get_event_loop().run_in_executor(None, gen_timeline, merged, lang)}
    elif gen_type == "glossary":
        return {"content": await asyncio.get_event_loop().run_in_executor(None, gen_glossary, merged, lang)}
    else:
        raise HTTPException(400, f"Unknown generation type: {gen_type}")


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(payload: dict):
    try:
        question     = payload.get("question", "").strip()
        source_ids   = payload.get("source_ids", [])
        chat_history = payload.get("history", [])
        if not question:
            raise HTTPException(400, "No question provided")

        context_parts = []
        for sid in source_ids:
            s = sources_db.get(sid)
            if not s or s.get("status") != "ready": continue
            content = s.get("content", ""); a = s.get("analysis", {}) or {}
            part = f"=== SOURCE: {s['name']} ===\n"
            part += content[:10000] if content else f"Summary: {a.get('summary','')}\nKey points: {'; '.join(a.get('key_points',[]))}\n"
            context_parts.append(part)

        history_text = "".join(f"\n{m.get('role','user').upper()}: {m.get('content','')}"
                               for m in chat_history[-6:])

        if not context_parts:
            prompt = f"You are NoteLM, a helpful AI research assistant.\n{history_text}\nUSER: {question}\nASSISTANT:"
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
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Chat error: {str(e)}")


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "3.0.0",
            "notebooks": len(notebooks_db), "sources": len(sources_db)}

@app.on_event("startup")
async def startup_event():
    _load_dbs()
    _migrate()
    for sid, s in sources_db.items():
        if s.get("status") == "pending":
            s["status"] = "error"
            s["error"] = "Interrupted by restart"
    _save_sources()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=False)
