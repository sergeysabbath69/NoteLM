# NoteLM — AI Research Assistant

NoteLM is a full-featured, self-hosted alternative to Google NotebookLM, powered by **Gemini 1.5 Flash**.

## Features

| Feature | Description |
|---|---|
| 📄 **Source Upload** | PDF, DOCX, TXT, Markdown, URLs |
| 🔍 **AI Analysis** | Summary, key points, topics, entities, sentiment, complexity |
| 📊 **Presentation** | Auto-generated PPTX slide deck |
| 🎧 **Audio Overview** | Text-to-speech narration of summary |
| 📈 **Infographic** | Matplotlib visual: topics, entities, sentiment chart |
| 🕸️ **Mind Map** | NetworkX topic graph visualization |
| 📖 **Study Guide** | Concepts, key terms, review Q&A, practice activities |
| ❓ **FAQ** | 15–20 natural Q&A pairs |
| 📋 **Executive Briefing** | BLUF, findings, implications, recommendations |
| 📅 **Timeline** | Chronological event table |
| 📚 **Glossary** | All terms with plain-language definitions |
| 💬 **Chat** | Multi-turn Q&A grounded in your sources |
| 📝 **Notes** | Persistent research notes workspace |

## Tech Stack

- **Backend:** FastAPI + Uvicorn
- **AI:** Google Gemini 1.5 Flash
- **Visuals:** Matplotlib, NetworkX
- **File parsing:** pdfplumber, python-docx
- **Presentation:** python-pptx
- **Audio:** gTTS
- **Frontend:** Vanilla HTML/CSS/JS (zero frameworks, zero dependencies)

## Quick Start

```bash
pip install -r requirements.txt
GEMINI_API_KEY=your_key python main.py
# Open http://localhost:80
```

## Deployment (systemd)

```ini
[Unit]
Description=NoteLM AI Research Assistant
After=network.target

[Service]
WorkingDirectory=/opt/notelm
ExecStart=/usr/bin/python3 main.py
Restart=always
Environment=GEMINI_API_KEY=your_key

[Install]
WantedBy=multi-user.target
```
