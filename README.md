---
title: Citation Genealogy Agent
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Citation Genealogy Agent

Traces the genealogy of a claim through its citation chain to find the primary source.

Built with **LangGraph** + **Groq** + **FastAPI**.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY
python webapp.py
```

Then open http://127.0.0.1:8000

## Deploy on Hugging Face Spaces

This repo is ready for HF Spaces with the **Docker** SDK.

1. Create a new Space (SDK: Docker, blank template).
2. Add your `GROQ_API_KEY` under *Settings → Variables and secrets*.
3. Push this repo to the Space remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   git push hf main
   ```

The `Dockerfile` exposes port `7860` (the port HF Spaces routes traffic to).

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq API key for LLM calls |

## Project structure

```
agent/         # LangGraph nodes, state, tools
templates/     # HTML UI
webapp.py      # FastAPI app (entrypoint)
main.py        # CLI entrypoint
```
