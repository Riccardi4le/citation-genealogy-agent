# Citation Genealogy Agent

A LangGraph-powered agent that traces citation chains backward from a claim to its primary source, detecting distortion at each hop.

## What it does

Given a claim and the paper it appears in, the agent:
1. Fetches the paper via [OpenAlex](https://openalex.org/)
2. Uses an LLM (Groq / Llama 3.3 70B) to locate the claim and identify which reference supports it
3. Follows that reference recursively, comparing how the claim evolves at each hop
4. Produces a distortion score and a full report with a Mermaid citation tree

## Installation

```bash
git clone https://github.com/your-username/citation-genealogy-agent.git
cd citation-genealogy-agent

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)
```

## Web Interface

```bash
python webapp.py
# → Open http://127.0.0.1:8000
```

The browser UI provides:
- **Input form** — claim, source reference, depth slider (1–8)
- **Live activity log** — every fetch and analysis verdict streamed in real time via SSE
- **Incremental citation tree** — Mermaid diagram that builds node by node as papers are analyzed, each node color-coded by verdict
- **Full report** — rendered markdown with the citation tree diagram inline, plus a download button for the `.md` file

No extra setup needed — the web interface uses the same agent and `.env` as the CLI.

---

## CLI Usage

```bash
# Basic
python main.py "claim text" --source "Author Year OR DOI"

# With DOI (most reliable)
python main.py "omega-3 reduces depression symptoms" \
  --source "10.1017/S0007114514000841" \
  --max-depth 3

# With author + year
python main.py "exercise improves memory" \
  --source "Erickson et al. 2011"

# Save report to file
python main.py "sugar impairs cognition" \
  --source "Smith et al. 2019" \
  --output report.md

# Full help
python main.py --help
```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your Groq API key (free at [console.groq.com](https://console.groq.com)) |
| `OPENALEX_EMAIL` | No | Email for OpenAlex polite pool (faster rate limits) |
| `LLM_MODEL` | No | Groq model to use (default: `llama-3.3-70b-versatile`) |

## Output

The report includes:
- **Mermaid diagram** of the full citation chain
- **Hop-by-hop table** showing how the claim evolves at each level
- **Distortion score** (0–1) with per-node verdicts: `supported`, `partial`, `distorted`, `contradicted`, `not_found`
- **Warnings** for papers with no abstract or unresolvable references

## Recent fixes

The following robustness issues were fixed during review:

- **OpenAlex text search used the first result blindly.**
  Action taken: search candidates are now ranked with a lightweight match score based on title overlap, year, author tokens, and DOI presence before selecting the work to trace.
- **The LLM prompt sounded more certain than the available evidence.**
  Action taken: the analysis prompt now states explicitly that it is working from title, abstract, and reference metadata only, and instructs the model to stay conservative.
- **Primary-source and distortion judgments could overstate confidence.**
  Action taken: the agent now frames those judgments as excerpt-based rather than pretending it inspected the full paper text.

Recommended usage after the fix:

- Prefer DOI input when available.
- Treat author/year queries as best-effort resolution, not perfect lookup.

### Web UI fixes

The browser interface was hardened as well:

- **User-visible logs and report rendering were vulnerable to unsafe HTML insertion.**
  Action taken: dynamic content is now escaped or sanitized before being inserted into the page.
- **Mermaid was configured too loosely for untrusted content.**
  Action taken: Mermaid now runs with stricter security settings.
- **Form input validation was too soft.**
  Action taken: the FastAPI request model now validates claim length, source length, and depth range server-side.

## Architecture

```
START → ingest → fetch → analyze → [route] → fetch (loop)
                                           ↘ report → END
```

The web interface (`webapp.py`) runs the same graph in a background thread and streams state diffs to the browser via Server-Sent Events (SSE). Built with FastAPI + uvicorn.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) + [Groq](https://groq.com/) + [OpenAlex API](https://openalex.org/) + [FastAPI](https://fastapi.tiangolo.com/).

## Limitations

- Analysis is **abstract-only** — no PDF parsing
- OpenAlex coverage varies; some papers may not be found
- In practice the agent should be treated as metadata/abstract-based rather than full-text aware.
- Maximum 50 referenced works fetched per paper

## License

MIT
