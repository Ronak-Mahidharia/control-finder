# The MCP server

AI assistants (Claude Desktop, Claude Code, and other MCP clients) can use Control Finder as three read-only tools. The server runs on this computer and talks over stdio. Its only network calls go to Ollama on this computer, so questions never leave it.

## The tools
| Tool | What it does |
|---|---|
| `search_controls` | The controls that best fit a question: `hybrid` (the default), `vector`, or `keyword`, up to 20. Keyword search works even without Ollama. |
| `get_control` | A control's statement, guidance, and SP 800-53A assessment objectives, by id or label (`ia-5.1` or `IA-5(1)`). A withdrawn id says so. |
| `answer_question` | A cited answer from the local model (qwen3:8b), from the controls the search found. When none of them answer the question, a fixed sentence says it isn't covered. |

Questions are capped at 2,000 characters. The search runs in memory from the committed embeddings, which step 5 showed gives the same rankings as the database, so no database is needed.

## What it tells the assistant
```text
Find the NIST SP 800-53 Rev 5.2.0 security controls that answer a question, and read them.
- Cite controls by their label (for example IA-5(1)), and describe only what a control's own text says; get_control returns it.
- answer_question answers only from the controls it was shown. When it says a question isn't covered, tell the person so, and don't present general knowledge as NIST's.
- Search results are candidates, not a compliance decision: which controls a system needs depends on its baseline and risk assessment.
- Not affiliated with or endorsed by NIST.
```

## Set it up
Needs Python 3.12 or newer, Ollama with nomic-embed-text (and qwen3:8b for `answer_question`), and a copy of this repository, because NIST's data and the embeddings are read from it.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-mcp.txt && pip install -e . --no-deps
python scripts/download_data.py                 # NIST's files, pinned and hash-checked
ollama pull nomic-embed-text && ollama pull qwen3:8b
python -m controlfinder.mcp_server              # starts and waits for an assistant; press Ctrl+C to stop
```

The examples below use `/path/to/control-finder` for the folder you cloned. Use the full path, not a relative one.

**Claude Code:**

```bash
claude mcp add control-finder -- /path/to/control-finder/.venv/bin/python -m controlfinder.mcp_server
```

**Claude Desktop:**
1. Open the Claude menu, then Settings…, then Developer, then Edit Config. On macOS this opens `~/Library/Application Support/Claude/claude_desktop_config.json`.
2. Add the server:

   ```json
   {
     "mcpServers": {
       "control-finder": {
         "command": "/path/to/control-finder/.venv/bin/python",
         "args": ["-m", "controlfinder.mcp_server"]
       }
     }
   }
   ```
3. Save, then quit Claude Desktop completely and open it again.

**Other MCP clients:** run the same command, `/path/to/control-finder/.venv/bin/python -m controlfinder.mcp_server`, as a stdio server.

## How it's checked
- `tests/test_mcp_server.py` connects a real MCP client to the server in-process. It checks the tools and their limits, and every question's search results against the published rankings, with the committed embeddings standing in for the model. A stand-in replaces the answer model.
- `scripts/run_mcp_check.py` runs the server over stdio with the real models: results in [results-mcp.md](results-mcp.md).
- Not affiliated with or endorsed by NIST. A reference tool, not legal or compliance advice.
