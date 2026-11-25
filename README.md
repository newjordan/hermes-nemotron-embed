# Embedhead — Hermes Nemotron Embed

[![CI](https://github.com/newjordan/hermes-nemotron-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/newjordan/hermes-nemotron-embed/actions/workflows/ci.yml)

<p align="center">
  <img src="art/embedhead-design-review-1967.png" alt="Embedhead — local semantic context for Hermes" width="720">
</p>

Embedhead is a standalone [Hermes Agent](https://github.com/NousResearch/hermes-agent)
context engine. When a conversation reaches the compaction threshold, it asks a
local NVIDIA Nemotron 3 Embed server which older turns still match the current
request, keeps those turns intact, and stores the omitted turns in a
profile-local SQLite archive so Hermes can search them later.

It does not replace Hermes' reasoning model and it does not summarize with an
auxiliary LLM. Compaction is a retrieval decision over whole turns. The
embedding call stays on the endpoint you configure, which defaults to loopback.

## Where it fits

Embedhead is a narrow sidecar. It is not a full memory system. Its job is one
deployment pattern: use a local Nemotron embedding model to choose which
complete conversation turns remain in the active context when Hermes compacts
it.

| Option | Primary mechanism | Best fit |
|---|---|---|
| **Embedhead** | Local Nemotron similarity selects atomic turns at compaction; omitted turns remain searchable in SQLite | NVIDIA GPU users who want a focused, local embedding sidecar |
| [hermes-cmx](https://github.com/arminanton/hermes-cmx) | Hybrid lexical/vector retrieval, verbatim assembly, every-turn injection, optional grounding | Full retrieval-first memory and evidence enforcement |
| [hermes-cursor-compressor](https://github.com/thiswind/hermes-cursor-compressor) | Cursor-style LLM summarization with JSONL history | A smaller alternative summarizer without semantic embeddings |
| Hermes built-in | Auxiliary-LLM summary plus the normal session transcript/search surfaces | Zero additional services and the default supported behavior |

## Requirements

- Hermes Agent with context-engine plugin support. CI tests this plugin against Hermes Agent 0.18.2 at commit `74fc222f17d5fa10400dbd35d75d45f5ae75df8c` on Python 3.11, 3.12, and 3.13.
- `uv` / `uvx` for managed startup.
- Linux or WSL2, a CUDA-capable NVIDIA GPU, and a compatible driver, when this plugin starts vLLM for you.

An endpoint you already run can be anywhere the plugin can reach. The managed
`start` command refuses anything except an `http` loopback URL whose path is
`/v2/embed`.

Version 0.1.1 is a public preview. The automated tests use a fake embedding
client, temporary SQLite databases, and loopback HTTP servers. They do not
download weights and they do not measure a live GPU. Before relying on a
server for an important session, run `hermes nemotron-embed benchmark`
against that server.

## Install and activate

```bash
hermes plugins install newjordan/hermes-nemotron-embed --enable
hermes nemotron-embed setup --start
```

`setup` validates the requested settings, writes them into the active profile's
`config.yaml`, and selects `context.engine: nemotron-embed`. Restart a
long-running gateway so new sessions pick up the engine:

```bash
hermes gateway restart
hermes nemotron-embed status
```

Managed serving uses `uvx` and `vllm==0.25.0`, the release named on the
Nemotron 3 Embed model card for `/v2/embed`. The process binds to `127.0.0.1`
(or `::1` when that is the configured host) and listens on the port in the
endpoint. The first start downloads vLLM and the selected Hugging Face
checkpoint. Managed startup also sets `VLLM_NO_USAGE_STATS=1` and
`DO_NOT_TRACK=1` in the child environment.

The default `1b` preset is `nvidia/Nemotron-3-Embed-1B-BF16`. The `8b` preset
is `nvidia/Nemotron-3-Embed-8B-BF16`:

```bash
hermes nemotron-embed setup --model 8b --start
```

`--model` also accepts a full Hugging Face model id.

The managed command is:

```bash
uvx --from vllm==0.25.0 vllm serve MODEL \
  --runner pooling \
  --host 127.0.0.1 \
  --port PORT \
  --served-model-name MODEL
```

`--convert embed` is omitted on purpose. Nemotron 3 Embed is already a pooling
checkpoint. Passing `--convert embed` asks vLLM to replace that pooler. The
model card's own serve example is `vllm serve MODEL`; `--runner pooling` only
forces the embedding runner.

## Use an existing server

The plugin speaks the vLLM `/v2/embed` retrieval API (`input_type` of `query`
or `document`, `embedding_types: ["float"]`, `truncate: "END"`). Point setup
at a server you already manage:

```bash
vllm serve nvidia/Nemotron-3-Embed-1B-BF16 \
  --runner pooling \
  --host 127.0.0.1 \
  --port 8000

hermes nemotron-embed setup \
  --endpoint http://127.0.0.1:8000/v2/embed
```

For a server protected by vLLM's API-key middleware, keep the token in the
environment. The profile config stores the variable name only:

```bash
hermes config set NEMOTRON_EMBED_API_KEY your-token
hermes nemotron-embed setup \
  --endpoint http://127.0.0.1:8000/v2/embed \
  --api-key-env NEMOTRON_EMBED_API_KEY
```

`status` reports `api_key_configured` and never prints the token. Requests do
not follow HTTP redirects and do not use `HTTP_PROXY` / `HTTPS_PROXY`, so a
configured loopback endpoint cannot be replayed at another origin by a proxy
or a 307.

## How compaction works

When Hermes reaches the configured context threshold, the engine:

1. Splits history into atomic user turns and keeps each assistant/tool chain with its user message. Leading non-user messages stay in a preamble that is always kept.
2. Always preserves the first `protect_first_turns` groups and the last `protect_last_turns` groups.
3. Embeds the current request (or the manual `/compress` focus topic) as a query and the older candidate groups as documents.
4. Ranks candidates by cosine similarity plus a small recency bonus, then keeps the highest-scoring groups that still fit the token budget. Kept groups are restored to chronological order.
5. Stores every omitted group, including its embedding when the server responded, in the profile-local SQLite archive.

The token budget is `max(1024, context_length * target_ratio)`. Hermes' fixed
overhead (system prompt and tool schemas) counts against that budget. If the
embedding server errors, compaction falls back to recency order and still
archives the omitted turns. If every candidate fits, or the result would not
shrink the request, the original message list is returned unchanged so Hermes
does not rotate the session for zero savings.

The active engine exposes `nemotron_context_search`. Hermes can call it to
recover omitted turns for the current session and for ancestor sessions
created by compression. A lineage link that would make two sessions see each
other is ignored. Search uses the stored vectors when their dimension matches
the query vector, and otherwise falls back to lexical overlap. ASCII words are
matched directly. Unspaced text such as Chinese or Japanese is indexed as
overlapping bigrams so a short query can hit a longer turn. Each result's
`content` is capped at 4,000 characters and `truncated` says when the stored
text was longer.

`memory_context` from newer Hermes hosts is accepted and ignored. Provider
memory is not mixed into the retrieval query.

## Configuration

`hermes nemotron-embed setup` writes these keys under `nemotron_embed` and sets
`context.engine`:

```yaml
nemotron_embed:
  endpoint: http://127.0.0.1:8089/v2/embed
  model: nvidia/Nemotron-3-Embed-1B-BF16
  api_key_env: NEMOTRON_EMBED_API_KEY
  threshold: 0.75
  target_ratio: 0.35

context:
  engine: nemotron-embed
```

A trailing slash on the endpoint is stripped before it is saved. `threshold`
must be from 0.25 through 0.95, `target_ratio` must be from 0.05 through 0.90,
and `target_ratio` must be lower than `threshold`. The endpoint must be an
`http` or `https` URL without embedded credentials. Boolean values are
rejected for numeric settings.

These optional keys use the defaults below when they are absent. Edit them in
`config.yaml`; `setup` preserves values that are already there.

| Key | Default | Meaning |
|---|---|---|
| `protect_first_turns` | 1 | Leading turn groups always kept (0–20) |
| `protect_last_turns` | 4 | Trailing turn groups always kept (1–50) |
| `request_timeout` | 120 | Embedding HTTP timeout in seconds (1–600) |
| `embedding_dimensions` | 1024 | Prefix of each vector kept and re-normalized (128–4096) |
| `max_chunk_chars` | 12000 | Characters of each turn group sent to the embedder |
| `recency_weight` | 0.05 | Added to cosine after scaling position to 0–1 (0–0.5) |
| `archive_path` | `$HERMES_HOME/nemotron-embed/archive.db` | SQLite archive location |

The model card states that Nemotron-3-Embed-1B-BF16 returns a 2048-dimensional
vector and that a prefix remains usable after it is sliced and L2-normalized.
`embedding_dimensions` is that prefix. A smaller value makes the archive
smaller and the in-process scan cheaper.

State is profile scoped:

- Archive: `$HERMES_HOME/nemotron-embed/archive.db` (mode `0600`; the `nemotron-embed` directory is `0700`)
- WAL/SHM sidecars, when SQLite creates them: mode `0600`
- Server state: `$HERMES_HOME/nemotron-embed/server.json` (mode `0600`)
- Server log: `$HERMES_HOME/nemotron-embed/vllm.log` (mode `0600`)

Conversation text is sent only to the configured embedding endpoint. Archived
turns are plaintext in that SQLite database. Treat the Hermes profile
directory, its WAL files, and its backups as conversation history. `stop`
signals only the process recorded in `server.json`, and only when the recorded
start time still matches, so a recycled PID is not killed.

See `examples/config.yaml` for a full profile snippet.

## Operations

```bash
hermes nemotron-embed status
hermes nemotron-embed start
hermes nemotron-embed stop
hermes nemotron-embed benchmark
hermes plugins update nemotron-embed
```

`status` prints JSON. `server.running` means this profile owns a live process.
`server.healthy` means a one-string `/v2/embed` probe succeeded, including for
an endpoint you started yourself. `start` and `stop` print a short status
block. `start --no-wait` returns after the process is recorded. `benchmark`
prints a JSON report and the word `passed` or `failed` on stderr. A failed
benchmark raises so the Hermes command exits unsuccessfully. The report checks
that the smoke conversation shrank, that one planted fact stayed in context,
that an omitted turn can be searched back, and that user/assistant pairs were
not split. It is a deployment check for the configured endpoint, not a
production savings claim.

To roll back, select Hermes' built-in compressor before disabling the plugin:

```bash
hermes nemotron-embed stop
hermes config set context.engine compressor
hermes plugins disable nemotron-embed
hermes plugins remove nemotron-embed
```

Removal leaves `$HERMES_HOME/nemotron-embed/` in place.

If `nemotron_embed` in `config.yaml` is invalid, the plugin still registers
the CLI so `setup` can repair it, and it does not register the context engine.
Hermes keeps using its built-in compressor until the config is valid.
`status` prints the validation error and the repair command.

## Model and software licenses

This plugin's source code is MIT licensed. See [LICENSE](LICENSE). NVIDIA's
model weights and model configuration are separate artifacts under the
OpenMDW License Agreement 1.1. Read the model card before downloading a
checkpoint:

- [Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
- [Nemotron-3-Embed-8B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16)

## Related Hermes projects

- [hermes-cmx](https://github.com/arminanton/hermes-cmx) — retrieval-first, verbatim context assembly with hybrid search and optional grounding.
- [hermes-cursor-compressor](https://github.com/thiswind/hermes-cursor-compressor) — Cursor-style summarization and searchable JSONL history.
- [hermes-mastra](https://github.com/srinitude/hermes-mastra) — a broader Mastra memory provider with semantic-search tools and an optional context wrapper.
- [Hermes context-engine documentation](https://github.com/NousResearch/hermes-agent/blob/main/website/docs/developer-guide/context-engine-plugin.md) — the host contract this plugin implements.

## Development

The plugin is a flat Hermes plugin directory (`package = false` in
`pyproject.toml`). Tests import it from this checkout. They also import the
pinned Hermes host, because the engine subclasses `agent.context_engine.ContextEngine`.

```bash
uv sync --group dev
git clone https://github.com/NousResearch/hermes-agent.git .hermes-agent
git -C .hermes-agent fetch --depth 1 origin 74fc222f17d5fa10400dbd35d75d45f5ae75df8c
git -C .hermes-agent checkout --detach FETCH_HEAD
uv sync --project .hermes-agent --locked --extra dev --python 3.12
PYTHONPATH=.hermes-agent .hermes-agent/.venv/bin/python -m pytest
uv run ruff check .
uv run ruff format --check .
python3 scripts/scan_secrets.py
```

`.hermes-agent/` is gitignored. CI runs that pytest suite on Python 3.11, 3.12,
and 3.13, then `ruff check`, `ruff format --check`, and `scripts/scan_secrets.py`.
Dev dependencies are pinned in `uv.lock` (`pytest==9.0.2`, `ruff==0.15.10`).

With a compatible server already running, the stdlib smoke eval reports
Recall@1, Recall@3, MRR, dimensions, and batch latency for a small fixed
corpus. It does not download weights by itself. It is a serving check, not an
RTEB score:

```bash
python evals/retrieval_smoke.py \
  --endpoint http://127.0.0.1:8089/v2/embed \
  --label local-gpu
```
