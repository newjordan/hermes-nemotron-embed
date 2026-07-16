# Nemotron Embed installed

Enable semantic context compaction and start the local server:

```bash
hermes nemotron-embed setup --start
```

The first start creates an isolated `uvx` environment and downloads the model,
so it can take several minutes. Managed serving requires Linux or WSL2 with a
supported NVIDIA GPU. Run `hermes nemotron-embed status` to verify readiness.

To use an existing vLLM server instead:

```bash
hermes nemotron-embed setup --endpoint http://127.0.0.1:8000/v2/embed
```
