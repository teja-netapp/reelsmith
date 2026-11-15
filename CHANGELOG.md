# Changelog

## 0.2.0
- **Choose your AI editor:** Gemini (e.g. Gemini 4 Hydrogen), a free local model on Ollama (default `qwen3-vl:8b-instruct`), Claude, or built-in rules. You can pick one per reel in the app; *Auto* uses the first editor that's ready.
- **Shared editing playbook:** every editor follows the same principles (hook, pacing, transition restraint, flow, audio, captions, beat sync).
- **Local editor pipeline:** the vision model logs each shot (cached), writes a compact plan with every field constrained by a JSON schema, then code converts it to exact cut points.
- **Craft pass (`craft.py`):** catches what an AI editor misses. All editors get word-safe speech cuts, removal of repeated moments and a duration fit. Local models and the built-in rules also get pacing bounds, ≤30% stylised transitions, camera-move variety, strongest hook first and a resolved ending.
- **GitHub-ready:** `.gitignore`, `scripts/publish_github.sh`, `update.sh`, a pytest suite and a GitHub Actions workflow.

## 0.1.0
- First version: analysis, Claude planner, cinematic renderer, web UI, CLI.
