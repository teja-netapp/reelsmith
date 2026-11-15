# Changelog

## 0.4.1
- **Choose your own Gemini model.** In Settings → Google Gemini, type any model name in the Model field (e.g. a free-tier `gemini-3.1-flash-lite`), or leave it empty for Auto. Suggestions list every model your key can see, including Gemma.
- **Test model button.** It checks that the name exists and makes one tiny request, so you know your key's tier can actually use the model:
  - an unknown name is refused and not saved;
  - a model with no free-tier quota (`limit: 0`) gets a clear warning;
  - models without JSON mode are detected.
- **Works with more models:**
  - Requests step down from strict JSON-schema output to plain text. This covers models without system instructions or JSON mode, such as Gemma.
  - The output token limit is read from the model itself.
  - A `models/` prefix copied from the docs is accepted.
- The edit report shows whether the model was **your choice** or **auto-detected**, and which output format worked.

## 0.4.0
- **New: Settings in the app (⚙ Settings, or click the Gemini / Local AI pill).** You no longer need to edit `.env` by hand.
  - **Gemini and Claude API keys:** paste a key and click *Save & verify*. The key is checked with Google or Anthropic **before** it's saved, and a rejected key is never written. If you're offline, the key is saved with a warning.
  - **Masked keys:** a saved key is only ever shown masked (`AIza…abcd`). Buttons let you test it again or remove it.
  - **Model picker:** after verifying, choose from the models your key can actually use. The panel also shows whether Gemini 4 Hydrogen is available to your key.
  - **Local AI (Ollama):** set the address, pick the model, and test the connection. If the model isn't downloaded yet, the panel gives you the exact `ollama pull …` command to copy.
  - **Default AI editor:** choose which editor new reels use.
- **No restart needed:** changes apply immediately, and saving updates `.env` in place, keeping your comments and other settings. The file is `chmod 600`, so only you can read it.
- **Security:**
  - The app only answers requests addressed to localhost, which blocks DNS rebinding.
  - It rejects changes sent from other websites open in your browser (cross-site requests).
  - The settings API responses are never cached.
- **Fewer surprises:** if a key is also exported in your shell, the panel warns that the shell value will win on the next restart.
- Tests: 7 new settings tests (16 in total).

## 0.3.0
- **New: "How this reel was made" panel** for every version.
  - **Editor:** which editor and exact model made it, the shot-logging model and cache, token counts, planning time, and any editors that were skipped or failed before it.
  - **Brief and settings:** what you asked for and the options used.
  - **Look and sound:** the color grade, text and music used.
  - **Edit decision list:** every shot with its source in/out, time on screen, speed, camera move, transition, reframing, sound and beat adjustment. Click a row to jump to that moment in the video.
  - **Changes made automatically:** every fix after the AI's plan, with the reason (validation, assistant editor, timeline, beat sync).
- **Version history:** v1, v2, … chips switch between versions, each with its own video and report. Download any version.
- **Downloads:** the report as Markdown or JSON, plus the AI's raw plan exactly as the model returned it.
- **MP4 metadata:** the editor and model are written into the video file's title and comment tags.
- **Fix:** fullscreen no longer crops the video (`object-fit: contain`).
- **Fix:** the right-hand panel no longer gets squeezed after a video loads or when leaving fullscreen. The phone column now has a fixed width, where it used to stretch to the video's 1080 px.
- **Fix:** the assistant editor no longer lets the same footage appear twice. Lengthening a shot stops at the next shot taken from the same clip, and partial repeats are trimmed.

## 0.2.0
- **Choose your AI editor:** Gemini (e.g. Gemini 4 Hydrogen), a free local model on Ollama (default `qwen3-vl:8b-instruct`), Claude, or built-in rules. You can pick one per reel in the app; *Auto* uses the first editor that's ready.
- **Shared editing playbook:** every editor follows the same principles (hook, pacing, transition restraint, flow, audio, captions, beat sync).
- **Local editor pipeline:** the vision model logs each shot (cached), writes a compact plan with every field constrained by a JSON schema, then code converts it to exact cut points.
- **Craft pass (`craft.py`):** catches what an AI editor misses. All editors get word-safe speech cuts, removal of repeated moments and a duration fit. Local models and the built-in rules also get pacing bounds, ≤30% stylised transitions, camera-move variety, strongest hook first and a resolved ending.
- **GitHub-ready:** `.gitignore`, `scripts/publish_github.sh`, `update.sh`, a pytest suite and a GitHub Actions workflow.

## 0.1.0
- First version: analysis, Claude planner, cinematic renderer, web UI, CLI.
