# Reelsmith: an AI cinematic reel editor (runs on your Mac)

You drop in clips or full-length videos, add an optional music track and describe the reel you want. Reelsmith returns a vertical 1080×1920 Instagram Reel with cinematic color grading, transitions, camera moves, beat-synced cuts, smart 9:16 reframing and word-by-word captions. You can then revise it with plain sentences like "faster, warmer, open on the drone shot".

## Choose your AI editor

| Editor | Cost | Quality of the edit | Speed on an M4 Air | Privacy |
|---|---|---|---|---|
| **Gemini** (e.g. Gemini 4 Hydrogen) | Free API tier, or pay-as-you-go | ★★★★☆ Sees all your shots at once and follows detailed briefs | ~20–60 s | Thumbnails and the brief are sent to Google. The free tier may use them to improve Google's products. |
| **Local AI**, Ollama `qwen3-vl:8b-instruct` | Free | ★★★☆☆ Good. The craft pass fills in what an 8B model misses | ~2–3 min for a new reel, ~1 min for a revision | Everything stays on your Mac |
| **Claude** | Paid API | ★★★★☆ | ~30–60 s | Sent to Anthropic |
| **Rules** (no AI) | Free | ★★☆☆☆ Keyword rules and quality scores | instant | Local |

Pick one per reel in the app, or set `LLM_PROVIDER` in `.env`. **Auto** uses the first editor that's ready, in this order: Claude → Gemini → Local → Rules. If an editor fails (no network, quota, a bad key), the reel falls back to the next one, so you always get a video.

> Your **Gemini app subscription is not an API key.** Get a free key at <https://aistudio.google.com/apikey>, then in the app open **⚙ Settings → Google Gemini**, paste it and click **Save & verify**. The API is billed separately from the app subscription. The free tier is enough for personal use; with paid billing a reel costs roughly a few cents.

### Settings (API keys, models, default editor)

Open **⚙ Settings** in the top bar, or click the Gemini / Local AI status pill.

- **Gemini / Claude:** paste a key and click **Save & verify**. The key is checked with Google or Anthropic first, and a rejected key is never saved. Once saved, the key only appears masked (`AIza…abcd`). Use **Test saved key** or **Remove key** to manage it.
- **Your own Gemini model:** type any model name (e.g. a free-tier `gemini-3.1-flash-lite`), or leave it empty for Auto. **Test model** makes one tiny request to confirm your key's tier can use it. Unknown names are refused, and models with no free quota get a warning.
- **Local AI (Ollama):** set the address and model, then click **Test connection**. If the model isn't downloaded yet, the panel shows the exact `ollama pull …` command.
- **Default AI editor:** the editor new reels start with.

Changes apply immediately, without a restart. They're stored in `.env` on this Mac, which is git-ignored and readable only by you. Editing `.env` by hand still works too.

The app only answers requests to `localhost` and refuses changes coming from other websites, so it's safe to keep it running while you browse.

All editors follow the same **editing playbook** (`app/playbook.py`): hook in the first second, pacing that matches the mood, rare and well-chosen transitions, story flow, speech-aware audio, beat sync. Every plan then goes through the **craft pass** (`app/craft.py`), which acts as an assistant editor and fixes rule breaks before rendering.

## It learns your taste (and gets better with every edit)

The local AI's weights don't change. Retraining an 8B model on a laptop is impractical, and a few dozen edits wouldn't be enough data. What learns is a **taste model** wrapped around it. That's the same reward-model idea that powers RLHF, applied at the scale of one person's preferences:

| You do this | It learns |
|---|---|
| Rate a version ★–★★★★★ | That edit's style is better or worse. It also compares it with your other versions of the same reel |
| Download a version | A weaker "this one" signal |
| Pick a grade in **Look Lab** | Which look families you like, for this kind of reel |
| Ask for a revision ("faster, warmer, no glitch") | Your request is parsed into concrete preferences straight away |

What it does with that on the next reel:

1. **House style in the prompt:** the local AI is told your learned preferences and shown your best-rated past edits as examples.
2. **Assistant editor:** your learned shot length, transition budget and dislikes (e.g. no glitch, no slow motion) are enforced.
3. **Variant search:** after the AI writes its plan, about 15–35 variants (pace, transitions, look, grain, letterbox, slow motion) are scored by the taste model, and the best one is rendered. While it's still unsure, it stays close to the AI's plan. Now and then it tries a near-best variant or a look you haven't seen, so it keeps discovering what you like.

You can see all of it in **⚙ Settings → Learning**: what it learned, how sure it is, and your rating trend (first 10 vs last 10). The same panel has controls to pause, reset, or export the dataset as JSONL, which you could use for a real fine-tune later. Each reel's report shows what the taste model changed and why.

`python tests/simulate_learning.py` runs a simulated user with a hidden taste through 40 reels. Their ratings rise over the rounds, and the learned house style matches the hidden taste.

> **Honest expectation:** an 8B local model won't out-think Claude on a complex brief. The two-pass story → edit planning, three-frame shot logging and your learned taste close much of the gap, and for **your** preferences specifically, the taste model can beat any general model. Rate your first 10–20 reels to train it.

## Your instructions are rules

Some sentences in your brief or a revision are instructions, not inspiration: "don't touch the colour grading, just add transitions", "no text", "only cuts", "keep the clips in order", "no slow-mo", "title 'GOA 24'", "15 second reel", "add whip pans and light leaks", "no glitches". Reelsmith detects them as you type and shows them as chips under the box. They are enforced on the final edit, whatever the AI suggested. The report's **Your instructions** box lists each rule, the sentence it came from, and anything that had to be fixed. Later instructions win.

## Color

- **Shot matching (on by default):** every shot's white balance and exposure is matched to the rest of the reel before the look, so cuts don't jump between color temperatures. Only near-neutral pixels are measured, so grass or sky can't fool it.
- **Color grading switch:** turn it off to keep your footage's original colours. Only transitions, effects, text and captions are added: no LUT and no shot matching, so the pixels match the source. **Strength** (0–150%) scales any look.
- **40 CapCut-style presets in 10 categories:**
  - **Life:** Natural, Fresh, Sunny, Food, Portrait.
  - **Vibrant:** Vivid, Pop, Tropical, Candy.
  - **Warm:** Golden Hour, Desert, Autumn.
  - **Film:** Kodak 2383, Portra, Eterna, Warm Film, CineStill 800T.
  - **Movies:** Teal & Orange, Blue Hour, Matrix, Storybook, Thriller.
  - **Retro:** 70s, Polaroid, 80s, Cross Process, Sepia.
  - **Night:** Neon, Cyberpunk, Midnight.
  - **Mood:** Moody, Emerald, Bleach Bypass, Nordic.
  - **Soft:** Airy, Pastel, Dreamy.
  - **B&W:** Noir, Soft B&W, Silver.

  Each one is baked into a 3D LUT that FFmpeg applies. Every preset is measured in tests: it must be clearly visible (CIEDE2000 against the original), distinct from every other preset, and keep skin tones natural on the natural looks.
- **Texture:** halation (the red film glow around highlights), bloom, grain and vignette.
- **HDR iPhone video:** HLG and Dolby Vision/PQ clips are tone-mapped to SDR, instead of looking flat and washed out.
- **Your own LUTs:** put `.cube` files in `assets/luts/` and they appear as looks.
- **Look Lab:** every look previewed on a frame of your reel, grouped by category. Click one to re-render.

## Transitions & effects

- **74 transitions in CapCut's categories:** Basic, Camera, Light, Blur, Glitch, Distortion, Slide, Mask, Overlay & social, and Trending. Examples: Mix, Black Fade, White Flash, Pull In/Out, whip pans, 3D flip, cube, light leaks, film burn, Flash Blast, chromatic blur, RGB split, liquid, fisheye, Screen Swap, split-screen push, heart, star, paper tear, film strip, Ribs Collage, zoom flash.
- **Clip effects**, synced to the music's beats: zoom pulse, shake, beat flash and RGB split; also VHS, film dust, glow and light leak.
- **Pick them yourself:** in the timeline, click a cut (▾) to choose its transition, or a shot's **fx** badge to choose its effect. Hover to preview it on your own shots; tick "use it for every cut" to apply one everywhere.

## See exactly how each reel was made

Under the video, **How this reel was made** shows the following for every version (v1, v2, …):

- **Editor:** the editor and exact model (e.g. `qwen3-vl:8b-instruct` or `gemini-4-hydrogen`), with token counts and planning time. It also lists which editors were skipped or failed first.
- **Brief and settings:** what you asked for and the options used.
- **Look and sound:** the color grade, finishing, text, captions and music.
- **Edit decision list:** each shot's source clip and in/out points, time on screen, speed, camera move, transition, reframing, sound and beat adjustment. Click a row to jump there.
- **Changes made automatically:** every fix made after the AI's plan, with the reason.

You can download the report as Markdown or JSON, and download **the AI's raw plan** to compare what the model wrote with what was rendered. The editor and model are also written into the MP4's metadata (`ffprobe -show_entries format_tags reel.mp4`).

## Quick start (macOS)

```bash
git clone <your repo url> reelsmith && cd reelsmith     # or unzip the folder
./setup.sh      # FFmpeg, Python deps, Ollama + qwen3-vl:8b-instruct (~6 GB), face model, fonts, Whisper
./run.sh        # opens http://127.0.0.1:8765. Then ⚙ Settings to add your Gemini key (optional)
```

`SKIP_OLLAMA=1 ./setup.sh` skips the local model if you only want Gemini.

## Keep it on GitHub and update from any Mac

**Run once, on the machine that has the code:**
```bash
scripts/publish_github.sh            # creates a PRIVATE repo "reelsmith" with the GitHub CLI and pushes
```
Your `.env` with your keys, your footage, your renders and the downloaded models are git-ignored, so they never get uploaded.

**On your personal Mac:**
```bash
git clone git@github.com:<you>/reelsmith.git && cd reelsmith && ./setup.sh
```

**After you push changes from anywhere:**
```bash
./update.sh      # git pull, reinstalls dependencies only if they changed, adds new .env settings
```

Every push runs the test suite on GitHub Actions (`.github/workflows/ci.yml`). It covers offline rules, the local-AI path through a fake Ollama server, mocked Gemini and Claude, and the web API, so a broken change shows up before you pull it. To run the tests locally: `pip install -r requirements-dev.txt && pytest -q`.

## How it works

```
clips ─► ANALYSE ─► PLAN (AI editor) ─► CRAFT PASS ─► TIMELINE ─► RENDER ─► reel.mp4
         │            │                   │              │            │
         │            │                   │              │            ├─ smart 9:16 crop path (faces → saliency/motion)
         │            │                   │              │            ├─ camera moves, 74 transitions, 40 looks, clip fx
         │            │                   │              │            ├─ grain, vignette, letterbox, titles, captions
         │            │                   │              │            └─ music + original sound, ducked under speech
         │            │                   │              └─ frame-exact cuts snapped to the beats
         │            │                   └─ word-safe cuts, no repeats, pacing, transition restraint, hook, duration
         │            ├─ Gemini / Claude: contact sheets + brief → full JSON plan
         │            └─ Local: vision model logs each shot (cached) → compact plan constrained by a JSON schema
         ├─ scene cuts, shot windows, sharpness/exposure/motion/face scores, contact sheets
         ├─ speech → word timestamps (faster-whisper, on-device)
         └─ music → tempo, beats, downbeats, most energetic section
```

| Module | What it does |
|---|---|
| `app/playbook.py` | Editing principles, plan schema and footage description shared by all editors |
| `app/providers/` | `gemini.py`, `ollama.py`, `claude.py`, plus the fallback chain in `__init__.py` |
| `app/craft.py` | The assistant-editor pass that enforces the playbook |
| `app/planner.py` | Orchestration, validation, the rule-based planner |
| `app/analyze.py`, `app/audio.py` | Footage, speech and music analysis |
| `app/timeline.py`, `app/render.py`, `app/effects.py`, `app/transitions.py`, `app/reframe.py` | The cinematic renderer |
| `app/main.py`, `web/` | Local web app |
| `tests/` | pytest suite, including `fake_ollama.py` |

## Command line

```bash
source .venv/bin/activate
python -m app.cli --clips trip1.mov trip2.mp4 --music song.mp3 --editor ollama \
  --prompt 'Moody cinematic Lisbon reel, open on the tram, warm film look, title "LISBON"' \
  --duration 30 --captions --out lisbon.mp4
```

Flags: `--editor auto|gemini|ollama|claude|heuristic`, `--no-beat-sync`, `--no-autopick` (use every clip in order), `--no-reframe`, `--quality preview|draft|final`, `--offline`.

## Tuning the local model (16 GB Mac)

- **Default:** `qwen3-vl:8b-instruct`, a 6.2 GB vision model that is fast on M-series chips. Use the `-instruct` tag. The plain `qwen3-vl:8b` tag is the slower *thinking* variant.
- **Faster:** set `OLLAMA_MODEL=qwen3-vl:4b-instruct` (3.3 GB). Shot logs get rougher, but it's about twice as fast.
- **Stronger, if you have the RAM:** `gemma4:12b` (~8 GB). Close other apps before using it on 16 GB.
- **Shot logs are cached.** Revisions and re-renders of the same footage skip logging. `LOCAL_MAX_SHOTS` (default 30) limits how many shots are logged per reel.

## Tips

- **Writing the brief:** name the mood, the pacing, the look, a title in quotes, and any must-have moments ("open on the tram", "end on the sunset").
- **Fast preview:** use *Fast preview* while iterating, then switch to *Final*.
- **Slow motion:** looks best on footage shot at 60 fps or higher.
- **Long videos:** paste their paths under *Advanced* so they aren't copied.
- **Manual control:** open *Edit the plan JSON*, change anything, then re-render.

## Requirements

- macOS 12 or later with Homebrew. Python 3.11–3.13, FFmpeg and Ollama are installed by `setup.sh`.
- About 8 GB of disk space with the local model, or about 1.5 GB without it.
