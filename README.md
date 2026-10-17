# AI Drawing Partner

A turn-based drawing canvas where a human and an AI draw together. The human sketches; Claude looks at the canvas and suggests what could come next; an image model (FLUX via Fal) generates a reference for that suggestion; and the app traces the reference into ordinary, editable pen strokes, placed into a region you mark or across the whole canvas.

Built for the DiversaTech hackathon, 17 October 2026.

Contributors:
- Darrin
- Gabriel
- Jonothan
- PokPong
- Claude

## How it works

```
 index.html (browser)                         app.py (Flask, :5001)
 ───────────────────                          ─────────────────────
 Draw strokes ──── Ask AI ──────────────────▶ POST /api/reason
                                                Claude (vision) reads the canvas
 Instruction box ◀─ reasoning + suggestion ◀─── returns reasoning + one suggestion

 Mark region ───── Get reference image ─────▶ POST /api/reference
 (+ edit the        canvas, region crop,        1. Claude (prompt planner) writes a tight
  instruction)      instruction, colour flag       image prompt for only that region
                                                2. FLUX (Fal) generates the image
 Reference shown ◀─ image (data URI) ◀───────── returns image + prompt used

 AI draw ──────── traced in the browser into strokes, replacing what was in the region
```

1. **Ask AI.** The canvas is sent to Claude, which replies with 1–2 sentences of reasoning and one concrete suggestion. The suggestion fills the instruction box, where you can edit it or add your own ideas (for example "…add a bright day with a smiling sun").
2. **Get reference image.** A planner call to Claude sees the full canvas (target region outlined in red), a crop of the region, and your instruction. It works out what the region covers, keeps only the part of the instruction that applies there, and writes a detailed image prompt with explicit part counts (five fingers, two eyes, and so on). FLUX then generates the image at the region's aspect ratio.
3. **AI draw.** The reference is traced into strokes in the browser:
   - The image is split into up to 8 colours.
   - Each colour is thinned to centrelines.
   - Each centreline is drawn with the line width measured at that point in the image, so thick and thin parts and solid fills are kept.
   - Lighter colours go underneath and dark outlines on top.
   - The traced strokes replace whatever was drawn inside the region; lines crossing the region's edge are clipped.

Every action is recorded in the **turn log** (proposer, executor, text, canvas thumbnail).

## Features

- **Canvas tools:**
  - Pen with adjustable size and colour.
  - Eraser that removes pixels, so it can erase part of a stroke.
  - Select / move: click to select a whole AI drawing; drag a box to select exactly what is inside it (strokes are split at the box edge); Delete removes the selection; Escape deselects.
  - Mark region, which sets the target area for incremental AI drawing.
  - Undo for every action (Cmd/Ctrl+Z), and Clear (which can also be undone).
- **AI draw modes:** *Incremental* replaces only the marked region. *Overwrite* replaces the whole canvas.
- **Colour or black-and-white** references (Colour checkbox).
- **Style tiers** by turn count: rough sketch for turns 0–5, line drawing for 6–15, detailed illustration from 16 on.
- **Guard rails:**
  - In Incremental mode, Get reference image needs a marked region.
  - AI draw needs a reference image.
  - AI draw is single-use until you Ask AI again or get a new reference.

## Setup

Requirements: Python 3.10+, an [Anthropic API key](https://console.anthropic.com/) and a [Fal API key](https://fal.ai/dashboard/keys) with credit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # then put your real keys in .env
```

`.env` is gitignored. Never commit real keys, and never put them in `.env.example`.

## Running

```bash
python app.py           # backend on http://127.0.0.1:5001
```

Then open `index.html` directly in a browser (double-click it). The backend only serves the API, so visiting `http://127.0.0.1:5001/` returns a 404; that is expected.

The backend runs on port **5001**, not 5000, because macOS AirPlay Receiver occupies port 5000.

## Configuration

All settings are environment variables, read from `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Claude API key |
| `FAL_KEY` | required | Fal API key for image generation |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Model for Ask AI reasoning |
| `PLANNER_MODEL` | `claude-opus-5-5` | Model that writes the image prompt |
| `PLANNER_EFFORT` | `low` | Planner effort (`low` to `max`); higher is slower and more careful |
| `FAL_MODEL` | `fal-ai/flux/schnell` | Text-to-image model for references |

Tracing constants (`TRACE_MAX_SIDE`, `MAX_COLORS`, `BG_LIGHTNESS`, …) are at the top of the tracing section in `index.html`. Style tiers (`STYLE_TIERS`) and line-weight phrases (`LINE_WEIGHTS`) are at the top of `app.py`.

## API

### `POST /api/reason`
Request: `{ "image_base64": "<png or data URL>", "context": "<recent reasoning>" }`
Response: `{ "reasoning": "...", "suggestion": "..." }`

### `POST /api/reference`
Request:

```json
{
  "prompt": "instruction text",
  "turn_count": 7,
  "line_weight": "thin",
  "color": true,
  "mode": "incremental",
  "aspect": 0.6,
  "area": { "x": 150, "y": 105, "w": 100, "h": 165, "canvas_w": 1000, "canvas_h": 850 },
  "canvas_image_base64": "<png>",
  "region_image_base64": "<png>",
  "context": "latest AI reasoning"
}
```

Response: `{ "image_url": "...", "image_data_uri": "data:...", "subject": "...", "image_prompt": "..." }`

`image_data_uri` is a copy downloaded by the server, so the browser can trace it without cross-origin canvas errors. If the planner call fails, the endpoint falls back to a template prompt instead of erroring.

### `POST /api/draw` (unused by the frontend)
This is an earlier SDXL inpainting endpoint, kept for reference. Its safety checker false-positives on line sketches and returns black images, which is why AI draw now traces the reference instead.

## Debugging

- **Flask terminal:** for every reference it logs the instruction, what the planner thinks the region contains, the part of the instruction it applied, and the final FLUX prompt.
- **Browser console:** `console.debug` shows the subject and image prompt for each reference.
- **"Could not reach the AI server":** the backend isn't running, or isn't on port 5001.
- **`User is locked. Reason: Exhausted balance`:** the Fal account needs credit.
- **`CERTIFICATE_VERIFY_FAILED`:** use the `certifi` package already in `requirements.txt` (re-run `pip install -r requirements.txt`).

## Known limitations

- **Region scope:** in Incremental mode, anything you ask for that falls outside the marked region is dropped by design. Use Overwrite for whole-scene changes.
- **Gradients:** gradients and heavy shading trace as bands of flat colour. The prompt asks FLUX for flat fills to avoid this.
- **Busy images:** busy references (nets, foliage) can produce over a thousand strokes. If tracing or dragging feels slow, lower `TRACE_MAX_SIDE`.
- **Eraser and moves:** eraser marks are fixed in place, so moving a stroke you partly erased brings the erased part back at its new position.
- **Turn log thumbnails:** they link to Fal-hosted image URLs, which expire after a while.

## Project structure

```
app.py            Flask backend: /api/reason, /api/reference (+ legacy /api/draw)
index.html        Entire frontend: canvas, tools, tracing, AI panels, turn log (no build step)
requirements.txt  Python dependencies
.env.example      Template for .env (placeholders only)
```
