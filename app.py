import base64
import json
import os
import random
import ssl
import urllib.request

import anthropic
import certifi
import fal_client
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv()

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

FAL_MODEL = os.environ.get("FAL_MODEL", "fal-ai/flux/schnell")

SYSTEM_PROMPT = (
    "You are a collaborative drawing partner looking at a user's in-progress sketch. "
    "Reply in exactly this format:\n"
    "Reasoning: <1-2 sentences about what could come next in the drawing>\n"
    "Suggestion: <one short, concrete next step, e.g. 'Add a ...'>\n"
    "Be specific to what you actually see. Keep the whole reply under 60 words."
)


def split_reply(text: str) -> tuple[str, str]:
    """Split the model's 'Reasoning: ... Suggestion: ...' reply into its parts."""
    head, sep, tail = text.partition("Suggestion:")
    reasoning = head.replace("Reasoning:", "", 1).strip()
    suggestion = tail.strip() if sep else ""
    return reasoning, suggestion

app = Flask(__name__)
# ALLOWED_ORIGINS: comma-separated frontend origins (e.g. the Vercel URL); "*" if unset.
CORS(app, origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","))

# Reads ANTHROPIC_API_KEY from the environment; never hardcoded.
client = anthropic.Anthropic()


def strip_data_url(b64: str) -> str:
    """Accept either raw base64 or a full 'data:image/png;base64,...' URL."""
    if b64.startswith("data:"):
        _, _, b64 = b64.partition(",")
    return b64.strip()


@app.post("/api/reason")
def reason():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image_base64")
    context = (data.get("context") or "").strip()

    if not isinstance(image_b64, str) or not image_b64:
        return jsonify({"error": "image_base64 is required"}), 400

    user_text = "Here is the current drawing."
    if context:
        user_text += f"\n\nRecent turn history:\n{context}"
    user_text += "\n\nWhat could come next?"

    try:
        message = client.messages.create(
            model=MODEL,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": strip_data_url(image_b64),
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"Anthropic API error: {e.message}"}), 502
    except anthropic.APIConnectionError:
        return jsonify({"error": "Could not reach the Anthropic API"}), 502

    text = "".join(b.text for b in message.content if b.type == "text").strip()
    reasoning, suggestion = split_reply(text)
    return jsonify({"reasoning": reasoning, "suggestion": suggestion})


# Line-weight hint from the frontend (pen size relative to the target area).
LINE_WEIGHTS = {
    "very_thin": "very thin fine-liner strokes of uniform width",
    "thin": "thin pen strokes of uniform width",
    "medium": "medium-weight marker strokes of uniform width",
    "bold": "bold thick marker strokes of uniform width",
    "very_bold": "very thick heavy marker strokes of uniform width",
}


# Claude turns the raw instruction + what's on the canvas into a tight, detailed
# image prompt for just the content of the target area, before Fal sees it.
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-opus-5-5")
PLANNER_EFFORT = os.environ.get("PLANNER_EFFORT", "low")

PLANNER_SYSTEM = """\
You write image prompts for a collaborative drawing app in which a human and an AI take turns drawing on one canvas.

You receive the full canvas (the target area is outlined with a red dashed rectangle when only part of the canvas is being changed), a crop showing exactly what is currently drawn inside the target area, and a drawing instruction.

A text-to-image model (FLUX) will draw a clean illustration on a plain white background from your prompt, in colour or in black and white as the colour note says. That drawing is traced into pen strokes and fitted into the target area, replacing everything currently inside it. Nothing outside the area changes. Write the prompt so the generated image is exactly the new content of the target area:

1. Scope. The target area always wins over the instruction. First work out which part of the drawing the area covers (look at the crop and where the red rectangle sits on the full canvas). Then keep only the part of the instruction that applies inside that area, and drop the rest. If the area covers the body but the instruction is about the face, do not draw a face: redraw the body, with any change the instruction implies for the body. If nothing in the instruction applies to the area, redraw the area's current contents cleanly. When the instruction adds to an element that is already inside the area (for example a face on a head), redraw that element too (the head outline) together with the addition, because the area's current contents are replaced. Never depict anything outside the area: if the area holds only a head, draw only that head, with no neck, body, arms or legs; if it holds only the body, draw no head.
2. Fidelity to the canvas. Use the crop to match the existing element's shape, proportions, orientation and position within the area, and call it what the canvas shows (a stick figure's round head is a plain circle, not a potato or a character).
3. Composition. The subject fills the frame, centred, at the area's aspect ratio, with nothing cut off.
4. Only what is needed. Draw exactly the requested subject. No props, scenery, ground line, shadows, background, frame, text or decorative extras unless the instruction asks for them. Describe what to draw in positive terms and do not name objects that should be absent, because naming an object makes the image model draw it.
5. Correct basic facts. Anatomy and object structure must be right, and countable parts are stated explicitly: a hand has five fingers (four fingers and one thumb); a face has two eyes side by side above one nose and one mouth; a person has two arms and two legs; animals have their correct number of legs, eyes and ears; a bicycle has two wheels.
6. Style. Match the canvas's style and level of detail: a simple stick figure or doodle gets an equally simple drawing. Follow the style and line-weight notes provided.
7. Detail. Write 40 to 90 words of concrete descriptive prose: the subject, each of its parts, their shapes, counts and placement, the view angle, and size relations. The background is always plain white. Follow the colour note: in colour mode, name a sensible colour for every part (a yellow sun, green leaves, skin tones for faces) and ask for flat solid fills with clean dark outlines, no gradients or texture; in black-and-white mode, use black lines only.
8. The human's own words. The instruction may combine an AI suggestion with the human's own ideas (for example "add a bright day with a smiling sun"). Treat everything the human wrote as wanted content, including scenery they ask for, as long as it fits inside the target area.
"""

PLANNER_SCHEMA = {
    "type": "object",
    # Field order makes Claude ground itself in the area before writing the prompt.
    "properties": {
        "area_contents": {"type": "string", "description": "Which part of the drawing the target area covers and what is in it now."},
        "applicable_instruction": {"type": "string", "description": "Only the part of the instruction that applies inside the target area."},
        "subject": {"type": "string", "description": "A few words naming what will be drawn."},
        "image_prompt": {"type": "string", "description": "The detailed prompt for the image model."},
    },
    "required": ["area_contents", "applicable_instruction", "subject", "image_prompt"],
    "additionalProperties": False,
}


def image_block(b64: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": strip_data_url(b64)}}


def plan_reference_prompt(instruction: str, data: dict, style: dict, weight: str) -> dict:
    """Ask Claude for {subject, image_prompt}; raises on any failure."""
    content = []
    if isinstance(data.get("canvas_image_base64"), str) and data["canvas_image_base64"]:
        content.append({"type": "text", "text": "Full canvas:"})
        content.append(image_block(data["canvas_image_base64"]))
    if isinstance(data.get("region_image_base64"), str) and data["region_image_base64"]:
        content.append({"type": "text", "text": "Current contents of the target area (this will be replaced):"})
        content.append(image_block(data["region_image_base64"]))

    area = data.get("area") if isinstance(data.get("area"), dict) else {}
    aspect = data.get("aspect") if isinstance(data.get("aspect"), (int, float)) else 1.0
    target = (
        f"the red-outlined area: {area.get('w')}x{area.get('h')} px with its top-left corner at "
        f"({area.get('x')}, {area.get('y')}) on a {area.get('canvas_w')}x{area.get('canvas_h')} px canvas"
        if data.get("mode") == "incremental" else "the whole canvas (everything on it is replaced)"
    )
    context = data.get("context") if isinstance(data.get("context"), str) else ""
    content.append({"type": "text", "text": (
        f"Instruction: {instruction}\n"
        f"Target area: {target}. Aspect ratio (width/height): {aspect:.2f}.\n"
        f"Style: {style['suffix']}. Line weight: {weight}.\n"
        f"Colour: {'full colour' if data.get('color') else 'black and white'}.\n"
        + (f"Latest AI reasoning about the drawing: {context[:600]}\n" if context else "")
    )})

    response = client.beta.messages.create(
        model=PLANNER_MODEL,
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={
            "effort": PLANNER_EFFORT,
            "format": {"type": "json_schema", "schema": PLANNER_SCHEMA},
        },
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"planner stopped: {response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    plan = json.loads(text)
    if not plan.get("image_prompt", "").strip():
        raise RuntimeError("planner returned an empty prompt")
    return plan


def image_size_for(aspect: object) -> dict:
    """768 px on the long side at the target's aspect ratio (multiples of 16).
    Higher than strictly needed so thin details (net mesh, hair) survive tracing."""
    a = float(aspect) if isinstance(aspect, (int, float)) and not isinstance(aspect, bool) and aspect > 0 else 1.0
    a = max(0.25, min(4.0, a))
    if a >= 1:
        return {"width": 768, "height": max(256, round(768 / a / 16) * 16)}
    return {"width": max(256, round(768 * a / 16) * 16), "height": 768}


@app.post("/api/reference")
def reference():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip() if isinstance(data.get("prompt"), str) else ""

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if not os.environ.get("FAL_KEY"):
        return jsonify({"error": "FAL_KEY is not configured on the server"}), 500

    style = style_for(data.get("turn_count"))
    weight = LINE_WEIGHTS.get(data.get("line_weight"), LINE_WEIGHTS["medium"])
    plan = {}
    try:
        plan = plan_reference_prompt(prompt[:600], data, style, weight)
        subject, image_prompt = plan["subject"], plan["image_prompt"].strip()
    except Exception:
        # Still usable without the planner, just less precise.
        app.logger.warning("prompt planner failed; using template prompt", exc_info=True)
        subject, image_prompt = prompt[:80], f"{prompt[:400]}, isolated subject only, {style['suffix']}, {weight}"
    if data.get("color"):
        # Flat fills and clean outlines trace into strokes far better than gradients.
        finish = "Flat-colour illustration with clean dark outlines and solid fills on a plain white background, no gradients, no texture."
    else:
        finish = "Black line drawing on a plain white background, no colour, no shading fill."
    full_prompt = f"{image_prompt} {finish}"
    app.logger.info(
        "reference\n  instruction=%r\n  area_contents=%r\n  applicable=%r\n  subject=%r\n  full_prompt=%r",
        prompt, plan.get("area_contents"), plan.get("applicable_instruction"), subject, full_prompt,
    )

    try:
        # Small output + few steps with FLUX schnell keeps latency low for the demo.
        result = fal_client.subscribe(
            FAL_MODEL,
            arguments={
                "prompt": full_prompt,
                "image_size": image_size_for(data.get("aspect")),
                "num_inference_steps": 4,
                "num_images": 1,
                "enable_safety_checker": True,
            },
        )
        image_url = result["images"][0]["url"]
    except (KeyError, IndexError, TypeError):
        return jsonify({"error": "Unexpected response from image service"}), 502
    except Exception as e:  # fal_client raises a variety of HTTP/network errors
        return jsonify({"error": f"Image generation failed: {e}"}), 502

    return jsonify({
        "image_url": image_url,
        "image_data_uri": fetch_data_uri(image_url),
        "subject": subject,
        "image_prompt": full_prompt,
    })


PROMPT_TEMPLATE = (
    "{user_intent}, isolated subject only, no additional background elements, "
    "no extra figures, no hands, no text, no watermark, no border"
)
NEGATIVE_PROMPT = (
    "extra fingers, mutated hands, extra limbs, blurry, watermark, text, signature, "
    "frame, border, multiple subjects, photo-realistic skin, human hand drawing, "
    "pencil holding hand"
)
DRAW_MODES = ("overwrite", "incremental")
INPAINT_MODEL = os.environ.get("FAL_INPAINT_MODEL", "fal-ai/fast-sdxl/inpainting")
OVERWRITE_MODEL = os.environ.get("FAL_OVERWRITE_MODEL", "fal-ai/fast-sdxl")
# Style tiers by completed-turn count. Edit this table to tune live.
# (highest turn_count in the tier, or None for "and up", name, prompt suffix, strength, guidance_scale)
STYLE_TIERS = [
    (5, "sketch", "rough pencil sketch, loose gestural lines, minimal detail, unfinished", 0.5, 5.0),
    (15, "line", "line drawing, moderate detail, light shading", 0.65, 7.0),
    (None, "detailed", "detailed illustration, refined linework, shading and texture", 0.8, 8.5),
]


def style_for(turn_count: object) -> dict:
    """Pick the style tier for a turn count; bad or negative values count as 0."""
    n = turn_count if isinstance(turn_count, int) and not isinstance(turn_count, bool) else 0
    n = max(0, n)
    for max_turn, name, suffix, strength, guidance in STYLE_TIERS:
        if max_turn is None or n <= max_turn:
            return {"tier": name, "suffix": suffix, "strength": strength, "guidance_scale": guidance}
    raise AssertionError("STYLE_TIERS must end with a catch-all (None) tier")
INFERENCE_STEPS = 25
MAX_IMAGE_B64_CHARS = 12_000_000  # ~9 MB decoded
MAX_FETCH_BYTES = 10_000_000
MAX_DRAW_ATTEMPTS = int(os.environ.get("MAX_DRAW_ATTEMPTS", "3"))
PNG_MAGIC_B64 ="iVBORw0KGgo"  # base64 of the PNG file signature


def build_prompt(user_intent: str, style: dict) -> str:
    """The only place a /api/draw prompt is made; raw user text is always wrapped."""
    return f"{PROMPT_TEMPLATE.format(user_intent=user_intent)}, {style['suffix']}"


def png_data_uri(b64: object, field: str) -> str:
    """Validate a base64 PNG (raw or data URL) and return it as a data URI for Fal."""
    if not isinstance(b64, str) or not b64:
        raise ValueError(f"{field} is required for incremental mode")
    b64 = strip_data_url(b64)
    if len(b64) > MAX_IMAGE_B64_CHARS:
        raise ValueError(f"{field} is too large")
    if not b64.startswith(PNG_MAGIC_B64):
        raise ValueError(f"{field} must be a base64-encoded PNG")
    return f"data:image/png;base64,{b64}"


def fetch_data_uri(url: str) -> str | None:
    """Download a generated image and return it as a data URI (None on failure)."""
    if not url.startswith("https://"):
        return None
    try:
        # certifi's CA bundle: python.org builds on macOS ship without system certs.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, timeout=20, context=ctx) as resp:
            body = resp.read(MAX_FETCH_BYTES + 1)
            ctype = resp.headers.get_content_type()
    except Exception:
        app.logger.warning("could not fetch generated image for data URI", exc_info=True)
        return None
    if len(body) > MAX_FETCH_BYTES:
        return None
    if not ctype.startswith("image/"):
        ctype = "image/png"
    return f"data:{ctype};base64,{base64.b64encode(body).decode()}"


def clamp_dim(value: object, default: int = 512) -> int:
    """Image side for overwrite mode: multiple of 8, between 256 and 1024."""
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return max(256, min(1024, value // 8 * 8))


@app.post("/api/draw")
def draw():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    raw_intent = data.get("user_intent")
    user_intent = raw_intent.strip()[:300] if isinstance(raw_intent, str) else ""

    if mode not in DRAW_MODES:
        return jsonify({"error": f"mode must be one of {list(DRAW_MODES)}"}), 400
    if not user_intent:
        return jsonify({"error": "user_intent is required"}), 400
    if not os.environ.get("FAL_KEY"):
        return jsonify({"error": "FAL_KEY is not configured on the server"}), 500

    style = style_for(data.get("turn_count"))
    constructed_prompt = build_prompt(user_intent, style)

    if mode == "overwrite":
        model = OVERWRITE_MODEL
        arguments = {
            "prompt": constructed_prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "image_size": {
                "width": clamp_dim(data.get("width")),
                "height": clamp_dim(data.get("height")),
            },
            "num_inference_steps": INFERENCE_STEPS,
            "guidance_scale": style["guidance_scale"],
            "num_images": 1,
            "enable_safety_checker": True,
        }
    else:
        try:
            image_uri = png_data_uri(data.get("canvas_image_base64"), "canvas_image_base64")
            mask_uri = png_data_uri(data.get("mask_base64"), "mask_base64")
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        model = INPAINT_MODEL
        arguments = {
            "image_url": image_uri,
            "mask_url": mask_uri,  # white = editable, black = protected
            "prompt": constructed_prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "strength": style["strength"],
            "guidance_scale": style["guidance_scale"],
            "num_inference_steps": INFERENCE_STEPS,
            "num_images": 1,
            "enable_safety_checker": True,
        }

    app.logger.info(
        "draw mode=%s model=%s tier=%s strength=%s guidance=%s\n  user_intent=%r\n  constructed_prompt=%r",
        mode, model, style["tier"], style["strength"], style["guidance_scale"],
        user_intent, constructed_prompt,
    )

    try:
        image_url = None
        for attempt in range(1, MAX_DRAW_ATTEMPTS + 1):
            result = fal_client.subscribe(
                model, arguments={**arguments, "seed": random.randrange(2**31)}
            )
            image_url = result["images"][0]["url"]
            flagged = bool((result.get("has_nsfw_concepts") or [False])[0])
            if not flagged:
                break
            # Fal's safety checker swaps the image for solid black. It false-positives
            # often on line sketches, and depends on the seed, so retry with a new one.
            app.logger.warning("draw attempt %d/%d blanked by safety checker", attempt, MAX_DRAW_ATTEMPTS)
            image_url = None
    except (KeyError, IndexError, TypeError):
        return jsonify({"error": "Unexpected response from image service"}), 502
    except Exception as e:
        app.logger.exception("fal call failed")
        return jsonify({"error": f"Image generation failed: {e}"}), 502

    if image_url is None:
        return jsonify({
            "error": f"The image model's safety filter blanked all {MAX_DRAW_ATTEMPTS} attempts "
                     "(it often false-positives on sketches). Try again or reword the instruction."
        }), 502

    return jsonify({
        "mode": mode,
        "image_url": image_url,
        # Same-origin copy so the frontend can draw it onto the canvas without
        # tainting it (which would block later toDataURL exports).
        "image_data_uri": fetch_data_uri(image_url),
        "user_intent": user_intent,
        "constructed_prompt": constructed_prompt,
        "style_tier": style["tier"],
        "strength": style["strength"],
        "guidance_scale": style["guidance_scale"],
        "negative_prompt": NEGATIVE_PROMPT,
    })


# Tutorial mode ("Teach me"): the browser splits the traced reference into ordered
# steps; Claude only writes one beginner-friendly instruction per step.
TUTORIAL_EFFORT = os.environ.get("TUTORIAL_EFFORT", "low")
MAX_TUTORIAL_STEPS = 6

TUTORIAL_SYSTEM = """\
You are a patient drawing teacher in a collaborative drawing app. The human draws everything themselves by tracing a light blue guide that appears on their canvas one step at a time.

You receive the current canvas (the target area is outlined in red when only part of the canvas is changing), the reference image the guide was traced from, the human's goal, and a preview of each step: earlier steps in grey, the strokes for that step in black.

Write one instruction per step, in the same order, for a beginner:
- Exactly one plain-language sentence (never two), at most 25 words, saying what to draw and where (for example "Draw a tall oval for the trunk, slightly left of center").
- Each step is one part of the drawing, in drawing order: "first" is the biggest part, "part" is the next part (usually one that joins what is already drawn), "details" collects the small finishing touches, "fill" is colouring in. Name the part (for example "the trunk") and say where it sits relative to parts already drawn.
- Describe only what is new in that step's black strokes, using simple shapes, positions and sizes. Name colours only for colour or fill steps.
- Use encouraging, concrete words; no jargon, no art-school terms.
- The title is 2 to 4 words naming the step (for example "Trunk outline").
Return exactly as many steps as you were given."""

TUTORIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "instruction": {"type": "string"},
                },
                "required": ["title", "instruction"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}


@app.post("/api/tutorial")
def tutorial():
    data = request.get_json(silent=True) or {}
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return jsonify({"error": "steps is required"}), 400
    steps = [s for s in steps if isinstance(s, dict)][:MAX_TUTORIAL_STEPS]
    if not steps:
        return jsonify({"error": "steps is required"}), 400

    instruction = data.get("instruction") if isinstance(data.get("instruction"), str) else ""
    content = []
    for label, field in (("Current canvas:", "canvas_image_base64"), ("Reference image the guide was traced from:", "reference_image_base64")):
        if isinstance(data.get(field), str) and data[field]:
            content.append({"type": "text", "text": label})
            content.append(image_block(data[field]))
    for n, step in enumerate(steps, 1):
        content.append({"type": "text", "text": (
            f"Step {n} of {len(steps)} ({step.get('kind') or 'strokes'}, "
            f"{step.get('stroke_count') or 'some'} strokes). Preview:"
        )})
        if isinstance(step.get("preview_base64"), str) and step["preview_base64"]:
            content.append(image_block(step["preview_base64"]))
    area_note = (
        "only the red-outlined area of the canvas" if data.get("mode") == "incremental" else "the whole canvas"
    )
    content.append({"type": "text", "text": (
        f"Goal: {instruction[:600] or 'draw the reference image'}\n"
        f"The guide covers {area_note}. Write {len(steps)} steps."
    )})

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={
                "effort": TUTORIAL_EFFORT,
                "format": {"type": "json_schema", "schema": TUTORIAL_SCHEMA},
            },
            system=TUTORIAL_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIStatusError as e:
        return jsonify({"error": f"Anthropic API error: {e.message}"}), 502
    except anthropic.APIConnectionError:
        return jsonify({"error": "Could not reach the Anthropic API"}), 502

    if response.stop_reason in ("refusal", "max_tokens"):
        return jsonify({"error": f"Tutorial text stopped early: {response.stop_reason}"}), 502
    try:
        text = next(b.text for b in response.content if b.type == "text")
        written = json.loads(text)["steps"]
    except (StopIteration, ValueError, KeyError, TypeError):
        return jsonify({"error": "Unexpected response from the model"}), 502

    # Exactly one entry per step sent: pad with generic text, drop extras.
    out = []
    for n in range(len(steps)):
        w = written[n] if n < len(written) and isinstance(written[n], dict) else {}
        out.append({
            "title": str(w.get("title") or f"Step {n + 1} of {len(steps)}").strip()[:60],
            "instruction": str(w.get("instruction") or "Trace the blue guide lines.").strip()[:300],
        })
    app.logger.info("tutorial\n  goal=%r\n  steps=%r", instruction, out)
    return jsonify({"steps": out})

if __name__ == "__main__":
    # Hosts like Render set PORT and need 0.0.0.0; locally use 127.0.0.1:5001
    # (not 5000: macOS AirPlay Receiver occupies it).
    port = os.environ.get("PORT")
    app.run(host="0.0.0.0" if port else "127.0.0.1", port=int(port or 5001), debug=not port)
