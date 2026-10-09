#!/usr/bin/env python3
"""
compare_lora_epochs.py

Generate the same prompts and seed with every LoRA epoch (and optionally several
strengths) through a running ComfyUI server, then stitch the results into
comparison sheets. Standard library only; Pillow is needed only for the sheets.

    python compare_lora_epochs.py --list-ckpts
    python compare_lora_epochs.py --list
    python compare_lora_epochs.py --ckpt MODEL.safetensors --match my_lora --dry-run
    python compare_lora_epochs.py --ckpt MODEL.safetensors --match my_lora --trigger mychar
"""
import argparse
import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime

# Default test prompts: (tags, width, height). Edit freely or use --prompts-file.
DEFAULT_PROMPTS = [
    ("upper body, looking at viewer, smile, simple background", 896, 1152),
    ("cowboy shot, outdoors, blue sky", 832, 1216),
    ("portrait, face focus, close-up, indoors", 1024, 1024),
]
DEFAULT_QUALITY = "masterpiece, best quality"
DEFAULT_NEGATIVE = "worst quality, low quality, lowres, bad anatomy, bad hands, watermark, text"


# ---------------------------------------------------------------- ComfyUI API
def http_json(url, data=None):
    req = urllib.request.Request(
        url, data=None if data is None else json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def object_choices(server, node, field):
    info = http_json(f"http://{server}/object_info/{node}")
    return info[node]["input"]["required"][field][0]


def fetch_image(server, meta):
    q = urllib.parse.urlencode({"filename": meta["filename"],
                                "subfolder": meta.get("subfolder", ""),
                                "type": meta.get("type", "output")})
    with urllib.request.urlopen(f"http://{server}/view?{q}") as r:
        return r.read()


def wait_all(server, pids, timeout):
    done, t0 = {}, time.time()
    while len(done) < len(pids) and time.time() - t0 < timeout:
        for pid in pids:
            if pid in done:
                continue
            h = http_json(f"http://{server}/history/{pid}")
            if pid in h:
                done[pid] = h[pid]
        print(f"\rwaiting {len(done)}/{len(pids)}", end="", flush=True)
        time.sleep(2)
    print()
    return done


def build_graph(ckpt, lora, strength, positive, negative, seed, w, h,
                steps, cfg, sampler, scheduler, clip_skip, prefix):
    """checkpoint -> [LoRA] -> CLIP skip -> KSampler -> VAE decode -> save"""
    g = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    model, clip = ["1", 0], ["1", 1]
    if lora:
        g["2"] = {"class_type": "LoraLoader",
                  "inputs": {"model": model, "clip": clip, "lora_name": lora,
                             "strength_model": strength, "strength_clip": strength}}
        model, clip = ["2", 0], ["2", 1]
    g["3"] = {"class_type": "CLIPSetLastLayer",
              "inputs": {"clip": clip, "stop_at_clip_layer": -abs(clip_skip)}}
    g["4"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": positive}}
    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": negative}}
    g["6"] = {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}}
    g["7"] = {"class_type": "KSampler",
              "inputs": {"model": model, "positive": ["4", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}}
    g["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["1", 2]}}
    g["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return g


# ---------------------------------------------------------------- sheet
def make_grid(rows, col_titles, thumb_w=448):
    """rows: [(row label, [PNG bytes or None, ...]), ...]. Every cell has the same size."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    label_w, title_h, gap = 260, 34, 4

    decoded = []
    for _, imgs in rows:
        r = []
        for b in imgs:
            if b is None:
                r.append(None)
                continue
            im = Image.open(io.BytesIO(b)).convert("RGB")
            r.append(im.resize((thumb_w, max(1, int(im.height * thumb_w / im.width)))))
        decoded.append(r)
    cell_h = max((im.height for r in decoded for im in r if im), default=thumb_w)

    W = label_w + (thumb_w + gap) * len(col_titles)
    H = title_h + (cell_h + gap) * len(rows)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)

    def clip_text(text, width):
        if d.textlength(text, font=font) <= width:
            return text
        while text and d.textlength(text + "...", font=font) > width:
            text = text[:-1]
        return text + "..."

    for c, t in enumerate(col_titles):
        d.text((label_w + c * (thumb_w + gap) + 4, 6),
               clip_text(f"[{c + 1}] {t}", thumb_w - 8), fill=(0, 0, 0), font=font)
    for ri, ((label, _), r) in enumerate(zip(rows, decoded)):
        y0 = title_h + ri * (cell_h + gap)
        d.text((6, y0 + 6), clip_text(label, label_w - 12), fill=(0, 0, 0), font=font)
        for c, im in enumerate(r):
            x0 = label_w + c * (thumb_w + gap)
            d.rectangle([x0, y0, x0 + thumb_w - 1, y0 + cell_h - 1], fill=(235, 235, 235))
            if im:
                sheet.paste(im, (x0, y0 + (cell_h - im.height) // 2))
            else:
                d.text((x0 + 8, y0 + 8), "(failed)", fill=(200, 0, 0), font=font)
    return sheet


# ---------------------------------------------------------------- main
def load_prompts(path):
    """One prompt per line. Optional size: 'tags | 832x1216' (default 832x1216)."""
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            tags, w, h = ln, 832, 1216
            if "|" in ln:
                tags, size = [x.strip() for x in ln.rsplit("|", 1)]
                w, h = (int(v) for v in size.lower().split("x"))
            out.append((tags, w, h))
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare LoRA epochs/strengths on a running ComfyUI server.")
    ap.add_argument("--server", default="127.0.0.1:8188", help="ComfyUI address (default: 127.0.0.1:8188)")
    ap.add_argument("--list", action="store_true", help="print LoRA names ComfyUI can see, then exit")
    ap.add_argument("--list-ckpts", action="store_true", help="print checkpoint names ComfyUI can see, then exit")
    ap.add_argument("--ckpt", help="checkpoint name exactly as shown by --list-ckpts")
    ap.add_argument("--match", nargs="*", default=[],
                    help="only LoRAs whose name contains any of these strings (default: all LoRAs)")
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.8], help="LoRA strengths (default: 0.8)")
    ap.add_argument("--trigger", default="", help="trigger word added to LoRA rows")
    ap.add_argument("--tags", default="", help="tags added to every LoRA row (e.g. clothing tags not baked into the trigger)")
    ap.add_argument("--baseline-tags", default="", help="tags for the no-LoRA baseline row")
    ap.add_argument("--no-baseline", action="store_true", help="skip the no-LoRA baseline row")
    ap.add_argument("--quality", default=DEFAULT_QUALITY, help="quality tags put first in every prompt")
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--prompts-file", help="text file, one prompt per line ('tags | WxH' optional)")
    ap.add_argument("--seed", type=int, default=12345, help="fixed seed shared by all images")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--sampler", default="euler_ancestral")
    ap.add_argument("--scheduler", default="normal")
    ap.add_argument("--clip-skip", type=int, default=2)
    ap.add_argument("--per-sheet", type=int, default=5, help="LoRA rows per sheet (default: 5)")
    ap.add_argument("--thumb", type=int, default=448, help="thumbnail width in the sheet")
    ap.add_argument("--save-prefix", default="lora_compare", help="ComfyUI output sub-folder / filename prefix")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds to wait for generation")
    ap.add_argument("--dry-run", action="store_true", help="show what would run, generate nothing")
    args = ap.parse_args()

    if args.list_ckpts:
        print("\n".join(object_choices(args.server, "CheckpointLoaderSimple", "ckpt_name")))
        return
    loras = object_choices(args.server, "LoraLoader", "lora_name")
    if args.list:
        print("\n".join(loras))
        return
    if not args.ckpt:
        raise SystemExit("--ckpt is required (see --list-ckpts)")

    targets = sorted(l for l in loras if not args.match or any(m in l for m in args.match))
    if not targets:
        raise SystemExit("No matching LoRA. Check names with --list.")
    prompts = load_prompts(args.prompts_file) if args.prompts_file else DEFAULT_PROMPTS

    base = [] if args.no_baseline else [("NO LoRA", None, 0.0)]
    conds = []
    for l in targets:
        short = l.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
        for s in args.strengths:
            conds.append((f"{short} @{s}", l, s))
    allc = base + conds
    print(f"{len(targets)} LoRA x {len(args.strengths)} strength x {len(prompts)} prompt"
          f" (+{len(base)} baseline row) = {len(allc) * len(prompts)} images")
    if args.dry_run:
        for label, _, _ in allc:
            print(" ", label)
        return

    client_id = str(uuid.uuid4())
    jobs = []  # (row, col, prompt_id)
    for ri, (label, lora, s) in enumerate(allc):
        for ci, (tags, w, h) in enumerate(prompts):
            if lora:
                parts = [args.quality, args.trigger, args.tags, tags]
            else:
                parts = [args.quality, args.baseline_tags, tags]
            positive = ", ".join(p for p in parts if p)
            stem = (lora.rsplit(".", 1)[0] if lora else "baseline").replace("/", "_").replace("\\", "_")
            graph = build_graph(args.ckpt, lora, s, positive, args.negative, args.seed, w, h,
                                args.steps, args.cfg, args.sampler, args.scheduler, args.clip_skip,
                                f"{args.save_prefix}/{stem}_s{s}_p{ci}")
            res = http_json(f"http://{args.server}/prompt", {"prompt": graph, "client_id": client_id})
            jobs.append((ri, ci, res["prompt_id"]))

    hist = wait_all(args.server, [j[2] for j in jobs], args.timeout)

    rows = [(label, [None] * len(prompts)) for label, _, _ in allc]
    for ri, ci, pid in jobs:
        out = hist.get(pid, {}).get("outputs", {}).get("9", {}).get("images")
        if out:
            rows[ri][1][ci] = fetch_image(args.server, out[0])

    try:
        head, body = rows[:len(base)], rows[len(base):]
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        for k, i in enumerate(range(0, len(body), args.per_sheet), 1):
            sheet = make_grid(head + body[i:i + args.per_sheet], [p[0] for p in prompts], args.thumb)
            name = f"compare_{stamp}_{k}.png"
            sheet.save(name)
            print("saved", name)
    except ImportError:
        print("Pillow is not installed, so no sheets were made (pip install pillow). "
              f"Individual images are in ComfyUI's output/{args.save_prefix}/.")


if __name__ == "__main__":
    main()
