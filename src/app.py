"""
Wind turbine detector - FastAPI + YOLO (Ultralytics) + single-file frontend.

Azure Container Apps version:
  - The model is NOT bundled in the repo. src/download_model.py pulls it from
    Azure Blob Storage into model/best.pt before this app starts.
  - Started by the Dockerfile with: uvicorn src.app:app --host 0.0.0.0 --port 8000

Pylon detections are ignored, so only wind turbines show up in the boxes,
the counts, the ZIP results and the CSV files.

Two ways to use it:
  - Single image:  POST /predict      (returns detections as JSON)
  - ZIP of images: POST /predict_zip  (starts a background job, then poll
                   GET /jobs/{id} and fetch GET /jobs/{id}/download)
"""

import csv
import io
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import PurePosixPath

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO

# download_model.py saves the blob here (relative to /app, the container workdir).
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")

# ZIP / bulk limits (override with environment variables if needed).
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MAX_ZIP_MB = int(os.getenv("MAX_ZIP_MB", "200"))
MAX_ZIP_IMAGES = int(os.getenv("MAX_ZIP_IMAGES", "300"))
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "50"))
# Detections whose class name contains this text are ignored (case-insensitive).
IGNORE_LABEL = "pylon"
JOB_TTL_SECONDS = 3600
THUMB_SIZE = (800, 800)

BOX_COLOR = (255, 193, 94)
LABEL_TEXT_COLOR = (28, 19, 5)
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
)

state = {}
model_lock = threading.Lock()  # the model runs one prediction at a time
jobs = {}  # job_id -> job dict (in memory; lives for JOB_TTL_SECONDS)
jobs_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Check that src/download_model.py ran and the storage secret is set."
        )
    state["model"] = YOLO(MODEL_PATH)
    print("Model ready. Classes:", state["model"].names)
    yield
    state.clear()


app = FastAPI(title="Turbine Spotter", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/info")
def info():
    model = state["model"]
    return {
        "classes": [n for n in model.names.values() if IGNORE_LABEL not in n.lower()],
        "model_path": MODEL_PATH,
        "zip_limits": {"max_zip_mb": MAX_ZIP_MB, "max_images": MAX_ZIP_IMAGES},
    }


# ---------------------------------------------------------------- detection


def run_detection(image, conf, iou, imgsz):
    """Run the model on one PIL image. Returns (detections, inference_ms)."""
    model = state["model"]
    with model_lock:
        start = time.perf_counter()
        result = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    detections = []
    for xyxy, score, cls in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
        strict=True,
    ):
        label = model.names[int(cls)]
        if IGNORE_LABEL in label.lower():
            continue  # pylon: not needed, skip it
        detections.append(
            {
                "label": label,
                "confidence": round(score, 4),
                "box": [round(v, 1) for v in xyxy],
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, elapsed_ms


# Plain `def` so FastAPI runs inference in a worker thread and the server stays responsive.
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    imgsz: int = Form(640),
):
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400, detail="That file is not a readable image."
        ) from exc

    detections, elapsed_ms = run_detection(image, conf, iou, imgsz)
    return {
        "width": image.width,
        "height": image.height,
        "inference_ms": elapsed_ms,
        "count": len(detections),
        "detections": detections,
    }


# ------------------------------------------------------------- ZIP / bulk


def load_font(size):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(image, detections):
    """Return a copy of the image with boxes and labels drawn on it."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    line_w = max(2, round(max(out.size) / 350))
    font = load_font(max(13, line_w * 6))
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_w)
        text = f"{d['label']} {round(d['confidence'] * 100)}%"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        box_w, box_h = right - left + 10, bottom - top + 8
        ty = y1 - box_h if y1 - box_h >= 0 else y1
        draw.rectangle([x1, ty, x1 + box_w, ty + box_h], fill=BOX_COLOR)
        draw.text((x1 + 5, ty + 4 - top), text, fill=LABEL_TEXT_COLOR, font=font)
    return out


def list_image_entries(zf):
    """Names of the image files inside the ZIP (skips folders and hidden/Mac junk)."""
    names = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename.replace("\\", "/")).parts
        if not parts or any(p.startswith(".") or p == "__MACOSX" for p in parts):
            continue
        if PurePosixPath(parts[-1]).suffix.lower() in IMAGE_EXTS:
            names.append(info.filename)
    return names


def output_name(idx, name):
    stem = PurePosixPath(name.replace("\\", "/")).stem
    return f"annotated/{idx + 1:04d}_{stem}.jpg"


def jpeg_bytes(image, quality):
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def process_zip_entry(zf, idx, name, params):
    """Detect on one image from the ZIP. Returns (row, full_jpeg, thumb_jpeg)."""
    row = {
        "index": idx,
        "name": name,
        "count": 0,
        "max_confidence": 0,
        "inference_ms": 0,
        "error": None,
        "detections": [],
    }
    if zf.getinfo(name).file_size > MAX_IMAGE_MB * 1024 * 1024:
        row["error"] = f"Larger than {MAX_IMAGE_MB} MB"
        return row, None, None
    try:
        with zf.open(name) as fh:
            image = Image.open(io.BytesIO(fh.read())).convert("RGB")
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        RuntimeError,
        zipfile.BadZipFile,
        Image.DecompressionBombError,
    ):
        row["error"] = "Not a readable image"
        return row, None, None

    detections, elapsed_ms = run_detection(image, **params)
    annotated = annotate(image, detections)
    thumb = annotated.copy()
    thumb.thumbnail(THUMB_SIZE)
    row.update(
        count=len(detections),
        max_confidence=detections[0]["confidence"] if detections else 0,
        inference_ms=elapsed_ms,
        detections=detections,
    )
    return row, jpeg_bytes(annotated, 88), jpeg_bytes(thumb, 80)


def write_csvs(out_zip, rows):
    summary = io.StringIO()
    writer = csv.writer(summary)
    writer.writerow(["file", "turbines", "max_confidence", "inference_ms", "status"])
    for r in rows:
        writer.writerow(
            [r["name"], r["count"], r["max_confidence"], r["inference_ms"], r["error"] or "ok"]
        )
    out_zip.writestr("results.csv", summary.getvalue())

    boxes = io.StringIO()
    writer = csv.writer(boxes)
    writer.writerow(["file", "label", "confidence", "x1", "y1", "x2", "y2"])
    for r in rows:
        for d in r["detections"]:
            writer.writerow([r["name"], d["label"], d["confidence"], *d["box"]])
    out_zip.writestr("detections.csv", boxes.getvalue())


def run_job(job_id):
    """Background worker: runs the model over every image in the uploaded ZIP."""
    job = jobs[job_id]
    workdir = job["workdir"]
    try:
        with (
            zipfile.ZipFile(job["zip_path"]) as zf,
            zipfile.ZipFile(
                os.path.join(workdir, "results.zip"), "w", zipfile.ZIP_STORED
            ) as out_zip,
        ):
            for idx, name in enumerate(job["names"]):
                row, full_jpeg, thumb_jpeg = process_zip_entry(
                    zf, idx, name, job["params"]
                )
                if full_jpeg is not None:
                    out_zip.writestr(output_name(idx, name), full_jpeg)
                    with open(os.path.join(workdir, f"{idx:04d}.jpg"), "wb") as fh:
                        fh.write(thumb_jpeg)
                job["rows"].append(row)
                job["done"] = idx + 1
            write_csvs(out_zip, job["rows"])
        job["status"] = "done"
    finally:
        if job["status"] == "running":
            job["status"] = "error"
            job["message"] = "Processing stopped unexpectedly. Check the server logs."
        if os.path.exists(job["zip_path"]):
            os.remove(job["zip_path"])


def cleanup_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with jobs_lock:
        stale = [
            key
            for key, job in jobs.items()
            if job["status"] != "running" and job["created"] < cutoff
        ]
        for key in stale:
            shutil.rmtree(jobs.pop(key)["workdir"], ignore_errors=True)


def get_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job


def reject(workdir, status_code, detail):
    shutil.rmtree(workdir, ignore_errors=True)
    raise HTTPException(status_code=status_code, detail=detail)


@app.post("/predict_zip")
def predict_zip(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    iou: float = Form(0.7),
    imgsz: int = Form(640),
):
    cleanup_jobs()
    workdir = tempfile.mkdtemp(prefix="job_")
    zip_path = os.path.join(workdir, "input.zip")

    size, limit = 0, MAX_ZIP_MB * 1024 * 1024
    with open(zip_path, "wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                out.close()
                reject(workdir, 413, f"The ZIP is larger than {MAX_ZIP_MB} MB.")
            out.write(chunk)

    if not zipfile.is_zipfile(zip_path):
        reject(workdir, 400, "That file is not a valid ZIP archive.")
    with zipfile.ZipFile(zip_path) as zf:
        names = list_image_entries(zf)
    if not names:
        reject(workdir, 400, "No images (jpg, png, webp, bmp, tif) found in the ZIP.")
    if len(names) > MAX_ZIP_IMAGES:
        reject(
            workdir,
            400,
            f"The ZIP has {len(names)} images; the limit is {MAX_ZIP_IMAGES} per ZIP.",
        )

    job_id = uuid.uuid4().hex[:12]
    job = {
        "status": "running",
        "message": "",
        "created": time.time(),
        "workdir": workdir,
        "zip_path": zip_path,
        "names": names,
        "params": {"conf": conf, "iou": iou, "imgsz": imgsz},
        "rows": [],
        "done": 0,
    }
    with jobs_lock:
        jobs[job_id] = job
    threading.Thread(target=run_job, args=(job_id,), daemon=True).start()
    return {"job_id": job_id, "total": len(names)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    rows = list(job["rows"])
    public = ("index", "name", "count", "max_confidence", "inference_ms", "error")
    return {
        "status": job["status"],
        "message": job["message"],
        "done": job["done"],
        "total": len(job["names"]),
        "turbines": sum(r["count"] for r in rows),
        "rows": [{k: r[k] for k in public} for r in rows],
    }


@app.get("/jobs/{job_id}/image/{idx}")
def job_image(job_id: str, idx: int):
    job = get_job(job_id)
    path = os.path.join(job["workdir"], f"{idx:04d}.jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/jobs/{job_id}/download")
def job_download(job_id: str):
    job = get_job(job_id)
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Results are not ready yet.")
    return FileResponse(
        os.path.join(job["workdir"], "results.zip"),
        media_type="application/zip",
        filename="turbine_results.zip",
    )


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Turbine Spotter</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --night: #0b1522;
    --dusk: #25465e;
    --glow: #e8a76a;
    --ridge: #08111b;
    --panel: rgba(9, 18, 30, 0.66);
    --line: rgba(255, 255, 255, 0.14);
    --ink: #eef3f7;
    --muted: #a5b6c4;
    --accent: #f2b35e;
    --box: #ffc15e;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    font-family: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink);
    background: linear-gradient(180deg, var(--night) 0%, #17314a 38%, var(--dusk) 62%, var(--glow) 100%) fixed;
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
  }

  /* ---------- background scene ---------- */
  #scene { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
  #scene svg.turbine { position: absolute; bottom: 0; overflow: visible; }
  .ridge { position: absolute; left: 0; right: 0; bottom: 0; width: 100%; height: 22vh; }
  .cloud {
    position: absolute; height: 34px; width: 220px; border-radius: 40px;
    background: rgba(255, 255, 255, 0.07); filter: blur(10px);
    animation: drift linear infinite;
  }
  @keyframes drift { from { transform: translateX(-30vw); } to { transform: translateX(130vw); } }

  /* ---------- layout ---------- */
  main { position: relative; z-index: 1; max-width: 1000px; margin: 0 auto; padding: 48px 20px 120px; }
  h1 {
    font-family: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.6rem); line-height: 1.02; margin: 0 0 10px; font-weight: 700;
    letter-spacing: -0.02em;
  }
  .lead { color: var(--muted); max-width: 52ch; margin: 0 0 30px; line-height: 1.5; font-size: 1.05rem; }

  .panel {
    background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); padding: 22px;
  }
  .grid { display: grid; grid-template-columns: 1fr 280px; gap: 18px; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }

  .stage {
    position: relative; min-height: 320px; border-radius: 12px; overflow: hidden;
    border: 1.5px dashed rgba(255, 255, 255, 0.28); display: grid; place-items: center;
    background: rgba(0, 0, 0, 0.18); cursor: pointer; transition: border-color .15s, background .15s;
  }
  .stage.drag, .stage:hover { border-color: var(--accent); background: rgba(242, 179, 94, 0.07); }
  .stage.has-image { border-style: solid; cursor: default; }
  .stage canvas { display: none; width: 100%; height: auto; max-height: 70vh; object-fit: contain; }
  .stage.has-image canvas { display: block; }
  .hint { text-align: center; color: var(--muted); padding: 24px; line-height: 1.5; }
  .hint strong { color: var(--ink); font-weight: 600; }
  .stage.has-image .hint { display: none; }
  .busy {
    position: absolute; inset: 0; display: none; place-items: center;
    background: rgba(8, 17, 27, 0.55); font-weight: 500;
  }
  .busy.on { display: grid; }
  .spinner {
    width: 34px; height: 34px; border-radius: 50%; margin: 0 auto 10px;
    border: 3px solid rgba(255, 255, 255, 0.2); border-top-color: var(--accent);
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .side { display: flex; flex-direction: column; gap: 18px; }
  label { display: block; font-size: 0.9rem; color: var(--muted); margin-bottom: 6px; }
  .val { color: var(--ink); font-variant-numeric: tabular-nums; float: right; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  select {
    width: 100%; padding: 9px 10px; border-radius: 9px; color: var(--ink);
    background: rgba(255, 255, 255, 0.08); border: 1px solid var(--line); font: inherit;
  }
  select option { color: #111; }
  button {
    font: inherit; font-weight: 600; padding: 12px 16px; border-radius: 10px; border: 0;
    cursor: pointer; background: var(--accent); color: #1c1305;
  }
  button.ghost { background: transparent; color: var(--ink); border: 1px solid var(--line); font-weight: 500; }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  button:focus-visible, input:focus-visible, select:focus-visible, .stage:focus-visible {
    outline: 2px solid #fff; outline-offset: 2px;
  }

  .results { margin-top: 18px; }
  .stats { display: flex; gap: 28px; flex-wrap: wrap; margin-bottom: 12px; }
  .stat b { display: block; font-family: "Bricolage Grotesque", system-ui, sans-serif; font-size: 1.9rem; line-height: 1; }
  .stat span { color: var(--muted); font-size: 0.85rem; }
  ul.dets { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; max-height: 220px; overflow: auto; }
  ul.dets li {
    display: flex; justify-content: space-between; gap: 12px; padding: 8px 12px; border-radius: 9px;
    background: rgba(255, 255, 255, 0.06); font-size: 0.93rem; font-variant-numeric: tabular-nums;
  }
  .err { color: #ffb4a8; margin-top: 12px; min-height: 1.2em; }
  .empty { color: var(--muted); font-size: 0.93rem; }


  [hidden] { display: none !important; }
  .tabs { display: flex; gap: 8px; margin-bottom: 16px; }
  .tab {
    background: transparent; color: var(--ink); border: 1px solid var(--line);
    font-weight: 500; padding: 8px 16px;
  }
  .tab[aria-selected="true"] { background: var(--accent); color: #1c1305; border-color: var(--accent); font-weight: 600; }
  .bar { height: 8px; border-radius: 99px; background: rgba(255, 255, 255, 0.12); overflow: hidden; margin: 10px 0; }
  .bar i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .3s; }
  .gallery {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px;
    margin-top: 14px; max-height: 560px; overflow: auto;
  }
  .card { background: rgba(255, 255, 255, 0.06); border-radius: 10px; overflow: hidden; font-size: 0.82rem; }
  .card img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; background: #000; }
  .card .cap { padding: 6px 8px; display: flex; justify-content: space-between; gap: 6px; }
  .card .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card.bad .cap { color: #ffb4a8; }
  #zipDl { margin-top: 6px; }

  @media (prefers-reduced-motion: reduce) { .cloud, .spinner { animation: none; } }
</style>
</head>
<body>

<div id="scene" aria-hidden="true">
  <svg class="ridge" viewBox="0 0 1440 220" preserveAspectRatio="none">
    <path d="M0 140 C 180 80, 360 170, 560 120 S 940 60, 1140 130 S 1360 150, 1440 100 L1440 220 L0 220 Z" fill="#0c1a29"/>
    <path d="M0 175 C 220 130, 420 200, 700 160 S 1160 120, 1440 170 L1440 220 L0 220 Z" fill="var(--ridge)"/>
  </svg>
</div>

<main>
  <h1>Turbine Spotter</h1>
  <p class="lead">Upload an aerial or ground photo, or a ZIP of many photos, and the model marks every wind turbine it finds, with a confidence score for each.</p>

  <section class="panel">
    <div class="tabs" role="tablist">
      <button class="tab" id="tabSingle" role="tab" aria-selected="true">Single image</button>
      <button class="tab" id="tabZip" role="tab" aria-selected="false">ZIP of images</button>
    </div>
    <div class="grid">
      <div>
        <div class="stage" id="stage" tabindex="0" role="button" aria-label="Choose or drop an image">
          <div class="hint"><strong>Drop an image here</strong><br>or click to choose a file</div>
          <canvas id="canvas"></canvas>
          <div class="busy" id="busy"><div><div class="spinner"></div>Finding turbines...</div></div>
        </div>
        <input type="file" id="file" accept="image/*" hidden>
        <div class="stage" id="zipStage" tabindex="0" role="button" aria-label="Choose or drop a ZIP file" hidden>
          <div class="hint" id="zipHint"><strong>Drop a ZIP of images here</strong><br>or click to choose a .zip file</div>
        </div>
        <input type="file" id="zipFile" accept=".zip,application/zip,application/x-zip-compressed" hidden>
        <div class="err" id="err" role="alert"></div>
      </div>

      <div class="side">
        <div>
          <label for="conf">Minimum confidence <span class="val" id="confVal">0.25</span></label>
          <input type="range" id="conf" min="0.05" max="0.95" step="0.05" value="0.25">
        </div>
        <div>
          <label for="iou">Overlap threshold (NMS) <span class="val" id="iouVal">0.70</span></label>
          <input type="range" id="iou" min="0.1" max="0.95" step="0.05" value="0.7">
        </div>
        <div>
          <label for="imgsz">Inference size</label>
          <select id="imgsz">
            <option value="640" selected>640 px</option>
            <option value="960">960 px</option>
            <option value="1280">1280 px</option>
          </select>
        </div>
        <button id="run" disabled>Detect turbines</button>
        <button id="reset" class="ghost">Clear image</button>
      </div>
    </div>

    <div class="results" id="results" hidden>
      <div class="stats">
        <div class="stat"><b id="count">0</b><span>turbines found</span></div>
        <div class="stat"><b id="ms">0</b><span>ms inference</span></div>
      </div>
      <ul class="dets" id="dets"></ul>
    </div>

    <div class="results" id="zipResults" hidden>
      <div class="stats">
        <div class="stat"><b id="zImages">0</b><span>images processed</span></div>
        <div class="stat"><b id="zTurbines">0</b><span>turbines found</span></div>
      </div>
      <div class="bar"><i id="zBar"></i></div>
      <div class="empty" id="zStatus"></div>
      <button id="zipDl" hidden>Download results (ZIP)</button>
      <div class="gallery" id="gallery"></div>
    </div>
  </section>
</main>

<script>
/* ---------- background: turbines, clouds ---------- */
(function buildScene() {
  const scene = document.getElementById("scene");
  const turbines = [
    { x: 6,  h: 46, dur: 11, op: 0.55 },
    { x: 24, h: 30, dur: 14, op: 0.4 },
    { x: 47, h: 38, dur: 9,  op: 0.5 },
    { x: 66, h: 26, dur: 13, op: 0.35 },
    { x: 82, h: 52, dur: 12, op: 0.7 },
    { x: 94, h: 34, dur: 10, op: 0.5 }
  ];
  const blade = '<path d="M49 50 Q47.6 28 50 4 Q52.4 28 51 50 Z"/>';
  turbines.forEach(t => {
    const w = t.h * 0.5;
    const el = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    el.setAttribute("class", "turbine");
    el.setAttribute("viewBox", "0 0 100 200");
    el.style.left = t.x + "vw";
    el.style.height = t.h + "vh";
    el.style.width = w + "vh";
    el.style.opacity = t.op;
    el.style.bottom = "6vh";
    el.innerHTML =
      '<g fill="#050b12">' +
        '<polygon points="48.6,50 51.4,50 53,200 47,200"/>' +
        '<g>' +
          '<animateTransform attributeName="transform" type="rotate" from="0 50 50" to="360 50 50" dur="' + t.dur + 's" repeatCount="indefinite"/>' +
          blade +
          '<g transform="rotate(120 50 50)">' + blade + '</g>' +
          '<g transform="rotate(240 50 50)">' + blade + '</g>' +
        '</g>' +
        '<circle cx="50" cy="50" r="2.6"/>' +
      '</g>';
    scene.appendChild(el);
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) el.pauseAnimations();
  });
  [[12, 90], [30, 120], [8, 150]].forEach(([top, dur], i) => {
    const c = document.createElement("div");
    c.className = "cloud";
    c.style.top = top + "vh";
    c.style.animationDuration = dur + "s";
    c.style.animationDelay = (-i * 40) + "s";
    scene.insertBefore(c, scene.firstChild);
  });
})();

/* ---------- app logic ---------- */
const $ = id => document.getElementById(id);
const stage = $("stage"), canvas = $("canvas"), ctx = canvas.getContext("2d");
const fileInput = $("file"), runBtn = $("run"), resetBtn = $("reset");
let currentFile = null, currentImg = null;

$("conf").oninput = e => $("confVal").textContent = (+e.target.value).toFixed(2);
$("iou").oninput = e => $("iouVal").textContent = (+e.target.value).toFixed(2);

stage.addEventListener("click", () => { if (!currentImg) fileInput.click(); });
stage.addEventListener("keydown", e => { if ((e.key === "Enter" || e.key === " ") && !currentImg) { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => fileInput.files[0] && loadFile(fileInput.files[0]));
["dragenter", "dragover"].forEach(ev => stage.addEventListener(ev, e => { e.preventDefault(); stage.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => stage.addEventListener(ev, e => { e.preventDefault(); stage.classList.remove("drag"); }));
stage.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

function loadFile(file) {
  if (!file.type.startsWith("image/")) { showError("Please choose an image file (JPG, PNG, WebP)."); return; }
  showError("");
  const img = new Image();
  img.onload = () => {
    currentFile = file; currentImg = img;
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);
    stage.classList.add("has-image");
    runBtn.disabled = false;
    $("results").hidden = true;
  };
  img.src = URL.createObjectURL(file);
}

resetBtn.onclick = () => {
  if (mode === "zip") { clearZip(); return; }
  currentFile = currentImg = null; fileInput.value = "";
  stage.classList.remove("has-image");
  runBtn.disabled = true; $("results").hidden = true; showError("");
};

function showError(msg) { $("err").textContent = msg; }

async function runSingle() {
  if (!currentFile) return;
  showError(""); $("busy").classList.add("on"); runBtn.disabled = true;
  const body = new FormData();
  body.append("file", currentFile);
  body.append("conf", $("conf").value);
  body.append("iou", $("iou").value);
  body.append("imgsz", $("imgsz").value);
  try {
    const res = await fetch("/predict", { method: "POST", body });
    if (!res.ok) throw new Error((await res.json()).detail || "The server could not process this image.");
    render(await res.json());
  } catch (err) {
    showError(err.message || "Could not reach the server. Is the server still running?");
  } finally {
    $("busy").classList.remove("on"); runBtn.disabled = false;
  }
}

function render(data) {
  ctx.drawImage(currentImg, 0, 0);
  const lw = Math.max(2, Math.round(Math.max(data.width, data.height) / 350));
  const fs = Math.max(13, Math.round(lw * 6));
  ctx.lineWidth = lw; ctx.font = "600 " + fs + "px 'Instrument Sans', system-ui, sans-serif"; ctx.textBaseline = "top";

  data.detections.forEach(d => {
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = "#ffc15e";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = d.label + " " + Math.round(d.confidence * 100) + "%";
    const tw = ctx.measureText(text).width + 10, th = fs + 8;
    const ty = y1 - th >= 0 ? y1 - th : y1;
    ctx.fillStyle = "#ffc15e"; ctx.fillRect(x1, ty, tw, th);
    ctx.fillStyle = "#1c1305"; ctx.fillText(text, x1 + 5, ty + 4);
  });

  $("count").textContent = data.count;
  $("ms").textContent = data.inference_ms;
  const list = $("dets"); list.innerHTML = "";
  if (!data.count) {
    list.innerHTML = '<li class="empty">No turbines found. Try lowering the minimum confidence.</li>';
  } else {
    data.detections.forEach((d, i) => {
      const li = document.createElement("li");
      li.innerHTML = "<span>" + d.label + " #" + (i + 1) + "</span><span>" + (d.confidence * 100).toFixed(1) + "%</span>";
      list.appendChild(li);
    });
  }
  $("results").hidden = false;
}

/* ---------- ZIP of images (bulk) ---------- */
let mode = "single", zipFileObj = null, jobId = null, pollTimer = null;
const zipStage = $("zipStage"), zipInput = $("zipFile");
const ZIP_HINT = "<strong>Drop a ZIP of images here</strong><br>or click to choose a .zip file";

runBtn.onclick = () => (mode === "zip" ? runZip() : runSingle());
$("tabSingle").onclick = () => setMode("single");
$("tabZip").onclick = () => setMode("zip");

function setMode(m) {
  mode = m;
  const zip = m === "zip";
  $("tabSingle").setAttribute("aria-selected", String(!zip));
  $("tabZip").setAttribute("aria-selected", String(zip));
  stage.hidden = zip; zipStage.hidden = !zip;
  $("results").hidden = true;
  $("zipResults").hidden = !(zip && jobId);
  runBtn.textContent = zip ? "Detect in ZIP" : "Detect turbines";
  resetBtn.textContent = zip ? "Clear ZIP" : "Clear image";
  runBtn.disabled = zip ? !zipFileObj : !currentFile;
  showError("");
}

zipStage.addEventListener("click", () => zipInput.click());
zipStage.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); zipInput.click(); } });
zipInput.addEventListener("change", () => zipInput.files[0] && loadZip(zipInput.files[0]));
["dragenter", "dragover"].forEach(ev => zipStage.addEventListener(ev, e => { e.preventDefault(); zipStage.classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => zipStage.addEventListener(ev, e => { e.preventDefault(); zipStage.classList.remove("drag"); }));
zipStage.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) loadZip(f); });

function loadZip(file) {
  if (!file.name.toLowerCase().endsWith(".zip")) { showError("Please choose a .zip file."); return; }
  showError(""); stopPolling();
  zipFileObj = file; jobId = null;
  const hint = $("zipHint"); hint.textContent = "";
  const name = document.createElement("strong"); name.textContent = file.name;
  hint.append(name, document.createElement("br"), (file.size / 1048576).toFixed(1) + " MB ready. Click Detect in ZIP.");
  $("zipResults").hidden = true;
  runBtn.disabled = false;
}

function clearZip() {
  stopPolling();
  zipFileObj = null; jobId = null; zipInput.value = "";
  $("zipHint").innerHTML = ZIP_HINT;
  $("zipResults").hidden = true; $("gallery").innerHTML = "";
  runBtn.disabled = true; showError("");
}

function stopPolling() { if (pollTimer) clearTimeout(pollTimer); pollTimer = null; }

async function runZip() {
  if (!zipFileObj) return;
  showError(""); runBtn.disabled = true; stopPolling();
  $("gallery").innerHTML = ""; $("zipDl").hidden = true;
  $("zImages").textContent = "0"; $("zTurbines").textContent = "0"; $("zBar").style.width = "0%";
  $("zStatus").textContent = "Uploading ZIP...";
  $("zipResults").hidden = false;
  const body = new FormData();
  body.append("file", zipFileObj);
  body.append("conf", $("conf").value);
  body.append("iou", $("iou").value);
  body.append("imgsz", $("imgsz").value);
  try {
    const res = await fetch("/predict_zip", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "The server could not read this ZIP.");
    jobId = data.job_id;
    poll();
  } catch (err) {
    showError(err.message || "Could not reach the server.");
    $("zipResults").hidden = true; runBtn.disabled = false;
  }
}

async function poll() {
  try {
    const res = await fetch("/jobs/" + jobId);
    if (!res.ok) throw new Error("Lost track of this job. Please run it again.");
    const job = await res.json();
    renderJob(job);
    if (job.status === "running") { pollTimer = setTimeout(poll, 1500); return; }
    runBtn.disabled = false;
    if (job.status === "error") showError(job.message || "Processing failed.");
  } catch (err) {
    showError(err.message || "Could not reach the server.");
    runBtn.disabled = false;
  }
}

function renderJob(job) {
  $("zImages").textContent = job.done + " / " + job.total;
  $("zTurbines").textContent = job.turbines;
  $("zBar").style.width = Math.round((job.done / job.total) * 100) + "%";
  $("zStatus").textContent = job.status === "running"
    ? "Processing " + job.done + " of " + job.total + "..."
    : (job.status === "done" ? "Finished." : (job.message || ""));
  const gal = $("gallery");
  for (let i = gal.children.length; i < job.rows.length; i++) gal.appendChild(makeCard(job.rows[i]));
  if (job.status === "done") $("zipDl").hidden = false;
}

function makeCard(r) {
  const el = document.createElement("div");
  el.className = "card" + (r.error ? " bad" : "");
  if (!r.error) {
    const img = document.createElement("img");
    img.loading = "lazy"; img.alt = r.name;
    img.src = "/jobs/" + jobId + "/image/" + r.index;
    el.appendChild(img);
  }
  const cap = document.createElement("div"); cap.className = "cap";
  const nm = document.createElement("span"); nm.className = "name"; nm.title = r.name; nm.textContent = r.name;
  const ct = document.createElement("span");
  ct.textContent = r.error ? r.error : r.count + " found";
  cap.append(nm, ct); el.appendChild(cap);
  return el;
}

$("zipDl").onclick = () => { window.location.href = "/jobs/" + jobId + "/download"; };
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)