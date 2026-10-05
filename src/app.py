"""
Wind turbine detector - FastAPI + YOLO (Ultralytics) + single-file frontend.

Azure Container Apps version:
  - The model is NOT bundled in the repo. src/download_model.py pulls it from
    Azure Blob Storage into model/best.pt before this app starts.
  - Started by the Dockerfile with: uvicorn src.app:app --host 0.0.0.0 --port 8000
"""

import io
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from ultralytics import YOLO

# download_model.py saves the blob here (relative to /app, the container workdir).
MODEL_PATH = os.getenv("MODEL_PATH", "model/best.pt")

state = {}


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
    return {"classes": list(model.names.values()), "model_path": MODEL_PATH}


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
    except Exception:
        raise HTTPException(status_code=400, detail="That file is not a readable image.")

    model = state["model"]
    start = time.perf_counter()
    result = model.predict(image, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    detections = []
    for xyxy, score, cls in zip(
        result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
    ):
        detections.append(
            {
                "label": model.names[int(cls)],
                "confidence": round(score, 4),
                "box": [round(v, 1) for v in xyxy],
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "width": image.width,
        "height": image.height,
        "inference_ms": elapsed_ms,
        "count": len(detections),
        "detections": detections,
    }


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
  <p class="lead">Upload an aerial or ground photo and the model marks every wind turbine it finds, with a confidence score for each.</p>

  <section class="panel">
    <div class="grid">
      <div>
        <div class="stage" id="stage" tabindex="0" role="button" aria-label="Choose or drop an image">
          <div class="hint"><strong>Drop an image here</strong><br>or click to choose a file</div>
          <canvas id="canvas"></canvas>
          <div class="busy" id="busy"><div><div class="spinner"></div>Finding turbines...</div></div>
        </div>
        <input type="file" id="file" accept="image/*" hidden>
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
  currentFile = currentImg = null; fileInput.value = "";
  stage.classList.remove("has-image");
  runBtn.disabled = true; $("results").hidden = true; showError("");
};

function showError(msg) { $("err").textContent = msg; }

runBtn.onclick = async () => {
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
};

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
</script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
