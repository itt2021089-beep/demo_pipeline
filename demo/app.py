"""
Supervisor demo UI — Mosquito Larva Classification System.

    python -m streamlit run demo/app.py

Presentation layer only. All inference is delegated to demo/inference.py, which
is unchanged; this file computes no predictions of its own and hard-codes no
result. Every figure shown is either a live forward pass or a value traceable
to a project file via demo/config.py.
"""

import os
import sys
import time

# Allow `streamlit run demo/app.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image

from demo import config
from demo import inference as inf

st.set_page_config(page_title="Mosquito Larva Classification System",
                   page_icon="🦟", layout="wide",
                   initial_sidebar_state="expanded")

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 4.5rem; padding-bottom: 3rem; max-width: 1280px; }

  /* Header banner — painted background so the title stays high-contrast in
     both the light and the dark Streamlit theme. */
  .hdr-band {
      background: linear-gradient(100deg,#1b5e20 0%,#2e7d32 45%,#43a047 100%);
      border-radius: 14px; padding: 1.15rem 1.5rem 1.25rem 1.5rem;
      margin: 0 0 1.5rem 0; box-shadow: 0 2px 12px rgba(0,0,0,.16);
  }
  .hdr-title {
      color: #ffffff;
      font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em;
      margin: 0 0 .22rem 0; line-height: 1.15;
      text-shadow: 0 1px 2px rgba(0,0,0,.22);
  }
  .hdr-sub {
      color: rgba(255,255,255,.90);
      font-size: 1.02rem; font-weight: 400; margin: 0; line-height: 1.35;
  }
  .hdr-rule { height: 3px; background: linear-gradient(90deg,#2e7d32,#66bb6a 45%,transparent);
              border-radius: 2px; margin: .9rem 0 1.4rem 0; }

  .sec-label {
      font-size: .74rem; font-weight: 700; letter-spacing: .13em;
      text-transform: uppercase; opacity: .55; margin: 1.6rem 0 .55rem 0;
  }

  /* pipeline */
  .pipe { display:flex; flex-direction:column; gap:0; }
  .node {
      border:1px solid rgba(128,128,128,.28); border-radius:9px;
      padding:.6rem .8rem; background:rgba(128,128,128,.05);
  }
  .node-t { font-size:.83rem; font-weight:650; line-height:1.25; }
  .node-s { font-size:.72rem; opacity:.62; margin-top:.12rem; }
  .node.done   { border-color:#2e7d32; background:rgba(46,125,50,.11); }
  .node.done .node-t { color:#2e7d32; }
  .node.stop   { border-color:#ef6c00; background:rgba(239,108,0,.11); }
  .node.stop .node-t { color:#ef6c00; }
  .node.skip   { border-style:dashed; opacity:.42; }
  .node.idle   { opacity:.5; }
  .arrow { text-align:center; font-size:1rem; opacity:.35; line-height:1.1; margin:.18rem 0; }

  /* cards */
  .card {
      border:1px solid rgba(128,128,128,.25); border-radius:12px;
      padding:1.05rem 1.2rem; margin-bottom:.35rem;
      background:rgba(128,128,128,.035);
  }
  .card.accent-g { border-left:4px solid #2e7d32; }
  .card.accent-o { border-left:4px solid #ef6c00; }
  .card.accent-b { border-left:4px solid #1565c0; }
  .card.muted { opacity:.72; }

  .card-h { font-size:.76rem; font-weight:700; letter-spacing:.1em;
            text-transform:uppercase; opacity:.6; margin-bottom:.5rem; }
  .verdict { font-size:1.85rem; font-weight:750; letter-spacing:-.02em;
             line-height:1.1; margin:.1rem 0 .1rem 0; }
  .verdict.g { color:#2e7d32; } .verdict.o { color:#ef6c00; }
  .sub { font-size:.8rem; opacity:.62; }

  .final {
      border:1px solid rgba(128,128,128,.3); border-radius:14px;
      padding:1.4rem 1.5rem; background:rgba(21,101,192,.055);
      border-left:5px solid #1565c0;
  }
  .final-l { font-size:.76rem; font-weight:700; letter-spacing:.13em;
             text-transform:uppercase; opacity:.6; }
  .final-v { font-size:2.5rem; font-weight:780; letter-spacing:-.03em;
             line-height:1.1; margin:.25rem 0 .1rem 0; }

  .prob-row { display:flex; justify-content:space-between; font-size:.83rem;
              margin-bottom:.1rem; }
  .prob-row .nm { font-weight:600; }
  .prob-row.dim { opacity:.55; font-weight:400; }

  .pill { display:inline-block; font-size:.68rem; font-weight:700;
          letter-spacing:.05em; padding:.16rem .5rem; border-radius:20px;
          text-transform:uppercase; }
  .pill.live { background:rgba(46,125,50,.16); color:#2e7d32; }
  .pill.warn { background:rgba(239,108,0,.16); color:#ef6c00; }
  .pill.info { background:rgba(21,101,192,.16); color:#1565c0; }
  .pill.grey { background:rgba(128,128,128,.16); opacity:.75; }

  .meta { font-size:.78rem; opacity:.65; line-height:1.5; }
  [data-testid="stMetricValue"] { font-size:1.45rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Stage 1 and Stage 2 models …")
def _warm():
    inf.warmup()
    return True


def _pct(x):
    return f"{x*100:.2f}%"


def _node(title, sub, state):
    st.markdown(
        f'<div class="node {state}"><div class="node-t">{title}</div>'
        f'<div class="node-s">{sub}</div></div>', unsafe_allow_html=True)


def _arrow():
    st.markdown('<div class="arrow">↓</div>', unsafe_allow_html=True)


def _probs(dist, winner):
    for name, p in dist.items():
        cls = "" if name == winner else " dim"
        st.markdown(
            f'<div class="prob-row{cls}"><span class="nm">{name}</span>'
            f'<span>{_pct(p)}</span></div>', unsafe_allow_html=True)
        st.progress(min(max(float(p), 0.0), 1.0))


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    '<div class="hdr-band">'
    '<div class="hdr-title">🦟 Mosquito Larva Classification System</div>'
    '<div class="hdr-sub">Multi-Stage Deep Learning Classification Pipeline '
    '— ResNet50 → EfficientNet-B0</div>'
    '</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar — research status
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Research Status")

    st.markdown('<div class="sec-label">A · Current live demo</div>',
                unsafe_allow_html=True)
    for row in config.LIVE_DEMO_STATUS:
        pill = "live" if row["status"] == "Live inference" else "info"
        st.markdown(
            f'<div style="margin-bottom:.55rem"><b style="font-size:.86rem">'
            f'{row["stage"]}</b><br><span class="pill {pill}">{row["status"]}</span>'
            f'<div class="meta" style="margin-top:.2rem">{row["detail"]}</div></div>',
            unsafe_allow_html=True)

    st.markdown('<div class="sec-label">B · Research results</div>',
                unsafe_allow_html=True)
    for e in config.EVALUATION_STATUS:
        pill = "info" if e["verified"] else "grey"
        label = "Historical Research Result" if e["verified"] else "Not evaluated here"
        st.markdown(
            f'<div style="margin-bottom:.55rem"><b style="font-size:.86rem">'
            f'{e["model"]}</b><br><span class="pill {pill}">{label}</span>'
            f'<div class="meta" style="margin-top:.2rem"><b>{e["result"]}</b><br>'
            f'{e["context"]}<br><code style="font-size:.72rem">{e["source"]}</code>'
            f'</div></div>', unsafe_allow_html=True)
    st.info(config.HISTORICAL_RESULTS_NOTE, icon="ℹ️")

    st.markdown('<div class="sec-label">C · Stage 2 class mapping</div>',
                unsafe_allow_html=True)
    st.warning(config.STAGE2_MAPPING_WARNING, icon="⚠️")
    st.markdown(
        '<span class="pill warn">User-Confirmed Mapping</span>'
        f'<div class="meta" style="margin-top:.3rem">'
        f'{ {i: c for i, c in enumerate(config.STAGE2_CLASSES)} }</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="meta" style="margin-top:.5rem">Stage 1 mapping '
        '<span class="pill info">Source-Verified</span><br>'
        f'{ {i: c for i, c in enumerate(config.STAGE1_CLASSES)} } — derived from '
        'the training dataset folder order.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-label">D · Dataset / evaluation</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="meta">{config.DATASET_STATUS_NOTE}</div>',
                unsafe_allow_html=True)

_warm()

# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
st.markdown('<div class="sec-label">1 · Input image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a mosquito larva image", type=["jpg", "jpeg", "png"],
    label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a JPG or PNG image to run the two-stage pipeline.", icon="⬆️")
    st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
    c = st.columns(9)
    for col, (t, s) in zip(c[::2], [
            ("Upload Image", "JPG / PNG"), ("Preprocessing", "resize · normalize"),
            ("Stage 1", "ResNet50"), ("Stage 2", "EfficientNet-B0"),
            ("Final Prediction", "class + confidence")]):
        with col:
            _node(t, s, "idle")
    for col in c[1::2]:
        with col:
            st.markdown('<div class="arrow" style="padding-top:1.1rem">→</div>',
                        unsafe_allow_html=True)
    st.stop()

img = Image.open(uploaded)
img_rgb = img.convert("RGB")
w, h = img.size
fmt = (img.format or os.path.splitext(uploaded.name)[1].lstrip(".")).upper()

# ── run the pipeline (single source of truth: demo/inference.py) ──
uploaded.seek(0)
t0 = time.perf_counter()
result = inf.run_pipeline(uploaded)
wall_ms = (time.perf_counter() - t0) * 1000

s1, s2 = result["stage1"], result["stage2"]
ran2 = result["stage2_executed"]

# ─────────────────────────────────────────────
# Layout: pipeline rail | image+preproc | results
# ─────────────────────────────────────────────
rail, mid, right = st.columns([1.05, 1.35, 2.0], gap="large")

with rail:
    st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
    _node("Upload Image", f"{fmt} · {w}×{h}", "done"); _arrow()
    _node("Preprocessing", "resize · tensor · normalize", "done"); _arrow()
    _node("Stage 1 — ResNet50", "binary classifier", "done"); _arrow()
    _node(s1["class"].replace("_", "-").upper(),
          f"{_pct(s1['confidence'])} confidence",
          "done" if s1["is_larva"] else "stop")
    _arrow()
    if ran2:
        _node("Stage 2 — EfficientNet-B0", "species classifier", "done"); _arrow()
        _node("Species Classification", s2["class"], "done"); _arrow()
        _node("Final Prediction", f"{result['final_class']} · "
              f"{_pct(result['final_confidence'])}", "done")
    else:
        _node("Stage 2 — EfficientNet-B0", "not executed", "skip"); _arrow()
        _node("Species Classification", "skipped", "skip"); _arrow()
        _node("Final Prediction", "Non-larva · stopped at Stage 1", "stop")

with mid:
    st.markdown('<div class="sec-label">2 · Original image</div>',
                unsafe_allow_html=True)
    st.image(img_rgb, use_container_width=True)
    m1, m2 = st.columns(2)
    m1.metric("Dimensions", f"{w} × {h}")
    m2.metric("File type", fmt)

    st.markdown('<div class="sec-label">3 · Preprocessing applied</div>',
                unsafe_allow_html=True)
    for stage_name, steps in config.PREPROCESSING_STEPS.items():
        run_note = ""
        if stage_name.startswith("Stage 2") and not ran2:
            run_note = ' <span class="pill grey">not executed</span>'
        st.markdown(f'<div class="card" style="padding:.75rem .9rem">'
                    f'<div class="card-h" style="margin-bottom:.35rem">'
                    f'{stage_name}{run_note}</div>' +
                    "".join(
                        f'<div class="prob-row"><span class="nm">{n}</span>'
                        f'<span style="opacity:.7;text-align:right;max-width:60%">'
                        f'{v}</span></div>' for n, v in steps) +
                    '</div>', unsafe_allow_html=True)

with right:
    # ── Stage 1 card ──
    st.markdown('<div class="sec-label">4 · Stage 1 result</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-h">Stage 1 — ResNet50 · '
                    '<span class="pill live">Live Prediction</span></div>',
                    unsafe_allow_html=True)
        a, b = st.columns([1.05, 1])
        with a:
            st.markdown(f'<div class="verdict {"g" if s1["is_larva"] else "o"}">'
                        f'{s1["class"].replace("_","-").upper()}</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="sub">Confidence <b>{_pct(s1["confidence"])}</b>'
                        f' · {s1["inference_ms"]:.0f} ms</div>',
                        unsafe_allow_html=True)
        with b:
            _probs(s1["probabilities"], s1["class"])

    if not s1["is_larva"]:
        st.error("**Stage 2 not executed** — pipeline stopped because the image "
                 "was classified as non-larva.", icon="🛑")

    # ── Stage 2 card ──
    st.markdown('<div class="sec-label">5 · Stage 2 result</div>',
                unsafe_allow_html=True)
    if ran2:
        with st.container(border=True):
            st.markdown('<div class="card-h">Stage 2 — EfficientNet-B0 · '
                        '<span class="pill live">Live Prediction</span> '
                        '<span class="pill warn">User-Confirmed Mapping</span></div>',
                        unsafe_allow_html=True)
            a, b = st.columns([1.05, 1])
            with a:
                st.markdown(f'<div class="verdict g">{s2["class"]}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="sub">Confidence '
                            f'<b>{_pct(s2["confidence"])}</b>'
                            f' · {s2["inference_ms"]:.0f} ms<br>'
                            f'output index {s2["class_index"]}</div>',
                            unsafe_allow_html=True)
            with b:
                _probs(s2["probabilities"], s2["class"])
    else:
        st.markdown('<div class="card muted" style="border-style:dashed">'
                    '<div class="card-h">Stage 2 — EfficientNet-B0 '
                    '<span class="pill grey">Not executed</span></div>'
                    '<div class="meta">The pipeline halts at Stage 1 when the '
                    'image is not classified as a larva, so no species '
                    'prediction is produced.</div></div>',
                    unsafe_allow_html=True)

    # ── Final prediction ──
    st.markdown('<div class="sec-label">6 · Final prediction</div>',
                unsafe_allow_html=True)
    label = "Species / Class" if ran2 else "Class"
    status = "Completed" if ran2 else "Stopped after Stage 1"
    status_pill = "live" if ran2 else "warn"
    st.markdown(
        f'<div class="final">'
        f'<div class="final-l">Final Prediction</div>'
        f'<div class="final-v">{result["final_class"]}</div>'
        f'<div class="sub">{label}</div>'
        f'<div style="margin-top:.6rem"><span class="final-l">Pipeline status</span>'
        f'&nbsp;&nbsp;<span class="pill {status_pill}">{status}</span></div>'
        f'</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    f1.metric("Confidence", _pct(result["final_confidence"]))
    f2.metric("Total inference time", f"{wall_ms:.0f} ms")

    t1, t2 = st.columns(2)
    t1.metric("Stage 1 time", f"{s1['inference_ms']:.0f} ms")
    t2.metric("Stage 2 time", f"{s2['inference_ms']:.0f} ms" if ran2 else "not run")

    st.caption("Times are model forward-pass durations; the total additionally "
               "includes image decoding and preprocessing.")

with st.expander("Raw pipeline output (JSON)"):
    st.json(result)

st.markdown('<div class="hdr-rule" style="margin-top:2rem"></div>',
            unsafe_allow_html=True)
st.caption("Live single-image demonstration. Predictions shown here are runtime "
           "output and are not an evaluation of model generalization; see the "
           "Research Status panel for what has and has not been formally "
           "evaluated.")
