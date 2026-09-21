import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pickle
import os
import time

from PIL import Image
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "face_database/face_database.pkl"
THRESHOLD = 0.30
MAX_IMAGE_WIDTH = 640
SCAN_DURATION = 3.5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Face Recognition // HOLO",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# MATRIX RAIN BACKGROUND (subtle)
# ============================================================

components.html(
    """
    <style>
        html, body {
            margin: 0; padding: 0;
            background: transparent;
            overflow: hidden;
        }
        #matrix {
            position: fixed;
            inset: 0;
            z-index: -1;
            opacity: 0.15;
            pointer-events: none;
        }
    </style>
    <canvas id="matrix"></canvas>
    <script>
        const c = document.getElementById('matrix');
        const ctx = c.getContext('2d');
        let W, H, cols, drops;
        const glyphs = "アカサタナハマヤラワ0123456789ABCDEF<>/\\\\|=+*#@$%";

        function init() {
            W = c.width = window.innerWidth || 1200;
            H = c.height = window.innerHeight || 800;
            cols = Math.floor(W / 16);
            drops = new Array(cols).fill(1).map(() => Math.random() * -100);
        }
        init();
        window.addEventListener('resize', init);

        function draw() {
            ctx.fillStyle = 'rgba(3, 6, 15, 0.10)';
            ctx.fillRect(0, 0, W, H);
            ctx.font = '14px monospace';
            for (let i = 0; i < cols; i++) {
                const ch = glyphs[Math.floor(Math.random() * glyphs.length)];
                const y = drops[i] * 16;
                ctx.fillStyle = Math.random() > 0.975 ? '#7fd8ff' : '#005a8a';
                ctx.fillText(ch, i * 16, y);
                if (y > H && Math.random() > 0.975) drops[i] = 0;
                drops[i]++;
            }
        }
        setInterval(draw, 45);
    </script>
    """,
    height=0,
    width=0,
)


# ============================================================
# CUSTOM CSS — BLUE HOLOGRAPHIC THEME
# ============================================================

st.markdown(
    """
    <style>

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes pulseGlow {
        0%   { box-shadow: 0 0 0 0 rgba(0, 200, 255, 0.45); }
        70%  { box-shadow: 0 0 0 16px rgba(0, 200, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 200, 255, 0); }
    }
    @keyframes pulseRed {
        0%   { box-shadow: 0 0 0 0 rgba(255, 60, 80, 0.45); }
        70%  { box-shadow: 0 0 0 16px rgba(255, 60, 80, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 60, 80, 0); }
    }
    @keyframes scanLineH {
        0%   { left: -100%; opacity: 0; }
        10%  { opacity: 1; }
        90%  { opacity: 1; }
        100% { left: 100%; opacity: 0; }
    }
    @keyframes float {
        0%,100% { transform: translateY(0); }
        50%     { transform: translateY(-8px); }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.25; }
    }
    @keyframes glitch {
        0%, 100% { text-shadow: 0 0 0 transparent; transform: translate(0); }
        20%      { text-shadow: -2px 0 #ff00a0, 2px 0 #00ffe1; transform: translate(-1px, 1px); }
        40%      { text-shadow:  2px 0 #ff00a0, -2px 0 #00ffe1; transform: translate(1px, -1px); }
        60%      { text-shadow: -1px 0 #ff00a0, 1px 0 #00ffe1; transform: translate(0, 0); }
        80%      { text-shadow:  1px 0 #ff00a0, -1px 0 #00ffe1; transform: translate(1px, 1px); }
    }

    .hero, .card, .result-known, .result-unknown {
        animation: fadeInUp 0.6s cubic-bezier(.21,.98,.6,1) both;
    }
    .card  { animation-delay: 0.10s; }

    .stApp {
        background:
            radial-gradient(circle at 12% 8%, rgba(0, 180, 255, 0.14), transparent 30%),
            radial-gradient(circle at 88% 18%, rgba(0, 90, 255, 0.12), transparent 32%),
            radial-gradient(circle at 50% 100%, rgba(0, 200, 255, 0.06), transparent 40%),
            #03060f;
        color: #eaf6ff;
    }
    .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 34px;
        border-radius: 22px;
        border: 1px solid rgba(0, 200, 255, 0.28);
        background: linear-gradient(135deg, rgba(0, 60, 130, 0.45), rgba(4, 10, 24, 0.96));
        box-shadow: 0 0 60px rgba(0, 160, 255, 0.10), inset 0 0 40px rgba(0, 200, 255, 0.05);
        margin-bottom: 22px;
    }
    .hero::after {
        content: "";
        position: absolute;
        left: 0; right: 0; top: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #42d8ff, transparent);
        animation: scanLineH 4s linear infinite;
    }
    .hero-title {
        font-size: 36px;
        font-weight: 900;
        letter-spacing: 1.2px;
        color: #f2fbff;
        margin-bottom: 6px;
        text-transform: uppercase;
        animation: glitch 6s infinite;
    }
    .hero-title span { color: #48d9ff; text-shadow: 0 0 14px #48d9ff; }
    .hero-subtitle {
        color: #8ea8bd;
        font-size: 14px;
        letter-spacing: 0.6px;
        font-family: 'Courier New', monospace;
    }
    .status {
        display: inline-block;
        margin-top: 16px;
        padding: 7px 14px;
        border-radius: 20px;
        background: rgba(0, 180, 255, 0.08);
        border: 1px solid rgba(0, 200, 255, 0.30);
        color: #6fdfff;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        animation: pulseGlow 2.4s infinite;
    }

    .card {
        background: linear-gradient(145deg, rgba(13, 26, 46, 0.96), rgba(5, 11, 24, 0.96));
        border: 1px solid rgba(80, 170, 220, 0.18);
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 14px 44px rgba(0, 0, 0, 0.30);
        transition: transform 0.28s ease, border-color 0.28s ease;
        position: relative;
        overflow: hidden;
    }
    .card::after {
        content: "";
        position: absolute;
        left: 0; right: 0; top: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,200,255,0.6), transparent);
        animation: scanLineH 5s linear infinite;
    }
    .card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 200, 255, 0.45);
    }
    .card-title { font-size: 19px; font-weight: 800; color: #eaf8ff; margin-bottom: 5px; }
    .card-description { font-size: 13px; color: #7891a8; margin-bottom: 12px; font-family: 'Courier New', monospace; }

    .result-known {
        padding: 22px; border-radius: 16px;
        background: linear-gradient(135deg, rgba(0, 200, 255, 0.12), rgba(0, 80, 120, 0.05));
        border: 1px solid rgba(0, 200, 255, 0.40);
        margin-top: 20px; position: relative; overflow: hidden;
        animation: fadeInUp 0.5s ease both, pulseGlow 2.4s infinite 0.6s;
    }
    .result-known::before {
        content: ""; position: absolute; inset: 0;
        background: linear-gradient(90deg, transparent, rgba(0,200,255,0.10), transparent);
        animation: scanLineH 3s linear infinite; pointer-events: none;
    }
    .result-unknown {
        padding: 22px; border-radius: 16px;
        background: linear-gradient(135deg, rgba(255, 60, 80, 0.12), rgba(100, 20, 30, 0.05));
        border: 1px solid rgba(255, 80, 100, 0.40);
        margin-top: 20px; position: relative; overflow: hidden;
        animation: fadeInUp 0.5s ease both, pulseRed 2.4s infinite 0.6s;
    }
    .result-unknown::before {
        content: ""; position: absolute; inset: 0;
        background: linear-gradient(90deg, transparent, rgba(255,80,100,0.10), transparent);
        animation: scanLineH 3s linear infinite; pointer-events: none;
    }
    .result-title { font-size: 26px; font-weight: 900; letter-spacing: 1.2px; text-transform: uppercase; }
    .result-score { font-size: 13px; color: #9ab1c5; margin-top: 6px; font-family: 'Courier New', monospace; }

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        min-height: 48px;
        border: 1px solid rgba(0, 220, 255, 0.40);
        background: linear-gradient(135deg, #0878b8, #0750a0);
        color: white;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        transition: all 0.22s ease;
    }
    .stButton > button:hover {
        border-color: #42e0ff;
        box-shadow: 0 0 26px rgba(0, 200, 255, 0.45);
        transform: translateY(-2px);
    }

    .stTextInput input {
        background: #0b1424 !important;
        color: #eaf8ff !important;
        border: 1px solid #1e3b55 !important;
        border-radius: 11px !important;
        font-family: 'Courier New', monospace !important;
    }
    .stTextInput input:focus {
        border-color: #48d9ff !important;
        box-shadow: 0 0 12px rgba(0,200,255,0.35) !important;
    }
    .stRadio label { color: #c4d8e8 !important; }

    [data-testid="stFileUploader"] {
        background: rgba(8, 20, 36, 0.75);
        border: 1px dashed rgba(0, 200, 255, 0.40);
        border-radius: 14px;
        padding: 10px;
    }

    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        animation: fadeIn 0.6s ease both;
        box-shadow: 0 12px 34px rgba(0, 0, 0, 0.45);
        border: 1px solid rgba(0, 200, 255, 0.20);
    }

    [data-testid="stMetric"] {
        background: rgba(10, 25, 42, 0.80);
        border: 1px solid rgba(80, 160, 220, 0.18);
        border-radius: 14px;
        padding: 14px;
    }
    [data-testid="stMetricValue"] {
        color: #7fd8ff !important;
        text-shadow: 0 0 12px rgba(0,200,255,0.5);
        font-family: 'Courier New', monospace !important;
    }

    [data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #48d9ff, #7fbfff) !important;
        box-shadow: 0 0 10px rgba(0,200,255,0.6);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid rgba(0, 200, 255, 0.15);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        color: #7f9aad;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        color: #48e0ff !important;
        text-shadow: 0 0 12px rgba(0,200,255,0.7);
    }

    .placeholder { animation: float 3.6s ease-in-out infinite; }

    .footer {
        text-align: center;
        color: #526a80;
        font-size: 11px;
        font-family: 'Courier New', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 🔵 BLUE HOLOGRAPHIC FACE SCAN TEMPLATE
# ============================================================

FACE_SCAN_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<style>
  html, body {
    margin: 0; padding: 0;
    background: #000;
    font-family: 'Courier New', monospace;
    overflow: hidden;
    color: #42d8ff;
    border-radius: 16px;
  }

  #overlay {
    position: relative;
    width: 100%;
    height: 100vh;
    background: #000;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.35s ease both;
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(0, 200, 255, 0.35);
    box-shadow:
      0 0 50px rgba(0,180,255,0.25),
      inset 0 0 60px rgba(0,180,255,0.06);
  }

  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; visibility: hidden; } }

  #stage {
    position: relative;
    width: 100%;
    height: 100%;
    background: #000;
    overflow: hidden;
  }

  #face {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    filter: drop-shadow(0 0 6px #42d8ff);
  }

  .scanline {
    position: absolute;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg,
      transparent, #42d8ff 40%, #a7e8ff 50%, #42d8ff 60%, transparent);
    box-shadow: 0 0 16px #42d8ff, 0 0 32px #42d8ff;
    animation: scanDown 3.2s linear infinite;
    z-index: 4;
  }
  @keyframes scanDown {
    0%   { top: -3%; opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { top: 103%; opacity: 0; }
  }

  .beam {
    position: absolute;
    inset: 10%;
    border-radius: 50%;
    background: radial-gradient(circle,
      rgba(0, 180, 255, 0.14) 0%,
      rgba(0, 140, 220, 0.05) 40%,
      transparent 70%);
    animation: beamPulse 3s ease-in-out infinite;
    z-index: 1;
    pointer-events: none;
  }
  @keyframes beamPulse {
    0%,100% { transform: scale(1);   opacity: 0.6; }
    50%     { transform: scale(1.15); opacity: 1; }
  }

  .corner {
    position: absolute;
    width: 56px;
    height: 56px;
    border: 2px solid #1a5f7a;
    z-index: 5;
    box-shadow: 0 0 12px rgba(26, 95, 122, 0.6);
  }
  .corner.tl { top: 20px;    left: 20px;    border-right: none; border-bottom: none; }
  .corner.tr { top: 20px;    right: 20px;   border-left: none;  border-bottom: none; }
  .corner.bl { bottom: 20px; left: 20px;    border-right: none; border-top: none; }
  .corner.br { bottom: 20px; right: 20px;   border-left: none;  border-top: none; }

  .corner::before {
    content: "";
    position: absolute;
    width: 100%; height: 100%;
    border: 2px solid #42d8ff;
    opacity: 0;
    animation: cornerGlow 2s ease-in-out infinite;
  }
  .corner.tl::before { top: -2px; left: -2px;    border-right: none; border-bottom: none; }
  .corner.tr::before { top: -2px; right: -2px;   border-left: none;  border-bottom: none; }
  .corner.bl::before { bottom: -2px; left: -2px; border-right: none; border-top: none; }
  .corner.br::before { bottom: -2px; right: -2px; border-left: none; border-top: none; }

  @keyframes cornerGlow {
    0%, 100% { opacity: 0; }
    50%      { opacity: 1; }
  }

  .reticle {
    position: absolute;
    top: 50%; left: 50%;
    width: 260px; height: 260px;
    transform: translate(-50%, -50%);
    border: 1px solid rgba(66, 216, 255, 0.25);
    border-radius: 50%;
    z-index: 3;
    pointer-events: none;
    animation: rotate 12s linear infinite;
  }
  .reticle::before,
  .reticle::after {
    content: "";
    position: absolute;
    border: 1px solid rgba(66, 216, 255, 0.15);
    border-radius: 50%;
  }
  .reticle::before { inset: 20px; }
  .reticle::after  { inset: 40px; }

  @keyframes rotate {
    to { transform: translate(-50%, -50%) rotate(360deg); }
  }

  .hud {
    position: absolute; inset: 0;
    z-index: 6; pointer-events: none;
    font-size: 9px;
    letter-spacing: 1.5px;
    color: rgba(66, 216, 255, 0.75);
    text-shadow: 0 0 6px rgba(66, 216, 255, 0.6);
    font-family: 'Courier New', monospace;
    text-transform: uppercase;
  }
  .hud .tl-text { position: absolute; top: 82px;    left: 84px; }
  .hud .tr-text { position: absolute; top: 82px;    right: 84px; text-align: right; }
  .hud .bl-text { position: absolute; bottom: 82px; left: 84px; }
  .hud .br-text { position: absolute; bottom: 82px; right: 84px; text-align: right; }

  .hud .blink { animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }

  #status {
    position: absolute;
    bottom: 26px;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 22px;
    border: 1px solid rgba(66, 216, 255, 0.4);
    border-radius: 999px;
    background: rgba(0, 15, 25, 0.85);
    color: #7fd8ff;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    text-shadow: 0 0 8px #42d8ff;
    box-shadow: 0 0 20px rgba(0,180,255,0.35);
    animation: statusFlicker 2.4s infinite;
    z-index: 7;
  }
  @keyframes statusFlicker {
    0%, 92%, 100% { opacity: 1; }
    94%           { opacity: .35; }
    96%           { opacity: 1; }
    97%           { opacity: .5; }
  }
</style>
</head>
<body>

<div id="overlay">
  <div id="stage">
    <canvas id="face" width="700" height="700"></canvas>
    <div class="beam"></div>
    <div class="scanline"></div>
    <div class="reticle"></div>

    <div class="corner tl"></div>
    <div class="corner tr"></div>
    <div class="corner bl"></div>
    <div class="corner br"></div>

    <div class="hud">
      <div class="tl-text">MODE: __MODE__<br>RES: 512-D</div>
      <div class="tr-text">STATUS: <span class="blink">SCAN</span><br>FPS: <span id="fps">60</span></div>
      <div class="bl-text">LMKS: 468<br>CONF: 0.987</div>
      <div class="br-text">MATCH: <span id="match">--</span><br>THRESH: 0.30</div>
    </div>

    <div id="status">&#9679; __STATUS__</div>
  </div>
</div>

<script>
/* =====================================================
   BLUE HOLOGRAPHIC NEURAL FACE
   ===================================================== */
var face = document.getElementById('face');
var fctx = face.getContext('2d');
var FW = face.width;
var FH = face.height;

var CX = FW / 2, CY = FH / 2;
var RX = FW * 0.28, RY = FH * 0.40, RZ = 130;

var LAT = 40, LON = 52;
var points = [];

for (var i = 0; i <= LAT; i++) {
  var phi = Math.PI * (i / LAT);
  for (var j = 0; j <= LON; j++) {
    var theta = 2 * Math.PI * (j / LON);
    var x = Math.sin(phi) * Math.cos(theta) * RX;
    var y = Math.cos(phi) * RY;
    var z = Math.sin(phi) * Math.sin(theta) * RZ;

    var nx = x / RX;
    var ny = y / RY;

    if (ny > 0.35) {
      var t = (ny - 0.35) / 0.65;
      x *= (1 - 0.55 * t * t);
    }
    if (Math.abs(ny) < 0.25 && Math.abs(nx) > 0.4) x *= 1.08;
    if (ny < -0.6) y *= 0.94;

    var noseZone = Math.abs(nx) < 0.18 && ny > -0.35 && ny < 0.35;
    if (noseZone && z > 0) z += 42 * (1 - Math.abs(nx) / 0.18);

    var eyeL = (nx + 0.35) * (nx + 0.35) + (ny + 0.05) * (ny + 0.05);
    var eyeR = (nx - 0.35) * (nx - 0.35) + (ny + 0.05) * (ny + 0.05);
    if (eyeL < 0.055 || eyeR < 0.055) z -= 24;

    if (Math.abs(nx) < 0.3 && ny > 0.35 && ny < 0.6 && z > 0) z -= 12;

    points.push({
      x: CX + x,
      y: CY + y,
      z: z,
      u: j / LON,
      v: i / LAT
    });
  }
}

function idx(i, j) { return i * (LON + 1) + j; }

var tGlobal = 0;

var landmarkUVs = [
  { u: 0.38, v: 0.45 }, { u: 0.42, v: 0.46 }, { u: 0.36, v: 0.47 },
  { u: 0.62, v: 0.45 }, { u: 0.58, v: 0.46 }, { u: 0.64, v: 0.47 },
  { u: 0.50, v: 0.55 }, { u: 0.50, v: 0.50 },
  { u: 0.50, v: 0.30 },
  { u: 0.30, v: 0.55 }, { u: 0.70, v: 0.55 }
];

function drawFace() {
  fctx.clearRect(0, 0, FW, FH);

  var rot = Math.sin(tGlobal * 0.4) * 0.06;
  var cosR = Math.cos(rot), sinR = Math.sin(rot);

  var proj = points.map(function(p) {
    var dx = p.x - CX;
    var dz = p.z;
    var rx = dx * cosR + dz * sinR;
    var rz = -dx * sinR + dz * cosR;
    var persp = 420 / (420 + rz);
    return {
      x: CX + rx * persp,
      y: CY + (p.y - CY) * persp,
      z: rz,
      u: p.u,
      v: p.v
    };
  });

  fctx.lineWidth = 0.5;

  for (var i = 0; i < LAT; i++) {
    for (var j = 0; j < LON; j++) {
      var a = proj[idx(i, j)];
      var b = proj[idx(i, j + 1)];
      var c = proj[idx(i + 1, j)];

      var frontA = (a.z + RZ) / (2 * RZ);
      var frontB = (b.z + RZ) / (2 * RZ);
      var frontC = (c.z + RZ) / (2 * RZ);

      var alphaH = 0.08 + 0.32 * ((frontA + frontB) / 2);
      var alphaV = 0.08 + 0.32 * ((frontA + frontC) / 2);

      fctx.strokeStyle = 'rgba(66, 216, 255, ' + alphaH.toFixed(3) + ')';
      fctx.beginPath(); fctx.moveTo(a.x, a.y); fctx.lineTo(b.x, b.y); fctx.stroke();

      fctx.strokeStyle = 'rgba(66, 216, 255, ' + alphaV.toFixed(3) + ')';
      fctx.beginPath(); fctx.moveTo(a.x, a.y); fctx.lineTo(c.x, c.y); fctx.stroke();
    }
  }

  fctx.strokeStyle = 'rgba(140, 235, 255, 0.95)';
  fctx.lineWidth = 1.8;
  fctx.shadowColor = '#42d8ff';
  fctx.shadowBlur = 8;
  fctx.beginPath();
  var started = false;
  for (var i3 = 0; i3 <= LAT; i3++) {
    for (var j3 = 0; j3 <= LON; j3++) {
      var p3 = proj[idx(i3, j3)];
      var nx3 = (p3.x - CX) / RX;
      if (p3.z > 50 && Math.abs(nx3) > 0.55) {
        if (!started) { fctx.moveTo(p3.x, p3.y); started = true; }
        else fctx.lineTo(p3.x, p3.y);
      }
    }
  }
  fctx.stroke();
  fctx.shadowBlur = 0;

  fctx.strokeStyle = 'rgba(160, 240, 255, 0.9)';
  fctx.lineWidth = 1.4;
  fctx.shadowColor = '#42d8ff';
  fctx.shadowBlur = 6;

  fctx.beginPath();
  for (var k = 0; k < points.length; k++) {
    if (points[k].u > 0.30 && points[k].u < 0.44 &&
        points[k].v > 0.42 && points[k].v < 0.50) {
      var pL = proj[k];
      if (points[k].u === 0.30 || points[k].u === 0.44) {
        fctx.moveTo(pL.x, pL.y);
      } else {
        fctx.lineTo(pL.x, pL.y);
      }
    }
  }
  fctx.stroke();

  fctx.beginPath();
  for (var k2 = 0; k2 < points.length; k2++) {
    if (points[k2].u > 0.56 && points[k2].u < 0.70 &&
        points[k2].v > 0.42 && points[k2].v < 0.50) {
      var pR = proj[k2];
      if (points[k2].u === 0.56) {
        fctx.moveTo(pR.x, pR.y);
      } else {
        fctx.lineTo(pR.x, pR.y);
      }
    }
  }
  fctx.stroke();

  fctx.beginPath();
  for (var k3 = 0; k3 < points.length; k3++) {
    if (points[k3].u > 0.42 && points[k3].u < 0.58 &&
        points[k3].v > 0.42 && points[k3].v < 0.60) {
      var pN = proj[k3];
      if (points[k3].v === 0.42) {
        fctx.moveTo(pN.x, pN.y);
      } else {
        fctx.lineTo(pN.x, pN.y);
      }
    }
  }
  fctx.stroke();

  fctx.beginPath();
  for (var k4 = 0; k4 < points.length; k4++) {
    if (points[k4].u > 0.36 && points[k4].u < 0.64 &&
        points[k4].v > 0.66 && points[k4].v < 0.74) {
      var pM = proj[k4];
      if (points[k4].v === 0.66) {
        fctx.moveTo(pM.x, pM.y);
      } else {
        fctx.lineTo(pM.x, pM.y);
      }
    }
  }
  fctx.stroke();

  fctx.shadowBlur = 0;

  for (var li = 0; li < landmarkUVs.length; li++) {
    var lu = landmarkUVs[li].u;
    var lv = landmarkUVs[li].v;

    var closest = null, closestDist = 999;
    for (var pi = 0; pi < points.length; pi++) {
      var du = Math.abs(points[pi].u - lu);
      var dv = Math.abs(points[pi].v - lv);
      var d = du * du + dv * dv;
      if (d < closestDist) { closestDist = d; closest = pi; }
    }

    if (closest !== null) {
      var pp = proj[closest];
      if (pp.z > 30) {
        var pulse = 2.5 + Math.sin(tGlobal * 3 + li) * 1.2;
        var grd = fctx.createRadialGradient(pp.x, pp.y, 0, pp.x, pp.y, 12);
        grd.addColorStop(0, 'rgba(220, 250, 255, 1)');
        grd.addColorStop(0.4, 'rgba(66, 216, 255, 0.85)');
        grd.addColorStop(1, 'rgba(66, 216, 255, 0)');
        fctx.fillStyle = grd;
        fctx.beginPath(); fctx.arc(pp.x, pp.y, 12, 0, Math.PI * 2); fctx.fill();

        fctx.fillStyle = '#eaf8ff';
        fctx.beginPath(); fctx.arc(pp.x, pp.y, pulse, 0, Math.PI * 2); fctx.fill();
      }
    }
  }

  tGlobal += 0.02;
  requestAnimationFrame(drawFace);
}

drawFace();

var fpsEl = document.getElementById('fps');
var matchEl = document.getElementById('match');
var statusEl = document.getElementById('status');

var frameCount = 0, lastFps = performance.now();
var matchVal = 0, matchDir = 1;

var statuses = [
  "\u25CF INITIALIZING NEURAL MESH...",
  "\u25CF ACQUIRING FACE LANDMARKS...",
  "\u25CF EXTRACTING 512-D EMBEDDING...",
  "\u25CF COMPARING AGAINST DATABASE...",
  "\u2714 ENROLLMENT COMPLETE"
];

function hudLoop() {
  frameCount++;
  var now = performance.now();
  if (now - lastFps > 500) {
    fpsEl.textContent = Math.round(frameCount * 1000 / (now - lastFps));
    frameCount = 0; lastFps = now;
  }
  matchVal += matchDir * 0.8;
  if (matchVal > 99.7) matchDir = -1;
  if (matchVal < 90) matchDir = 1;
  matchEl.textContent = matchVal.toFixed(2) + '%';
  requestAnimationFrame(hudLoop);
}
hudLoop();

var si = 0;
var statusInterval = setInterval(function() {
  si = (si + 1) % statuses.length;
  statusEl.textContent = statuses[si];
}, __STATUS_INTERVAL__);

setTimeout(function() {
  clearInterval(statusInterval);
  document.getElementById('overlay').style.animation = 'fadeOut 0.4s ease forwards';
}, __DURATION_MS__);
</script>
</body>
</html>
"""


def show_face_scan_animation(duration=3.5, mode="enroll", person_name=""):
    """Renders an inline blue holographic face-scan animation."""
    if mode == "enroll" and person_name:
        status_msg = "ENROLLING " + person_name.upper()
    elif mode == "recognize":
        status_msg = "SCANNING BIOMETRIC DATA"
    else:
        status_msg = "PROCESSING BIOMETRIC DATA"

    duration_ms = int(duration * 1000)
    status_interval = max(1, int(duration_ms / 5))

    html = (
        FACE_SCAN_TEMPLATE
        .replace("__MODE__", mode.upper())
        .replace("__STATUS__", status_msg)
        .replace("__DURATION_MS__", str(duration_ms))
        .replace("__STATUS_INTERVAL__", str(status_interval))
    )

    components.html(html, height=440, scrolling=False)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model


# ============================================================
# DATABASE
# ============================================================

def load_database():
    if not os.path.exists(DATABASE_PATH):
        return {}
    try:
        with open(DATABASE_PATH, "rb") as file:
            data = pickle.load(file)
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if not isinstance(value, list):
                    data[key] = [value]
        return data
    except Exception:
        return {}


def save_database(database):
    folder = os.path.dirname(DATABASE_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(DATABASE_PATH, "wb") as file:
        pickle.dump(database, file)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(embedding1, embedding2):
    embedding1 = np.asarray(embedding1, dtype=np.float32).ravel()
    embedding2 = np.asarray(embedding2, dtype=np.float32).ravel()
    n1 = np.linalg.norm(embedding1)
    n2 = np.linalg.norm(embedding2)
    if n1 == 0 or n2 == 0:
        return -1.0
    return float(np.dot(embedding1 / n1, embedding2 / n2))


# ============================================================
# RECOGNITION
# ============================================================

def recognize_face(new_embedding, database):
    best_name = "UNKNOWN"
    best_similarity = -1.0
    for name, embeddings in database.items():
        stored = embeddings if isinstance(embeddings, list) else [embeddings]
        for stored_embedding in stored:
            try:
                similarity = cosine_similarity(new_embedding, stored_embedding)
            except Exception:
                continue
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name
    if best_similarity >= THRESHOLD:
        return best_name, best_similarity, True
    return "UNKNOWN", best_similarity, False


# ============================================================
# IMAGE HELPERS
# ============================================================

def resize_if_needed(pil_image, max_width=MAX_IMAGE_WIDTH):
    if pil_image.width <= max_width:
        return pil_image
    ratio = max_width / float(pil_image.width)
    new_size = (max_width, int(pil_image.height * ratio))
    return pil_image.resize(new_size, Image.LANCZOS)


def process_image(image, model):
    if isinstance(image, Image.Image):
        image_rgb = np.array(image.convert("RGB"))
    else:
        image_rgb = np.asarray(image)
    image_rgb = np.ascontiguousarray(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    faces = model.get(image_bgr)
    return image_rgb, faces


def draw_face_box(image, face, label=None, known=None):
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = np.ascontiguousarray(image)
    output = image.copy()

    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    height, width = output.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    color = (0, 220, 255)
    if known is True:  color = (0, 255, 120)
    elif known is False: color = (60, 60, 255)

    cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)

    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(0, y1 - th - 12)
        cv2.rectangle(output, (x1, top), (x1 + tw + 12, y1), color, -1)
        cv2.putText(output, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 15, 25), 2, cv2.LINE_AA)
    return output


def extract_single_embedding(image, model):
    _, faces = process_image(image, model)
    if len(faces) == 0:
        return None, "No face detected"
    if len(faces) > 1:
        return None, f"{len(faces)} faces detected — please use a single face"
    return np.asarray(faces[0].embedding, dtype=np.float32), None


# ============================================================
# LOAD RESOURCES
# ============================================================

model = load_model()
database = load_database()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            &#9673; AI Face <span>Recognition</span> System
        </div>
        <div class="hero-subtitle">
            &gt; Intelligent biometric identification using
            face detection and deep 512-D embeddings
        </div>
        <div class="status">&#9679; System Online</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

enroll_tab, recognition_tab = st.tabs(["👤  ENROLL", "🔍  RECOGNITION"])


# ============================================================
# ENROLLMENT TAB
# ============================================================

with enroll_tab:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">👤 Enroll New Person</div>
            <div class="card-description">
                &gt; Add face samples to the recognition database
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:

        name = st.text_input(
            "Person Name",
            placeholder="Example: Prasad",
            key="person_name"
        )

        enrollment_method = st.radio(
            "Input Method",
            ["📁 Upload Image", "🎥 Webcam"],
            horizontal=True,
            key="enrollment_method"
        )

        uploaded_files = None
        camera_image = None

        if enrollment_method == "📁 Upload Image":

            uploaded_files = st.file_uploader(
                "Upload face images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="enrollment_upload"
            )

            if uploaded_files:
                st.caption(f"📸 {len(uploaded_files)} image(s) selected")

            enroll_clicked = st.button(
                "🚀 ENROLL PERSON",
                type="primary",
                key="enroll_upload_button"
            )

            if enroll_clicked:
                if not name.strip():
                    st.error("Please enter a name.")
                elif not uploaded_files:
                    st.error("Please upload at least one image.")
                else:
                    st.session_state["scan_running"] = True
                    st.session_state["scan_mode"] = "enroll"
                    st.session_state["scan_person"] = name.strip()
                    st.session_state["scan_preview_rgb"] = None
                    st.session_state["scan_done"] = False
                    st.rerun()

        else:

            camera_image = st.camera_input(
                "Take a face photo",
                key="enrollment_camera"
            )

            save_clicked = st.button(
                "🚀 SAVE FACE",
                type="primary",
                key="enroll_camera_button"
            )

            if save_clicked:
                if not name.strip():
                    st.error("Please enter a name.")
                elif camera_image is None:
                    st.error("Please capture a photo first.")
                else:
                    st.session_state["scan_running"] = True
                    st.session_state["scan_mode"] = "enroll"
                    st.session_state["scan_person"] = name.strip()
                    st.session_state["scan_preview_rgb"] = None
                    st.session_state["scan_done"] = False
                    st.rerun()

    with right:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">📷 Face Preview</div>
                <div class="card-description">
                    &gt; Face scan & result will appear here
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if st.session_state.get("scan_running", False):

            show_face_scan_animation(
                duration=SCAN_DURATION,
                mode=st.session_state.get("scan_mode", "enroll"),
                person_name=st.session_state.get("scan_person", "")
            )

            time.sleep(SCAN_DURATION)

            enroll_ok = True
            saved_count = 0
            failed = []
            preview_rgb = None

            if enrollment_method == "📁 Upload Image":
                current_files = uploaded_files
                if current_files:
                    embeddings = []
                    for uploaded_file in current_files:
                        try:
                            image = Image.open(uploaded_file).convert("RGB")
                            image = resize_if_needed(image)
                            emb, err = extract_single_embedding(image, model)
                            if emb is None:
                                failed.append(f"{uploaded_file.name}: {err}")
                            else:
                                embeddings.append(emb)
                        except Exception as e:
                            failed.append(f"{uploaded_file.name}: {e}")

                    if embeddings:
                        pname = name.strip()
                        database.setdefault(pname, []).extend(embeddings)
                        save_database(database)
                        saved_count = len(embeddings)
                    else:
                        enroll_ok = False

                    try:
                        first_img = Image.open(current_files[0]).convert("RGB")
                        first_img = resize_if_needed(first_img)
                        first_rgb, first_faces = process_image(first_img, model)
                        if len(first_faces) >= 1:
                            preview_rgb = draw_face_box(
                                first_rgb, first_faces[0],
                                label="DETECTED", known=True
                            )
                        else:
                            preview_rgb = first_rgb
                    except Exception:
                        preview_rgb = None

            else:
                if camera_image is not None:
                    image = Image.open(camera_image).convert("RGB")
                    image = resize_if_needed(image)
                    emb, err = extract_single_embedding(image, model)
                    if emb is None:
                        failed.append(err)
                        enroll_ok = False
                    else:
                        pname = name.strip()
                        database.setdefault(pname, []).append(emb)
                        save_database(database)
                        saved_count = 1
                        try:
                            first_rgb, first_faces = process_image(image, model)
                            if len(first_faces) >= 1:
                                preview_rgb = draw_face_box(
                                    first_rgb, first_faces[0],
                                    label="DETECTED", known=True
                                )
                            else:
                                preview_rgb = first_rgb
                        except Exception:
                            preview_rgb = None

            st.session_state["scan_preview_rgb"] = preview_rgb
            st.session_state["scan_failed"] = failed
            st.session_state["scan_saved_count"] = saved_count
            st.session_state["scan_running"] = False
            st.session_state["scan_done"] = True
            st.rerun()

        elif st.session_state.get("scan_done", False):

            preview_rgb = st.session_state.get("scan_preview_rgb")

            if preview_rgb is not None:
                st.image(preview_rgb, width=420)
                st.markdown(
                    """
                    <div style="text-align:center;
                                color:#7fd8ff;
                                font-size:12px;
                                font-family:'Courier New',monospace;
                                letter-spacing:1.6px;
                                margin-top:6px;">
                        ✔ FACE CAPTURED &amp; ANALYZED
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            saved_count = st.session_state.get("scan_saved_count", 0)
            failed = st.session_state.get("scan_failed", [])
            person = st.session_state.get("scan_person", "")

            if saved_count > 0:
                st.success(f"✅ {person} enrolled successfully! ({saved_count} sample(s) saved)")
            else:
                st.error("❌ Enrollment failed — no valid face found.")

            if failed:
                st.warning("Some items were skipped:")
                for item in failed:
                    st.write(f"• {item}")

            if st.button("🧹 CLEAR", key="clear_scan_button"):
                for k in (
                    "scan_running", "scan_mode", "scan_person",
                    "scan_preview_rgb", "scan_failed",
                    "scan_saved_count", "scan_done"
                ):
                    st.session_state.pop(k, None)
                st.rerun()

        else:

            idle_image = None
            if enrollment_method == "📁 Upload Image" and uploaded_files:
                idle_image = Image.open(uploaded_files[0]).convert("RGB")
            elif enrollment_method == "🎥 Webcam" and camera_image is not None:
                idle_image = Image.open(camera_image).convert("RGB")

            if idle_image is not None:
                idle_image = resize_if_needed(idle_image)
                st.image(idle_image, width=420)
                st.markdown(
                    """
                    <div style="text-align:center;
                                color:#7fd8ff;
                                font-size:12px;
                                font-family:'Courier New',monospace;
                                letter-spacing:1.6px;
                                margin-top:6px;">
                        &gt; PRESS ENROLL TO SCAN
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="placeholder" style="
                        height:280px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        text-align:center;
                        border:1px dashed rgba(0,200,255,0.28);
                        border-radius:16px;
                        background:rgba(5,15,30,0.55);
                        color:#557086;
                        font-family:'Courier New',monospace;
                    ">
                        <div>
                            <div style="font-size:52px; margin-bottom:10px;">&#9673;</div>
                            <div style="font-size:15px; font-weight:700; letter-spacing:1.4px;">
                                AWAITING INPUT
                            </div>
                            <div style="font-size:12px; margin-top:6px; color:#7fd8ff;">
                                &gt; Upload an image or use webcam
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                """
                <div style="text-align:center; color:#718ba3;
                            font-size:11px; margin-top:12px;
                            font-family:'Courier New',monospace;
                            letter-spacing:1.6px;">
                    FACE DETECTION → EMBEDDING → DATABASE
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# RECOGNITION TAB
# ============================================================

with recognition_tab:

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔍 Identify Person</div>
            <div class="card-description">
                &gt; Compare a new face against enrolled identities
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:

        recognition_method = st.radio(
            "Input Method",
            ["📁 Upload Image", "🎥 Webcam"],
            horizontal=True,
            key="recognition_method"
        )

        if recognition_method == "📁 Upload Image":

            uploaded_file = st.file_uploader(
                "Upload face image",
                type=["jpg", "jpeg", "png"],
                key="recognition_upload"
            )

            recognize_clicked = st.button(
                "🔍 RECOGNIZE FACE",
                type="primary",
                key="recognize_upload_button"
            )

            if recognize_clicked:
                if not database:
                    st.error("No enrolled people found. Please enroll first.")
                elif uploaded_file is None:
                    st.error("Please upload an image.")
                else:
                    image = Image.open(uploaded_file).convert("RGB")
                    image = resize_if_needed(image)
                    image_rgb, faces = process_image(image, model)

                    if len(faces) == 0:
                        st.error("❌ No face detected.")
                    elif len(faces) > 1:
                        st.warning(f"⚠️ {len(faces)} faces detected.")
                        st.info("Please use an image containing one person.")
                    else:
                        face = faces[0]
                        name_result, similarity, known = recognize_face(
                            face.embedding, database
                        )
                        result_image = draw_face_box(
                            image_rgb, face,
                            label=name_result.upper(),
                            known=known
                        )
                        st.session_state["recognition_result_image"] = result_image
                        st.session_state["recognition_name"] = name_result
                        st.session_state["recognition_similarity"] = similarity
                        st.session_state["recognition_known"] = known

        else:

            camera_image = st.camera_input(
                "Capture face",
                key="recognition_camera"
            )

            recognize_cam_clicked = st.button(
                "🔍 RECOGNIZE CAPTURED FACE",
                type="primary",
                key="recognize_camera_button"
            )

            if recognize_cam_clicked:
                if not database:
                    st.error("No enrolled people found. Please enroll first.")
                elif camera_image is None:
                    st.error("Please capture a photo first.")
                else:
                    image = Image.open(camera_image).convert("RGB")
                    image = resize_if_needed(image)
                    image_rgb, faces = process_image(image, model)

                    if len(faces) == 0:
                        st.error("❌ No face detected.")
                    elif len(faces) > 1:
                        st.warning("⚠️ Multiple faces detected.")
                    else:
                        face = faces[0]
                        name_result, similarity, known = recognize_face(
                            face.embedding, database
                        )
                        result_image = draw_face_box(
                            image_rgb, face,
                            label=name_result.upper(),
                            known=known
                        )
                        st.session_state["recognition_result_image"] = result_image
                        st.session_state["recognition_name"] = name_result
                        st.session_state["recognition_similarity"] = similarity
                        st.session_state["recognition_known"] = known

        if "recognition_result_image" in st.session_state:
            if st.button("🧹 CLEAR RESULT", key="clear_result_button"):
                for key in (
                    "recognition_result_image",
                    "recognition_name",
                    "recognition_similarity",
                    "recognition_known",
                ):
                    st.session_state.pop(key, None)
                st.rerun()

    with right:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🎯 Recognition Result</div>
                <div class="card-description">
                    &gt; AI analysis and identity matching
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        if "recognition_result_image" in st.session_state:

            st.image(
                st.session_state["recognition_result_image"],
                width=420
            )

            name_result = st.session_state["recognition_name"]
            similarity = st.session_state["recognition_similarity"]
            known = st.session_state["recognition_known"]

            st.markdown("---")

            if known:
                st.markdown(
                    f"""
                    <div class="result-known">
                        <div class="result-title">🟢 {name_result.upper()}</div>
                        <div class="result-score">&gt; Identity verified — ACCESS GRANTED</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div class="result-unknown">
                        <div class="result-title">🔴 UNKNOWN</div>
                        <div class="result-score">
                            &gt; No enrolled identity matched — ACCESS DENIED
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            metric1, metric2 = st.columns(2)
            with metric1:
                st.metric("Similarity", f"{similarity:.4f}")
            with metric2:
                st.metric("Threshold", f"{THRESHOLD:.2f}")

            st.progress(float(min(max(similarity, 0.0), 1.0)))

            if known:
                st.success("Face successfully matched with the enrolled database.")
            else:
                st.warning("Similarity is below the matching threshold.")

        else:
            st.markdown(
                """
                <div class="placeholder" style="
                    height:280px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    border:1px dashed rgba(0,200,255,0.28);
                    border-radius:16px;
                    background:rgba(5,15,30,0.55);
                    color:#557086;
                    font-family:'Courier New',monospace;
                ">
                    <div>
                        <div style="font-size:52px; margin-bottom:10px;">&#9673;</div>
                        <div style="font-size:15px; font-weight:700; letter-spacing:1.4px;">
                            WAITING FOR FACE
                        </div>
                        <div style="font-size:12px; margin-top:6px; color:#7fd8ff;">
                            &gt; Upload an image or use webcam
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        &#9673; AI Face Recognition System
        &nbsp; • &nbsp;
        InsightFace
        &nbsp; • &nbsp;
        ArcFace Embeddings
        &nbsp; • &nbsp;
        Cosine Similarity
    </div>
    """,
    unsafe_allow_html=True
)