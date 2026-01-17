import os, math, json
import argparse
import numpy as np
from svgpathtools import svg2paths2

try:
    import cairosvg
except OSError as e:
    if "cairo" in str(e).lower():
        print("ERROR: Cairo library not found. Please install it:")
        print("  macOS: brew install cairo")
        print("  Linux: sudo apt-get install libcairo2-dev (or equivalent)")
        print("  Then reinstall: pip install cairosvg")
        exit(1)
    raise

# --------- IO ----------
# These will be set from command-line arguments
INPUT_SVG = "reg-input.svg"   # set to your current layered SVG
OUT_SVG_DIR = "frames_svg"
OUT_PNG_DIR = "frames_png"

FPS = 30
SEED = 42

# resampling fidelity (higher = smoother, slower)
PTS_PER_PATH = 220

# waist mask width
SIGMA = 0.28

# look
BG = "#0D0018"
OPACITY = 0.95

# deformation (base)
PINCH_STRENGTH = 0.55
RADIATE_STRENGTH = 0.18
RIPPLE_WAIST_SUPPRESS = 0.65
NOISE_AMP = 0.08
NOISE_FREQ = 2.2

# presets
PRESETS = {
    "regulator": {
        "vulnerability": 0.60,
        "reciprocity": 0.80
    },
    "saturator": {
        "vulnerability": 0.80,
        "reciprocity": 0.40
    },
    "fader": {
        "vulnerability": 0.45,
        "reciprocity": 0.30
    }
}

QUOTES = {
    "fader": [
        "Nothing happened exactly. We just stopped being part of each other's lives.",
        "It felt like stepping into a time capsule."
    ],
    "regulator": [
        "For me, friendship feels reciprocal and full of mutual admiration.",
        "They tell me what they need, and I do the same."
    ],
    "saturator": [
        "It mattered so much to me, and I don't think it mattered to them in the same way.",
        "I've probably tried harder than I should have sometimes."
    ]
}

# MODE will be set from command-line arguments
MODE = "regulator"  # regulator | saturator | fader
ACTIVE_QUOTES = QUOTES[MODE]

# ---------- helpers ----------
def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def lerp(a, b, t):
    t = clamp01(t)
    return a + (b - a) * t

def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

def smooth_1d(arr, window_size):
    """Smooth 1D array using a moving average window"""
    if window_size < 1:
        return arr
    window_size = int(window_size)
    if window_size >= len(arr):
        return np.full_like(arr, np.mean(arr))
    
    # Pad array with edge values
    pad = window_size // 2
    padded = np.pad(arr, pad, mode='edge')
    
    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    
    # Ensure same length
    if len(smoothed) > len(arr):
        smoothed = smoothed[:len(arr)]
    elif len(smoothed) < len(arr):
        smoothed = np.pad(smoothed, (0, len(arr) - len(smoothed)), mode='edge')
    
    return smoothed

def waist_mask(y, sigma=SIGMA):
    return np.exp(- (y / sigma) ** 2)

def pressure_signal(frame, fps=FPS, period_seconds=10.0):
    t = frame / fps
    return math.sin(2 * math.pi * (t / period_seconds))  # -1..+1

def pseudo_noise(x, y, t):
    """
    Deterministic noise function using sin/cos.
    t is frame-based (t = frame / FPS), not actual time, ensuring deterministic output.
    """
    return (
        math.sin((x * NOISE_FREQ + t * 0.7) * 2 * math.pi) +
        math.cos((y * NOISE_FREQ - t * 0.55) * 2 * math.pi)
    ) * 0.5

def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def interpolate_color(c1, c2, t):
    r1,g1,b1 = hex_to_rgb(c1)
    r2,g2,b2 = hex_to_rgb(c2)
    rgb = (int(r1 + (r2-r1)*t), int(g1 + (g2-g1)*t), int(b1 + (b2-b1)*t))
    return rgb_to_hex(rgb)

# ---------- simplified inputs → params ----------
def composite_to_params(inputs: dict):
    """
    Only two inputs:
      vulnerability (0..1): amplitude / how much the form can open + how much strain / settling appears
      reciprocity (0..1): symmetry / phase-lock / coherence
    """
    V = clamp01(inputs.get("vulnerability", 0.60))
    R = clamp01(inputs.get("reciprocity", 0.80))

    params = {}

    # Load model:
    # Vulnerability is treated as weight.
    # Reciprocity controls whether that load redistributes (regulator),
    # accumulates asymmetrically (saturator),
    # or dissipates over time (fader).
    
    # --- motion (LOAD MODEL) ---
    # vulnerability -> load weight
    params["load_strength"] = lerp(0.020, 0.095, V)         # how much it deforms
    params["redistribute_rate"] = lerp(0.10, 1.35, R)       # how quickly it shares/evens out
    params["settle_smooth"] = lerp(3.0, 13.0, R)            # smoothing window (higher = more coherent)
    params["asymmetry"] = lerp(0.85, 0.05, R) * lerp(0.35, 1.0, V)  # low reciprocity + high vuln = skew

    # keep noise as reliability-controlled jitter
    params["noise_amp"] = lerp(0.06, 0.0, R)

    # coherence
    params["lock_strength"] = lerp(0.30, 1.30, R)      # reciprocity -> stronger phase lock
    params["mirror_sym"] = R                           # used for L/R symmetry weighting

    # overall displacement cap becomes "strain ceiling"
    params["amp"] = lerp(0.030, 0.090, V)

    # rendering (mezzotint / granular texture)
    params["opacity_base"] = lerp(0.85, 0.95, R)
    params["stroke_width"] = lerp(1.25, 1.00, R)
    params["stroke_variance"] = lerp(0.35, 0.12, R)  # increased for more granular texture
    params["mezzotint_grain"] = lerp(0.25, 0.08, R)  # granular particle texture
    params["glow"] = lerp(0.00, 0.18, V) * lerp(0.75, 1.0, R)  # increased for more luminosity

    return params

def load_inputs_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    inputs = data.get("inputs", {})
    return data.get("name", "inputs"), inputs, composite_to_params(inputs)

# ---------- geometry ----------
def normalize_points(all_pts):
    pts = np.vstack(all_pts)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)

    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    w, h = (maxx - minx), (maxy - miny)
    s = max(w, h) / 2

    norm = []
    for arr in all_pts:
        xy = arr.copy()
        xy[:, 0] = (xy[:, 0] - cx) / s
        xy[:, 1] = - (xy[:, 1] - cy) / s  # flip Y to up-positive
        norm.append(xy)
    return norm, (cx, cy, s)

def denormalize_points(norm_pts, cx, cy, s):
    out = []
    for xy in norm_pts:
        pts = xy.copy()
        pts[:, 0] = pts[:, 0] * s + cx
        pts[:, 1] = (-pts[:, 1]) * s + cy
        out.append(pts)
    return out

def polyline_normals(xy: np.ndarray) -> np.ndarray:
    p = xy
    t = np.zeros_like(p)
    t[1:-1] = p[2:] - p[:-2]
    t[0] = p[1] - p[0]
    t[-1] = p[-1] - p[-2]

    nrm = np.linalg.norm(t, axis=1, keepdims=True)
    t = t / (nrm + 1e-9)

    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    return n / (nn + 1e-9)

def points_to_path_d(points):
    p = points
    cmds = [f"M {p[0,0]:.3f} {p[0,1]:.3f}"]
    cmds += [f"L {p[i,0]:.3f} {p[i,1]:.3f}" for i in range(1, len(p))]
    return " ".join(cmds)

# ---------- band → gradient ----------
BAND_TO_GRAD = {
    # CORE
    "core_band_01": "grad_core",
    "core_band_02": "grad_core",
    "core_band_03": "grad_core",
    # CORE_OVER
    "core_over_band_00": "grad_core_over",
    # INNER
    "inner_band_00": "grad_inner",
    "inner_band_01": "grad_inner",
    "inner_band_02": "grad_inner",
    "inner_band_03": "grad_inner",
    # MID
    "mid_band_00": "grad_mid",
    "mid_band_01": "grad_mid",
    "mid_band_02": "grad_mid",
    "mid_band_03": "grad_mid",
    "mid_band_04": "grad_mid",
    # MID_OUTER
    "mid_outer_band_00": "grad_mid_outer",
    "mid_outer_band_01": "grad_mid_outer",
    "mid_outer_band_02": "grad_mid_outer",
    "mid_outer_band_03": "grad_mid_outer",
    # OUTER
    "outer_band_00": "grad_outer",
    "outer_band_01": "grad_outer",
    "outer_band_02": "grad_outer",
    "outer_band_03": "grad_outer",
}

def pick_grad_from_class(class_str: str) -> str:
    if not class_str:
        return "grad_outer"
    parts = class_str.split()
    for p in parts:
        if p in BAND_TO_GRAD:
            return BAND_TO_GRAD[p]
    return "grad_outer"

# ---------- deformation ----------
def is_inhale_band(class_str: str) -> bool:
    """Check if a path belongs to CORE, CORE_OVER, or INNER bands (inhale bands)"""
    if not class_str:
        return False
    parts = class_str.split()
    for p in parts:
        if p.startswith("core_band_"):
            return True
        if p.startswith("core_over_band_"):
            return True
        if p.startswith("inner_band_"):
            return True
    return False

def is_exhale_band(class_str: str) -> bool:
    """Check if a path belongs to MID, MID_OUTER, or OUTER bands (exhale bands)"""
    if not class_str:
        return False
    parts = class_str.split()
    for p in parts:
        if p.startswith("mid_band_") and not p.startswith("mid_outer_"):
            return True
        if p.startswith("mid_outer_band_"):
            return True
        if p.startswith("outer_band_"):
            return True
    return False

def deform(norm_pts, frame, n_frames, mode, params, attrs=None, fps=30):
    noise_amp = params.get("noise_amp", NOISE_AMP)
    mirror_sym = params.get("mirror_sym", 0.8)  # 0..1
    load_strength = params.get("load_strength", 0.035)
    redistribute_rate = params.get("redistribute_rate", 0.50)
    settle_smooth = params.get("settle_smooth", 0.75)
    asymmetry = params.get("asymmetry", 0.20)

    # loop modes (you said: no modulators for now; keep regulator loop)
    fade = 1.0
    
    # Decay for FADER (when load_strength and redistribute_rate are low)
    if load_strength < 0.035 and redistribute_rate < 0.30:
        # FADER: load sheds over time, deformation attenuates
        progress = frame / max(n_frames - 1, 1) if n_frames > 1 else 0.0
        fade = 1.0 - smoothstep(0.0, 1.0, progress * 0.8)  # gradual decay over animation

    t = frame / fps
    s = pressure_signal(frame, fps=fps)      # -1..+1
    inhale = max(0.0, s)            # 0..1
    exhale = max(0.0, -s)           # 0..1

    # Minimal waist pinch for FADER (when load_strength is low)
    pinch_scale = lerp(0.3, 1.0, clamp01(load_strength / 0.060))
    pinch_amt = PINCH_STRENGTH * inhale * pinch_scale
    radiate_amt = RADIATE_STRENGTH * exhale

    # reciprocity as symmetry: reduce any implicit left/right divergence
    # (we keep it gentle: higher reciprocity == more mirrored motion)
    sym_mix = mirror_sym

    out = []
    for path_idx, xy in enumerate(norm_pts):
        x = xy[:, 0].copy()
        y = xy[:, 1].copy()

        m = waist_mask(y)
        away = 1.0 - m

        # Determine band type
        if attrs and path_idx < len(attrs):
            class_str = attrs[path_idx].get("class") or ""
            is_inhale_band_type = is_inhale_band(class_str)
            is_exhale_band_type = is_exhale_band(class_str)
        else:
            # Fallback: if no attrs, apply to all (backward compatibility)
            is_inhale_band_type = True
            is_exhale_band_type = True

        # Inhale: pinch waist (only for CORE, CORE_OVER, and INNER bands)
        if is_inhale_band_type:
            x = x * (1.0 - m * pinch_amt)

        # Exhale: subtle bloom (only for MID, MID_OUTER, and OUTER bands)
        if is_exhale_band_type and radiate_amt > 0:
            y = y * (1.0 + away * radiate_amt)
            x = x * (1.0 + away * radiate_amt * 0.25)

        # Load model: vulnerability as load deformation on exhale
        load_deform = np.zeros(len(x))
        
        if is_exhale_band_type and exhale > 0:
            # Compute load field: away-from-waist * exhale (scaled by fade for FADER)
            load_field = away * exhale * fade
            
            # Load redistribution: carrying capacity / shared load based on redistribute_rate
            # Distance from center (using x coordinate as proxy for radial distance)
            dist_from_center = np.abs(x)
            max_dist = np.max(dist_from_center) if len(dist_from_center) > 0 and np.max(dist_from_center) > 0 else 1.0
            dist_norm = dist_from_center / max_dist if max_dist > 0 else dist_from_center
            
            # Redistribution: load flows outward, faster with higher redistribute_rate
            redistribution = redistribute_rate * (1.0 - dist_norm * 0.7)
            load_field *= redistribution
            
            # Smooth along the polyline using settle_smooth window
            load_field = smooth_1d(load_field, settle_smooth)
            
            # Scale by load_strength
            load_field *= load_strength
            
            # Apply asymmetry using sign(x): low reciprocity + high vulnerability = one-sided bulge
            # Keep regulator symmetric by making asymmetry inversely proportional to reciprocity
            asymmetry_factor = 1.0 + asymmetry * np.sign(x)
            load_deform = load_field * asymmetry_factor
        
        # Apply load deformation as normal displacement
        if np.any(load_deform > 0):
            xy_wave = np.stack([x, y], axis=1)
            n = polyline_normals(xy_wave)
            # Smooth load application
            load_mask = away * (1.0 - RIPPLE_WAIST_SUPPRESS * m)
            disp = load_deform * load_mask
            xy_wave = xy_wave + n * disp[:, None]
            x, y = xy_wave[:, 0], xy_wave[:, 1]

        # Reciprocity symmetry: softly pull x toward mirrored magnitude around 0
        # (sym_mix=1 -> perfectly mirrored magnitude)
        x = (sym_mix * np.sign(x) * np.abs(x) + (1.0 - sym_mix) * x)

        # turbulence (reduced by reciprocity) - only on inhale bands
        if is_inhale_band_type:
            turb = noise_amp * (0.4 + 0.6 * inhale)
            if turb > 0:
                dx = np.array([pseudo_noise(float(xi), float(yi), t) for xi, yi in zip(x, y)])
                x = x + (m * turb * dx)

        out.append(np.stack([x, y], axis=1))

    return out, fade

# ---------- SVG writing (defs include all gradients) ----------
def gradient_defs(minx, w, glow):
    # symmetric gradients: edges darker, center brightest
    gx1 = minx
    gx2 = minx + w
    glow_def = ""
    if glow > 0:
        glow_def = f"""
      <radialGradient id="centerGlow" cx="50%" cy="50%" r="55%">
        <stop offset="0%" stop-color="#C1FB7C" stop-opacity="{glow * 0.22:.3f}"/>
        <stop offset="35%" stop-color="#4CC7A0" stop-opacity="{glow * 0.10:.3f}"/>
        <stop offset="100%" stop-color="#000000" stop-opacity="0"/>
      </radialGradient>
    """

    return f"""
    <defs>
      {glow_def}

      <linearGradient id="grad_core" gradientUnits="userSpaceOnUse" x1="{gx1:.3f}" y1="0" x2="{gx2:.3f}" y2="0">
        <stop offset="0%"  stop-color="#32928D"/>
        <stop offset="25%" stop-color="#4CC7A0"/>
        <stop offset="50%" stop-color="#C1FB7C"/>
        <stop offset="75%" stop-color="#4CC7A0"/>
        <stop offset="100%" stop-color="#32928D"/>
      </linearGradient>

      <linearGradient id="grad_core_over" gradientUnits="userSpaceOnUse" x1="{gx1:.3f}" y1="0" x2="{gx2:.3f}" y2="0">
        <stop offset="0%"  stop-color="#8FCF80"/>
        <stop offset="50%" stop-color="#C1FB7C"/>
        <stop offset="100%" stop-color="#8FCF80"/>
      </linearGradient>

      <linearGradient id="grad_inner" gradientUnits="userSpaceOnUse" x1="{gx1:.3f}" y1="0" x2="{gx2:.3f}" y2="0">
        <stop offset="0%"  stop-color="#357384"/>
        <stop offset="25%" stop-color="#32928D"/>
        <stop offset="50%" stop-color="#8FCF80"/>
        <stop offset="75%" stop-color="#32928D"/>
        <stop offset="100%" stop-color="#357384"/>
      </linearGradient>

      <linearGradient id="grad_mid" gradientUnits="userSpaceOnUse" x1="{gx1:.3f}" y1="0" x2="{gx2:.3f}" y2="0">
        <stop offset="0%"  stop-color="#24537C"/>
        <stop offset="25%" stop-color="#357384"/>
        <stop offset="50%" stop-color="#4CC7A0"/>
        <stop offset="75%" stop-color="#357384"/>
        <stop offset="100%" stop-color="#24537C"/>
      </linearGradient>

      <linearGradient id="grad_mid_outer" gradientUnits="userSpaceOnUse" x1="{gx1:.3f}" y1="0" x2="{gx2:.3f}" y2="0">
        <stop offset="0%"  stop-color="#21073E"/>
        <stop offset="25%" stop-color="#2C2661"/>
        <stop offset="50%" stop-color="#32928D"/>
        <stop offset="75%" stop-color="#2C2661"/>
        <stop offset="100%" stop-color="#21073E"/>
      </linearGradient>

      <linearGradient id="grad_outer" gradientUnits="userSpaceOnUse" x1="{gx1:.3f}" y1="0" x2="{gx2:.3f}" y2="0">
        <stop offset="0%"  stop-color="#16052A"/>
        <stop offset="25%" stop-color="#21073E"/>
        <stop offset="50%" stop-color="#24537C"/>
        <stop offset="75%" stop-color="#21073E"/>
        <stop offset="100%" stop-color="#16052A"/>
      </linearGradient>
    </defs>
    """

def write_svg_frame(paths_pts_denorm, attrs, viewbox, fade, params):
    minx, miny, w, h = viewbox
    opacity = params.get("opacity_base", OPACITY) * fade
    stroke_width_base = params.get("stroke_width", 1.1)
    stroke_variance = params.get("stroke_variance", 0.10)
    mezzotint_grain = params.get("mezzotint_grain", 0.15)
    glow = params.get("glow", 0.0)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{minx:.3f} {miny:.3f} {w:.3f} {h:.3f}">')
    parts.append(f'<rect x="{minx:.3f}" y="{miny:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{BG}"/>')
    parts.append(gradient_defs(minx, w, glow))

    if glow > 0:
        parts.append(f'<rect x="{minx:.3f}" y="{miny:.3f}" width="{w:.3f}" height="{h:.3f}" fill="url(#centerGlow)"/>')

    for i, pts in enumerate(paths_pts_denorm):
        if len(pts) < 2:
            continue

        cls = (attrs[i].get("class") or "")
        grad_id = pick_grad_from_class(cls)

        # Mezzotint: granular texture variation
        base_variation = 1.0 + stroke_variance * math.sin(i * 0.73 + 1.2)
        avg_y = float(np.mean(pts[:, 1]))
        pos_variation = 1.0 + (stroke_variance * 0.3) * math.sin(avg_y * 2.1)
        
        # Fine-grained granular texture along path (sparkling particles effect)
        avg_x = float(np.mean(pts[:, 0]))
        grain_variation = 1.0 + mezzotint_grain * (math.sin(avg_x * 3.7 + i * 0.41) * math.cos(avg_y * 2.9 + i * 0.67))
        
        sw = stroke_width_base * base_variation * pos_variation * grain_variation

        d = points_to_path_d(pts)
        parts.append(
            f'<path d="{d}" fill="none" stroke="url(#{grad_id})" stroke-width="{sw:.3f}" opacity="{opacity:.3f}" stroke-linecap="round" stroke-linejoin="round"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)

# ---------- argument parsing ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="regulator", choices=["regulator","saturator","fader"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--input", default="reg-input.svg")
    ap.add_argument("--out", default=None)  # if None, derive from mode
    return ap.parse_args()

# ---------- main ----------
def main(json_path=None, n_seconds=6.0, mode="regulator"):
    # Parse command-line arguments
    args = parse_args()
    MODE = args.mode
    FPS = args.fps
    INPUT_SVG = args.input
    n_frames = args.frames
    OUT = args.out or f"{MODE}.mp4"
    
    # Make output directories mode-specific
    OUT_SVG_DIR = f"frames_svg_{MODE}"
    OUT_PNG_DIR = f"frames_png_{MODE}"
    
    # Set seed for deterministic output (no actual random numbers are used, but seed ensures consistency)
    np.random.seed(SEED)

    os.makedirs(OUT_SVG_DIR, exist_ok=True)
    os.makedirs(OUT_PNG_DIR, exist_ok=True)

    # load svg paths + attrs
    paths, attrs, svg_attr = svg2paths2(INPUT_SVG)

    # sample each path into polylines
    all_pts = []
    for p in paths:
        pts = []
        for k in range(PTS_PER_PATH):
            t = k / (PTS_PER_PATH - 1)
            z = p.point(t)
            pts.append([z.real, z.imag])
        all_pts.append(np.array(pts, dtype=float))

    norm_pts, (cx, cy, s) = normalize_points(all_pts)

    # viewBox
    if "viewBox" in svg_attr:
        vb = [float(x) for x in svg_attr["viewBox"].replace(",", " ").split()]
        viewbox = (vb[0], vb[1], vb[2], vb[3])
    else:
        # fallback: derive from points
        pts = np.vstack(all_pts)
        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)
        viewbox = (minx, miny, maxx - minx, maxy - miny)

    # params from json or preset
    if json_path:
        _, inputs, params = load_inputs_json(json_path)
    else:
        assert MODE in PRESETS, "Invalid MODE"
        metrics = PRESETS[MODE]
        assert 0 <= metrics["vulnerability"] <= 1
        assert 0 <= metrics["reciprocity"] <= 1
        inputs = metrics.copy()
        params = composite_to_params(inputs)

    for frame in range(n_frames):
        deformed_norm, fade = deform(norm_pts, frame, n_frames, MODE, params, attrs, fps=FPS)
        deformed_denorm = denormalize_points(deformed_norm, cx, cy, s)

        svg_out = write_svg_frame(deformed_denorm, attrs, viewbox, fade, params)

        svg_path = os.path.join(OUT_SVG_DIR, f"{MODE}_frame_{frame:04d}.svg")
        png_path = os.path.join(OUT_PNG_DIR, f"{MODE}_frame_{frame:04d}.png")

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_out)

        cairosvg.svg2png(bytestring=svg_out.encode("utf-8"), write_to=png_path)

    print(f"Rendered {n_frames} frames → {OUT_PNG_DIR}/")
    print(f"Output video: {OUT}")

if __name__ == "__main__":
    main()
