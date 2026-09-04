# P0 Verification Report: Adhi's NST

**Date:** 2026-09-04  
**Environment:** Python 3.12.1 | PyTorch 2.2.2+cpu | Flask 3.1.2 | Pillow 12.0.0  
**Device:** CPU (no CUDA GPU available)  

---

## Verification Checklist

| # | Check | Status | Evidence |
|---|-------|--------|----------|
| 1 | Repository Audited | ✅ PASS | `docs/PROJECT_AUDIT.md` created |
| 2 | Environment Installs | ✅ PASS | `pip install -r requirements.txt` — all 17 packages installed |
| 3 | `pip check` succeeds | ✅ PASS | `No broken requirements found.` |
| 4 | Python 3.12.1 venv created | ✅ PASS | `.venv/` created with `py -3.12` |
| 5 | VGG Weights file exists | ✅ PASS | `NST_Code/vgg_normalised.pth` — 80.1 MB |
| 6 | Decoder Weights file exists | ✅ PASS | `NST_Code/experiment/final_exp/decoder_final.pth` — 14.0 MB |
| 7 | VGG Loads (state_dict) | ✅ PASS | `scripts/verify_model.py` → `VGG: LOADED` |
| 8 | Decoder Loads (state_dict) | ✅ PASS | `scripts/verify_model.py` → `Decoder: LOADED` |
| 9 | Models set to eval() | ✅ PASS | Both `encoder.eval()` and `decoder.eval()` called |
| 10 | Demo Content Image loads | ✅ PASS | `Demo_IO_Images/i-p/i_p image.jpg` (512×512) |
| 11 | Demo Style 1 image loads | ✅ PASS | `Demo_IO_Images/i-p/style 1.png` (300×300) |
| 12 | Demo Style 2 image loads | ✅ PASS | `Demo_IO_Images/i-p/style 2.jpg` (1179×1536) |
| 13 | AdaIN math verified | ✅ PASS | Mean error: 3.86e-08, Std error: 1.07e-06 |
| 14 | AdaIN tensor shapes | ✅ PASS | Output shape `[2, 512, 32, 32]` matches input |
| 15 | Style 1 Inference (real model) | ✅ PASS | `artifacts/verification/style1_result.png` — 465 KB — 1490ms |
| 16 | Style 2 Inference (real model) | ✅ PASS | `artifacts/verification/style2_result.png` — 415 KB — 1408ms |
| 17 | Alpha 0.0 works | ✅ PASS | `artifacts/verification/alpha_0.png` — 376 KB |
| 18 | Alpha 0.5 works | ✅ PASS | `artifacts/verification/alpha_05.png` — 407 KB |
| 19 | Alpha 1.0 works | ✅ PASS | `artifacts/verification/alpha_1.png` — 465 KB |
| 20 | Invalid alpha -0.1 rejected | ✅ PASS | `ValueError` raised correctly |
| 21 | Invalid alpha 1.5 rejected | ✅ PASS | `ValueError` raised correctly |
| 22 | Machine-specific paths removed | ✅ PASS | `/home/ubuntu/Desktop/...` replaced with `pathlib` |
| 23 | Hardcoded secret removed | ✅ PASS | `SECRET_KEY` reads from env var |
| 24 | Debug mode removed (production) | ✅ PASS | Debug only enabled when `APP_ENV=development` |
| 25 | `.env.example` created | ✅ PASS | All env vars documented |
| 26 | `.gitignore` updated | ✅ PASS | `.env`, `.venv/`, `myenv/` added |
| 27 | Models load once on startup | ✅ PASS | `encoder` and `decoder` are module-level globals |
| 28 | Flask starts | ✅ PASS | Running at `http://127.0.0.1:5000/` |
| 29 | Flask homepage loads | ✅ PASS | HTTP 200 OK on GET / |
| 30 | Missing inputs handled | ✅ PASS | "Please select or upload a Content Image." shown |
| 31 | Invalid image file caught | ✅ PASS | "corrupted or not a valid image" message shown |
| 32 | Alpha out-of-range rejected | ✅ PASS | "Style Strength (Alpha) must be between 0.0 and 1.0" |
| 33 | Real inference from Flask POST | ✅ PASS | `stylized_e4629de8.png` — 439 KB saved to uploads |
| 34 | Result file verified on disk | ✅ PASS | File exists, non-zero size (439,886 bytes) |
| 35 | Download route works | ✅ PASS | HTTP 200 with `Content-Disposition: attachment` |
| 36 | Unique random output filenames | ✅ PASS | UUID-based names (`stylized_xxxxxxxx.png`) |
| 37 | Image content validated (PIL) | ✅ PASS | `img.verify()` used, not extension alone |

---

## Model Verification Output
```
Device: CPU
VGG: LOADED
Decoder: LOADED
Model verification: SUCCESS
```

## Inference Verification Output
```
Running inference verification on cpu...
--- Testing AdaIN and calc_mean_std ---
AdaIN shape verification: PASS [Shape: [2, 512, 32, 32]]
AdaIN mean transfer error: 3.86e-08 (PASS)
AdaIN std transfer error: 1.07e-06 (PASS)
AdaIN mathematical verification: SUCCESS

Loaded demo content image: i_p image.jpg ((512, 512))
Loaded demo style 1 image: style 1.png ((300, 300))
Loaded demo style 2 image: style 2.jpg ((1179, 1536))

--- Generating Style 1 Transfer (alpha=1.0) ---
Saved: artifacts/verification/style1_result.png [Latency: 1490.2ms, Size: (512, 512)]

--- Generating Style 2 Transfer (alpha=1.0) ---
Saved: artifacts/verification/style2_result.png [Latency: 1408.2ms, Size: (512, 512)]

--- Testing Alpha Interpolation ---
Alpha 0.0: Saved alpha_0.png [Latency: 1423.4ms]
Alpha 0.5: Saved alpha_05.png [Latency: 1411.8ms]
Alpha 1.0: Saved alpha_1.png [Latency: 1459.6ms]
Alpha -0.1 correctly rejected: PASS
Alpha 1.5 correctly rejected: PASS

ALL INFERENCE AND MATHEMATICAL TESTS PASSED!
```

## Flask End-to-End Test Suite Output
```
Loading models on device: cpu
Neural Style Transfer models loaded successfully.

[Test 1] Testing GET / ...
PASS: Homepage loaded cleanly without errors.

[Test 2] Testing POST without files (missing inputs)...
PASS: Missing inputs correctly caught and error banner displayed.

[Test 3] Testing POST with corrupted/invalid image file...
PASS: Corrupted/invalid image file correctly rejected.

[Test 4] Testing invalid alpha values...
PASS: Out-of-bounds alpha value rejected.

[Test 5] Testing valid style transfer via Flask POST...
PASS: Style transfer executed. Result filename: stylized_e4629de8.png
PASS: Result file verified on disk (439886 bytes).

[Test 6] Testing GET /download/<filename>...
PASS: Download route returned 200 with attachment disposition.

ALL FLASK TEST SUITE TESTS PASSED!
```

## Generated Artifacts
| File | Size | Description |
|------|------|-------------|
| `artifacts/verification/style1_result.png` | 465 KB | Content + Style 1 at alpha=1.0 |
| `artifacts/verification/style2_result.png` | 415 KB | Content + Style 2 at alpha=1.0 |
| `artifacts/verification/alpha_0.png` | 376 KB | Style 1 at alpha=0.0 (near-original) |
| `artifacts/verification/alpha_05.png` | 407 KB | Style 1 at alpha=0.5 (blended) |
| `artifacts/verification/alpha_1.png` | 465 KB | Style 1 at alpha=1.0 (full style) |
| `NST_Code/static/uploads/stylized_e4629de8.png` | 430 KB | Flask-generated inference result |

## Files Created / Modified
| File | Action |
|------|--------|
| `.venv/` | CREATED — Python 3.12 virtual environment |
| `.env.example` | CREATED — Environment configuration template |
| `.gitignore` | MODIFIED — Added `.env`, `.venv/`, `myenv/`, uploads |
| `docs/ENVIRONMENT.md` | CREATED — Documented exact verified package versions |
| `docs/PROJECT_AUDIT.md` | EXISTS — Audit already completed |
| `NST_Code/app.py` | MODIFIED — Pathlib paths, env config, validation, security |
| `NST_Code/train.py` | MODIFIED — Replaced hardcoded defaults with pathlib |
| `NST_Code/templates/index.html` | MODIFIED — Download button, alpha badge visible |
| `scripts/verify_model.py` | CREATED — Model loading verification script |
| `scripts/test_inference.py` | CREATED — Full AdaIN + inference + alpha test |
| `scripts/test_flask_app.py` | CREATED — Automated Flask end-to-end test suite |

---

## Known Issues / Limitations
- **No CUDA**: Running on CPU only. Inference ~1,400ms per image. GPU would be 5-10x faster.
- **Flask_Bootstrap 3.3.7.1** is a legacy package, but functional.
- **Example gallery** in UI uses hardcoded filenames (`brad_pitt.jpg`, `sketch.png`) — images must be present in `NST_Code/examples/`.
- **Single-worker Flask** — not suitable for concurrent users in production without Gunicorn multi-worker config.

## P0 STATUS: COMPLETE ✅
**All 37 P0 checks pass. No fake outputs. No mocked inference.**
