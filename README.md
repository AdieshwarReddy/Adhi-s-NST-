# ai-nst-project

Deployed link :   https://adhisnst.streamlit.app/
AdaIN Neural Style Transfer (NST) Web App
==========================================

Description
-----------
A lightweight Flask‑based web application that applies arbitrary artistic
styles to any image using the Adaptive Instance Normalization (AdaIN) algorithm
(Huang & Belongie, 2017). The app runs locally without external services and
includes sample style images.

Features
--------
* AdaIN style transfer – no GANs, fast inference
* Simple HTML UI (templates/index.html)
* Dockerfile for one‑click containerised run
* Sample content and style images in Demo_IO_Images/
* Stateless – no database required
* Ready to deploy on Render, Railway, Azure App Service, Fly.io, etc.

Quick Start (Local)
--------------------
Prerequisites
* Python 3.9+
* Git
* (Optional) Docker

Steps
1. Clone the repo
   git clone https://github.com/AdieshwarReddy/Adhi-s-NST-.git
   cd "Adhi's NST"

2. (Optional) Create a virtual environment
   python -m venv .venv
   .venv\Scripts\activate   (PowerShell)  or  source .venv/bin/activate (Linux/macOS)

3. Install dependencies
   pip install -r requirements.txt

4. Run the Flask server
   python NST_Code/app.py

5. Open a browser and go to http://127.0.0.1:5000/
   Upload a content image and a style image, set “alpha”, and click “Run”.

Docker Run
----------
# Build the image (run from the repository root)
docker build -t adhi-nst .

# Run the container, exposing port 5000
docker run -p 5000:5000 adhi-nst

Visit http://localhost:5000/ to use the app.

Project Structure
-----------------
Adhi's NST/
├─ .gitignore
├─ Dockerfile
├─ requirements.txt
├─ README.md (this file) / README.txt
├─ NST_Code/
│  ├─ app.py                # Flask entry point
│  ├─ train.py              # Model training script (optional)
│  ├─ utils/
│  │   ├─ models.py          # VGG encoder + decoder definitions
│  │   └─ utils.py           # AdaIN implementation
│  └─ templates/
│      └─ index.html         # UI page
├─ Demo_IO_Images/
│  ├─ i-p/                  # Content (input) samples
│  └─ o-p/                  # Style (output) samples
├─ docs/
│  ├─ PROJECT_AUDIT.md       # High‑level audit and issues
│  ├─ IMPLEMENTATION_PLAN.md
│  ├─ ENVIRONMENT.md        # Environment details
│  └─ P0_VERIFICATION.md   # Baseline verification report
└─ scripts/
   ├─ test_flask_app.py      # Simple health‑check script
   ├─ test_inference.py     # CLI inference test
   └─ verify_model.py       # Model loading verification script

Deployment Options (Free Tier)
------------------------------
Render
  * Build: pip install -r requirements.txt
  * Start: python NST_Code/app.py
  * Port: 5000

Railway
  * Connect repo, Railway auto‑detects Dockerfile

Azure App Service
  * Deployment Center → GitHub repo
  * Startup command: python NST_Code/app.py
  * App Setting: PORT=5000

Fly.io
  * fly launch (choose Dockerfile)
  * fly deploy

No database is required for the current stateless inference pipeline.
If you later need persistence, you can add Supabase or any DB, but the core
application works without one.

Customization & Renaming
-----------------------
* Repository URL – change on GitHub → Settings → Repository name,
  then update local remote:
    git remote set-url origin https://github.com/<user>/<new-repo>.git

* Project folder – rename in Explorer, then
    git add -A && git commit -m "Rename project folder" && git push

* Web page title – edit the <title> tag in NST_Code/templates/index.html

* Docker image tag – after renaming:
    docker build -t <new-name>:latest .
    docker push <new-name>:latest

Learning the Codebase
---------------------
- docs/PROJECT_AUDIT.md – architecture, design decisions, known issues
- NST_Code/utils/models.py – VGG encoder, decoder, weight loading
- NST_Code/utils/utils.py – adaptive_instance_normalization, calc_mean_std
- NST_Code/app.py – Flask routes, image handling, inference workflow
- scripts/test_inference.py – CLI usage for inference without UI
- scripts/verify_model.py – verifies that model weights load correctly
- Dockerfile – container build steps

Contributing
------------
1. Fork the repository.
2. Create a feature branch: git checkout -b feature/your‑feature
3. Make changes and ensure the server still starts.
4. Run test scripts (scripts/test_inference.py) for sanity checks.
5. Submit a Pull Request with a clear description.
   * Keep the repo stateless – no DB credentials or large binaries.
   * Add any new dependencies to requirements.txt.

License
-------
MIT License – see LICENSE file.

Contact
-------
Author: Adieshwar Reddy  
Email: mogiliadieshwarreddy5919@gmail.com  
YouTube: https://www.youtube.com/@AdieshwarReddyMogili  
GitHub Issues: open an issue for bugs, feature requests, or help.

Happy styling!
