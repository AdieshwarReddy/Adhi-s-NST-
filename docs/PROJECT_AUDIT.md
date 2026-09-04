# Project Audit: Adhi's NST

## CURRENT ARCHITECTURE
The project currently uses a Flask monolith architecture. 
- **Frontend**: HTML/CSS/JS with Jinja2 templates (Bootstrap, custom CSS).
- **Backend**: Flask handling file uploads and routing.
- **AI**: PyTorch running in-process. VGG and Decoder models are loaded globally when the backend starts.
- **Storage**: Local filesystem (`static/uploads`).

## REPOSITORY TREE
```
C:.
|   .gitignore
|   code.ipynb
|   Procfile.txt
|   README.md
|   requirements.txt
|   
+---Demo_IO_Images
|   +---i-p
|   |       i_p image.jpg
|   |       style 1.png
|   |       style 2.jpg
|   |       
|   \---o-p
|           o_p style 1.jpg
|           o_p style 2.jpg
|           
\---NST_Code
    |   adain_algo.png
    |   app.py
    |   train.py
    |   vgg_normalised.pth
    |   
    +---.vscode
    |       settings.json
    |       
    +---content_data
    |       ... (12 items)
    |       
    +---examples
    |       ... (5 items)
    |       
    +---experiment
    |   \---final_exp
    |           decoder_final.pth
    |           ... (training samples)
    |           
    +---static
    |   \---uploads
    |           ... (5 items)
    |           
    +---style_data
    |       ... (19 items)
    |       
    +---templates
    |       index.html
    |       
    \---utils
            models.py
            utils.py
```

## EXISTING FEATURES
- Upload content and style images via UI.
- Alpha slider to control style strength.
- Inference running on CPU/GPU depending on availability.
- Generated image saving and display in the UI.
- Example gallery in the UI (partially hardcoded).

## AI PIPELINE
1. **Preprocessing**: Resize to 512x512, convert to RGB, normalize to Tensor.
2. **VGG Encoder**: Extracts features up to `relu4_1`.
3. **AdaIN**: Normalizes content features and scales/shifts them using style feature statistics (mean/std).
4. **Alpha Interpolation**: `stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats`.
5. **Decoder**: Reconstructs the image from stylized features.
6. **Postprocessing**: Clamp [0,1], convert back to PIL Image.

## TRAINING PIPELINE
- **Data**: Uses custom `ImageFolderDataset` with `DataLoader`.
- **Losses**: 
  - Content loss: MSE between stylized features and content features at `relu4_1`.
  - Style loss: MSE between mean and std of stylized and style features across `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`.
- **Optimizer**: Adam with learning rate decay scheduling.
- **Saves**: Model state dicts (decoder, optimizer) and sample images saved periodically.

## INFERENCE PIPELINE
- Weights loaded once globally in `app.py`.
- `style_transfer()` function executes the entire forward pass (preprocessing to postprocessing).
- Executes under `torch.no_grad()`.

## MODEL WEIGHTS
- `vgg_normalised.pth`: 80.1 MB
- `decoder_final.pth`: 14.0 MB

## WHAT WORKS
- The core math for AdaIN (`adaptive_instance_normalization`, `calc_mean_std`) appears mathematically correct (using unbiased variance + eps).
- VGG Encoder correctly extracts 4 layers.
- Decoder correctly maps features back to image space.

## WHAT FAILS (BUGS)
- The codebase uses strict hardcoded, machine-specific paths (e.g., `/home/ubuntu/Desktop/NST_Code/...`). 
- If run immediately, the app will crash trying to load models from non-existent absolute paths.

## DEPENDENCY ISSUES
- The codebase targets `Flask==3.1.2`, `torch==2.2.2`, `torchvision==0.17.2`.
- Relies on older or specific package versions, but the `requirements.txt` is clear.

## PORTABILITY ISSUES
- Highly unportable due to `/home/ubuntu/Desktop/...` absolute paths scattered in both `app.py` and `train.py`.

## SECURITY ISSUES
- Flask `SECRET_KEY` is hardcoded to `'supersecretkey'`.
- Debug mode is turned on by default in `__main__` via `use_debugger=True`.
- Path traversal vulnerabilities are slightly mitigated by `secure_filename`, but upload sizes are unconstrained.
- Image content is not validated strictly beyond checking the file extension.

## UI ISSUES
- High quality CSS, but some buttons/links have empty `#` hrefs.
- Alpha slider doesn't immediately reflect numerical value dynamically on drag (needs JS connection).
- Hardcoded examples gallery relies on specific filenames being present.

## DEPLOYMENT ISSUES
- `Procfile.txt` has a typo in its naming (should be `Procfile` without `.txt` for most PaaS).
- Missing `Dockerfile`.
- In-process PyTorch model means heavy RAM usage per worker (Gunicorn). Needs proper configuration to avoid memory exhaustion.

## TESTING STATUS
- No automated tests exist.

## DATABASE STATUS
- No database implemented; strictly ephemeral local storage currently.

## RECOMMENDED NEXT STEPS
1. Set up virtual environment and install dependencies.
2. Refactor absolute paths in `app.py` and `train.py` to use relative paths (`pathlib`).
3. Validate models by running the script locally and producing outputs.
4. Extract the AI inference logic into a separate `ml/` service folder.
