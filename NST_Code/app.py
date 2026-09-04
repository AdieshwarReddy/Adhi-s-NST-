import io
import os
import uuid
from pathlib import Path
import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from PIL import Image
from torchvision import transforms

# Set up base directories using pathlib
BASE_DIR = Path(__file__).resolve().parent

# Import existing AdaIN models and utilities
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std

app = Flask(__name__)

# Security and Application Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'adhis-nst-default-secret-change-in-prod')
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'static' / 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
max_upload_mb = int(os.environ.get('MAX_UPLOAD_MB', 10))
app.config['MAX_CONTENT_LENGTH'] = max_upload_mb * 1024 * 1024
image_size = int(os.environ.get('NST_IMAGE_SIZE', 512))

Bootstrap(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0)
    submit = SubmitField('Transfer Style')


# Determine device
requested_device = os.environ.get('MODEL_DEVICE', 'auto').lower()
if requested_device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
elif requested_device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Resolve model weights paths
vgg_weights_path = Path(os.environ.get('VGG_WEIGHTS_PATH') or (BASE_DIR / 'vgg_normalised.pth')).resolve()
decoder_weights_path = Path(os.environ.get('DECODER_WEIGHTS_PATH') or (BASE_DIR / 'experiment' / 'final_exp' / 'decoder_final.pth')).resolve()

if not vgg_weights_path.exists():
    raise FileNotFoundError(f"VGG weights not found at: {vgg_weights_path}")
if not decoder_weights_path.exists():
    raise FileNotFoundError(f"Decoder weights not found at: {decoder_weights_path}")

# Load models ONCE on startup
print(f"Loading models on device: {device}")
encoder = VGGEncoder(str(vgg_weights_path)).to(device)
encoder.eval()

decoder = Decoder().to(device)
decoder.load_state_dict(torch.load(str(decoder_weights_path), map_location=device))
decoder.eval()
print("Neural Style Transfer models loaded successfully.")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def validate_image_file(file_storage):
    """Validate that the uploaded file is a valid and readable image."""
    if not file_storage or not file_storage.filename:
        return False, "No file provided."
    filename = secure_filename(file_storage.filename)
    if not allowed_file(filename):
        return False, f"File format not allowed. Allowed: {', '.join(app.config['ALLOWED_EXTENSIONS'])}"
    try:
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.verify()
        file_storage.stream.seek(0)
        return True, None
    except Exception:
        file_storage.stream.seek(0)
        return False, "Uploaded file is corrupted or not a valid image."


def style_transfer(content_image, style_image, encoder, decoder, alpha, device, target_size=512):
    """Run AdaIN neural style transfer on PIL images."""
    content_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1.0 - alpha) * content_feats

        stylized_tensor = decoder(stylized_feats)

    return stylized_tensor


def save_image(image_tensor, save_path):
    """Save PyTorch tensor output as an image."""
    image = image_tensor.cpu().clone().squeeze(0).clamp(0, 1)
    pil_image = transforms.ToPILImage()(image)
    pil_image.save(save_path)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST':
        # Retrieve persistent filenames if already uploaded in previous step
        content_filename = form.content_path.data or None
        style_filename = form.style_path.data or None

        # Check newly uploaded content image
        if form.content.data and form.content.data.filename:
            is_valid, msg = validate_image_file(form.content.data)
            if not is_valid:
                error = f"Content image error: {msg}"
            else:
                ext = secure_filename(form.content.data.filename).rsplit('.', 1)[1].lower()
                unique_content_name = f"content_{uuid.uuid4().hex[:8]}.{ext}"
                content_save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_content_name)
                form.content.data.save(content_save_path)
                content_filename = unique_content_name
                form.content_path.data = content_filename

        # Check newly uploaded style image
        if not error and form.style.data and form.style.data.filename:
            is_valid, msg = validate_image_file(form.style.data)
            if not is_valid:
                error = f"Style image error: {msg}"
            else:
                ext = secure_filename(form.style.data.filename).rsplit('.', 1)[1].lower()
                unique_style_name = f"style_{uuid.uuid4().hex[:8]}.{ext}"
                style_save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_style_name)
                form.style.data.save(style_save_path)
                style_filename = unique_style_name
                form.style_path.data = style_filename

        # Validate inputs present
        if not error:
            if not content_filename:
                error = 'Please select or upload a Content Image.'
            elif not style_filename:
                error = 'Please select or upload a Style Image.'

        # Validate alpha
        if not error:
            try:
                alpha = float(form.alpha.data if form.alpha.data is not None else 1.0)
                if not (0.0 <= alpha <= 1.0):
                    error = 'Style Strength (Alpha) must be between 0.0 and 1.0.'
            except (ValueError, TypeError):
                error = 'Style Strength (Alpha) must be a valid number between 0.0 and 1.0.'

        # Perform Inference if validation passed
        if not error and content_filename and style_filename:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            if not os.path.exists(content_path):
                error = "Content image file not found on server. Please upload again."
            elif not os.path.exists(style_path):
                error = "Style image file not found on server. Please upload again."
            else:
                try:
                    content_image = Image.open(content_path).convert('RGB')
                    style_image = Image.open(style_path).convert('RGB')

                    stylized_tensor = style_transfer(
                        content_image, style_image, encoder, decoder, alpha, device, target_size=image_size
                    )

                    result_filename = f"stylized_{uuid.uuid4().hex[:8]}.png"
                    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                    save_image(stylized_tensor, result_path)

                    result_image = result_filename
                except Exception as e:
                    # In production do not expose full traceback
                    app_env = os.environ.get('APP_ENV', 'development')
                    if app_env == 'production':
                        error = "An internal error occurred during style transfer. Please try again."
                    else:
                        error = f"Inference error: {str(e)}"

    return render_template(
        'index.html',
        form=form,
        result_image=result_image,
        content_image=content_filename,
        style_image=style_filename,
        error=error
    )


@app.route('/uploads/<filename>')
def send_image(filename):
    safe_filename = secure_filename(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename)


@app.route('/download/<filename>')
def download_image(filename):
    safe_filename = secure_filename(filename)
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        safe_filename,
        as_attachment=True,
        download_name=f"adhis_nst_{safe_filename}"
    )


DEMO_DIR = BASE_DIR.parent / 'Demo_IO_Images'


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


@app.route('/demo_images/<path:filename>')
def send_demo_image(filename):
    return send_from_directory(DEMO_DIR, filename)


if __name__ == '__main__':
    is_dev = os.environ.get('APP_ENV', 'development').lower() == 'development'
    app.run(host='127.0.0.1', port=5000, debug=is_dev)
