import os
import sys
from pathlib import Path
import io
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

# Set base directories and sys path
BASE_DIR = Path(__file__).resolve().parent
NST_DIR = BASE_DIR / 'NST_Code'
if str(NST_DIR) not in sys.path:
    sys.path.insert(0, str(NST_DIR))

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

st.set_page_config(
    page_title="Adhi's NST — Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide"
)

# Custom Glassmorphism Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.3rem;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #64748b;
        font-size: 0.9rem;
    }
    .footer a {
        color: #818cf8;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎨 Adhi\'s NST — Style Transfer Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time Arbitrary Style Transfer using Adaptive Instance Normalization (Huang & Belongie, 2017)</div>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_path = NST_DIR / 'vgg_normalised.pth'
    decoder_path = NST_DIR / 'experiment' / 'final_exp' / 'decoder_final.pth'
    
    if not vgg_path.exists():
        raise FileNotFoundError(f"VGG weights not found at: {vgg_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder weights not found at: {decoder_path}")
    
    encoder = VGGEncoder(str(vgg_path)).to(device)
    encoder.eval()
    
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(str(decoder_path), map_location=device))
    decoder.eval()
    
    return encoder, decoder, device

try:
    encoder, decoder, device = load_models()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    model_loaded = False

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    alpha = st.slider(
        "✨ Style Strength (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05,
        help="1.0 = 100% style transfer, 0.0 = 100% original content."
    )
    if model_loaded:
        st.success(f"Running on: **{device.type.upper()}**")
    
    st.markdown("---")
    st.subheader("💡 Demo Presets")
    use_preset = st.checkbox("Use Demo Preset Images", value=False)
    preset_style_choice = "Sketch Style"
    if use_preset:
        preset_style_choice = st.radio("Choose Preset Style:", ["Sketch Style (Style 1)", "Cubist Painting (Style 2)"])

# Image Selection / Upload
col1, col2 = st.columns(2)
content_img = None
style_img = None

demo_ip_path = BASE_DIR / 'Demo_IO_Images' / 'i-p' / 'i_p image.jpg'
demo_s1_path = BASE_DIR / 'Demo_IO_Images' / 'i-p' / 'style 1.png'
demo_s2_path = BASE_DIR / 'Demo_IO_Images' / 'i-p' / 'style 2.jpg'

with col1:
    st.subheader("🖼️ Content Image")
    if use_preset and demo_ip_path.exists():
        content_img = Image.open(demo_ip_path).convert('RGB')
        st.image(content_img, caption="Demo Portrait (Content)", use_container_width=True)
    else:
        content_file = st.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png', 'webp'], key='content')
        if content_file:
            content_img = Image.open(content_file).convert('RGB')
            st.image(content_img, caption="Uploaded Content", use_container_width=True)

with col2:
    st.subheader("🎨 Style Image")
    if use_preset:
        s_path = demo_s1_path if "Style 1" in preset_style_choice else demo_s2_path
        if s_path.exists():
            style_img = Image.open(s_path).convert('RGB')
            st.image(style_img, caption=f"Demo {preset_style_choice}", use_container_width=True)
    else:
        style_file = st.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png', 'webp'], key='style')
        if style_file:
            style_img = Image.open(style_file).convert('RGB')
            st.image(style_img, caption="Uploaded Style", use_container_width=True)

st.markdown("---")

if st.button("🚀 Generate Artwork", type="primary", use_container_width=True):
    if content_img is None or style_img is None:
        st.warning("⚠️ Please provide both a Content Image and a Style Image (or check 'Use Demo Preset Images').")
    elif not model_loaded:
        st.error("❌ Models are not loaded.")
    else:
        with st.spinner("Transforming artwork using AdaIN neural network..."):
            try:
                transform = transforms.Compose([
                    transforms.Resize((512, 512)),
                    transforms.ToTensor()
                ])
                c_tensor = transform(content_img).unsqueeze(0).to(device)
                s_tensor = transform(style_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    c_feat = encoder(c_tensor, is_test=True)
                    s_feat = encoder(s_tensor, is_test=True)
                    target_feat = adaptive_instance_normalization(c_feat, s_feat)
                    target_feat = alpha * target_feat + (1.0 - alpha) * c_feat
                    out_tensor = decoder(target_feat)

                out_tensor = out_tensor.squeeze(0).cpu().clamp(0, 1)
                res_img = transforms.ToPILImage()(out_tensor)

                st.success("🎉 Artwork Generated Successfully!")
                st.image(res_img, caption=f"Stylized Artwork (Alpha = {alpha})", use_container_width=True)

                buf = io.BytesIO()
                res_img.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Artwork",
                    data=buf.getvalue(),
                    file_name=f"adhis_nst_alpha_{alpha}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as ex:
                st.error(f"Inference error: {ex}")

# Examples Gallery Section
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;">
    <p style="color: #a855f7; font-size: 0.85rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.2rem;">GALLERY</p>
    <h2 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.3rem;">Example Transfers</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">Each example shows Content + Style → Stylized Output</p>
</div>
""", unsafe_allow_html=True)

gal_col1, gal_col2 = st.columns(2)

demo_o1_path = BASE_DIR / 'Demo_IO_Images' / 'o-p' / 'o_p style 1.jpg'
demo_o2_path = BASE_DIR / 'Demo_IO_Images' / 'o-p' / 'o_p style 2.jpg'

with gal_col1:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <h4 style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 1rem;">✏️ Transfer 1: Pencil Sketch Style</h4>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.4])
    with c1:
        if demo_ip_path.exists():
            st.caption("CONTENT")
            st.image(str(demo_ip_path), use_container_width=True)
    with c2:
        if demo_s1_path.exists():
            st.caption("STYLE")
            st.image(str(demo_s1_path), use_container_width=True)
    with c3:
        if demo_o1_path.exists():
            st.caption("PENCIL SKETCH OUTPUT")
            st.image(str(demo_o1_path), use_container_width=True)

with gal_col2:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <h4 style="color: #e2e8f0; font-size: 1.1rem; margin-bottom: 1rem;">🎨 Transfer 2: Cubist Art Style</h4>
    </div>
    """, unsafe_allow_html=True)
    c4, c5, c6 = st.columns([1, 1, 1.4])
    with c4:
        if demo_ip_path.exists():
            st.caption("CONTENT")
            st.image(str(demo_ip_path), use_container_width=True)
    with c5:
        if demo_s2_path.exists():
            st.caption("STYLE")
            st.image(str(demo_s2_path), use_container_width=True)
    with c6:
        if demo_o2_path.exists():
            st.caption("CUBIST ART OUTPUT")
            st.image(str(demo_o2_path), use_container_width=True)

# How It Works Pipeline Section
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 1.5rem; margin-bottom: 2rem;">
    <p style="color: #a855f7; font-size: 0.85rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.2rem;">PIPELINE</p>
    <h2 style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.3rem;">How It Works</h2>
    <p style="color: #94a3b8; font-size: 0.95rem;">Real-time feedforward architecture with no optimization loops</p>
</div>
""", unsafe_allow_html=True)

p1, p2, p3, p4, p5 = st.columns(5)
with p1:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; text-align: center;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🖼️</div>
        <h5 style="color: #f1f5f9; font-size: 1rem; margin-bottom: 0.2rem;">1. Input Images</h5>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Content and Style photos uploaded & resized</p>
    </div>
    """, unsafe_allow_html=True)
with p2:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; text-align: center;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🧠</div>
        <h5 style="color: #f1f5f9; font-size: 1rem; margin-bottom: 0.2rem;">2. VGG Encoder</h5>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Extracts deep feature maps at relu4_1</p>
    </div>
    """, unsafe_allow_html=True)
with p3:
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 12px; padding: 1rem; text-align: center;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">✨</div>
        <h5 style="color: #818cf8; font-size: 1rem; margin-bottom: 0.2rem;">3. AdaIN Alignment</h5>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Aligns feature mean & std statistics</p>
    </div>
    """, unsafe_allow_html=True)
with p4:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; text-align: center;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🔄</div>
        <h5 style="color: #f1f5f9; font-size: 1rem; margin-bottom: 0.2rem;">4. Decoder</h5>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Inverts stylized features back into pixels</p>
    </div>
    """, unsafe_allow_html=True)
with p5:
    st.markdown("""
    <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1rem; text-align: center;">
        <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">🎨</div>
        <h5 style="color: #f1f5f9; font-size: 1rem; margin-bottom: 0.2rem;">5. Artwork</h5>
        <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Instant high-resolution stylized output</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Created by <strong>Adieshwar Reddy Mogili</strong></p>
    <a href="https://github.com/AdieshwarReddy" target="_blank">🐙 GitHub</a>
    <a href="https://www.linkedin.com/in/adieshwar-reddy-mogili-3b4b11332/" target="_blank">💼 LinkedIn</a>
    <a href="https://www.youtube.com/@AdieshwarReddyMogili" target="_blank">▶️ YouTube</a>
    <a href="mailto:mogiliadieshwarreddy5919@gmail.com">✉️ Contact</a>
</div>
""", unsafe_allow_html=True)
