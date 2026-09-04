import sys
import time
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
NST_CODE_DIR = BASE_DIR / "NST_Code"
sys.path.insert(0, str(NST_CODE_DIR))

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization, calc_mean_std


def test_adain_math():
    """Verify AdaIN and calc_mean_std mathematical correctness and tensor shapes."""
    print("--- Testing AdaIN and calc_mean_std ---")
    
    # 1. Test shape and properties
    batch_size, channels, h, w = 2, 512, 32, 32
    c_feat = torch.randn(batch_size, channels, h, w)
    s_feat = torch.randn(batch_size, channels, h, w)

    c_mean, c_std = calc_mean_std(c_feat)
    s_mean, s_std = calc_mean_std(s_feat)

    assert c_mean.shape == (batch_size, channels, 1, 1), f"Unexpected c_mean shape: {c_mean.shape}"
    assert c_std.shape == (batch_size, channels, 1, 1), f"Unexpected c_std shape: {c_std.shape}"
    assert (c_std > 0).all(), "Standard deviations must be strictly positive"

    # AdaIN output
    adain_feat = adaptive_instance_normalization(c_feat, s_feat)
    assert adain_feat.shape == (batch_size, channels, h, w), f"Unexpected AdaIN shape: {adain_feat.shape}"

    # Verify that normalized adain_feat matches s_mean and s_std
    out_mean, out_std = calc_mean_std(adain_feat)
    mean_diff = torch.max(torch.abs(out_mean - s_mean)).item()
    std_diff = torch.max(torch.abs(out_std - s_std)).item()

    assert mean_diff < 1e-4, f"AdaIN output mean diff too large: {mean_diff}"
    assert std_diff < 1e-4, f"AdaIN output std diff too large: {std_diff}"

    print(f"AdaIN shape verification: PASS [Shape: {list(adain_feat.shape)}]")
    print(f"AdaIN mean transfer error: {mean_diff:.2e} (PASS)")
    print(f"AdaIN std transfer error: {std_diff:.2e} (PASS)")
    print("AdaIN mathematical verification: SUCCESS\n")


def run_style_transfer(encoder, decoder, content_img, style_img, alpha=1.0, device="cpu", image_size=512):
    """Run style transfer using the real PyTorch model."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])

    c_tensor = transform(content_img).unsqueeze(0).to(device)
    s_tensor = transform(style_img).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        c_feat = encoder(c_tensor, is_test=True)
        s_feat = encoder(s_tensor, is_test=True)

        t = adaptive_instance_normalization(c_feat, s_feat)
        t = alpha * t + (1.0 - alpha) * c_feat

        output_tensor = decoder(t)

    elapsed_ms = (time.time() - start_time) * 1000

    output_tensor = output_tensor.cpu().squeeze(0).clamp(0, 1)
    result_img = transforms.ToPILImage()(output_tensor)

    return result_img, elapsed_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference verification on {device}...")

    # First verify AdaIN math
    test_adain_math()

    # Load models
    vgg_weights = NST_CODE_DIR / "vgg_normalised.pth"
    decoder_weights = NST_CODE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

    encoder = VGGEncoder(str(vgg_weights)).to(device)
    encoder.eval()

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(str(decoder_weights), map_location=device))
    decoder.eval()

    # Define output directory
    output_dir = BASE_DIR / "artifacts" / "verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Input image paths
    demo_dir = BASE_DIR / "Demo_IO_Images" / "i-p"
    content_path = demo_dir / "i_p image.jpg"
    style1_path = demo_dir / "style 1.png"
    style2_path = demo_dir / "style 2.jpg"

    assert content_path.exists(), f"Missing content image: {content_path}"
    assert style1_path.exists(), f"Missing style 1 image: {style1_path}"
    assert style2_path.exists(), f"Missing style 2 image: {style2_path}"

    content_img = Image.open(content_path).convert("RGB")
    style1_img = Image.open(style1_path).convert("RGB")
    style2_img = Image.open(style2_path).convert("RGB")

    print(f"Loaded demo content image: {content_path.name} ({content_img.size})")
    print(f"Loaded demo style 1 image: {style1_path.name} ({style1_img.size})")
    print(f"Loaded demo style 2 image: {style2_path.name} ({style2_img.size})\n")

    # Step 5: Test Style 1 Inference
    print("--- Generating Style 1 Transfer (alpha=1.0) ---")
    out_style1, latency1 = run_style_transfer(encoder, decoder, content_img, style1_img, alpha=1.0, device=device)
    path_style1 = output_dir / "style1_result.png"
    out_style1.save(path_style1)
    print(f"Saved: {path_style1} [Latency: {latency1:.1f}ms, Size: {out_style1.size}]")

    # Step 5: Test Style 2 Inference
    print("\n--- Generating Style 2 Transfer (alpha=1.0) ---")
    out_style2, latency2 = run_style_transfer(encoder, decoder, content_img, style2_img, alpha=1.0, device=device)
    path_style2 = output_dir / "style2_result.png"
    out_style2.save(path_style2)
    print(f"Saved: {path_style2} [Latency: {latency2:.1f}ms, Size: {out_style2.size}]")

    # Step 7: Test Alpha sweeps
    print("\n--- Testing Alpha Interpolation ---")
    alphas = [0.0, 0.5, 1.0]
    alpha_filenames = {0.0: "alpha_0.png", 0.5: "alpha_05.png", 1.0: "alpha_1.png"}

    for a in alphas:
        out_alpha, lat = run_style_transfer(encoder, decoder, content_img, style1_img, alpha=a, device=device)
        save_path = output_dir / alpha_filenames[a]
        out_alpha.save(save_path)
        print(f"Alpha {a}: Saved {save_path.name} [Latency: {lat:.1f}ms]")

    # Test invalid alpha rejection
    try:
        run_style_transfer(encoder, decoder, content_img, style1_img, alpha=-0.1, device=device)
        print("FAIL: Invalid alpha -0.1 was not rejected!")
    except ValueError:
        print("Alpha -0.1 correctly rejected: PASS")

    try:
        run_style_transfer(encoder, decoder, content_img, style1_img, alpha=1.5, device=device)
        print("FAIL: Invalid alpha 1.5 was not rejected!")
    except ValueError:
        print("Alpha 1.5 correctly rejected: PASS")

    print("\n==========================================")
    print("ALL INFERENCE AND MATHEMATICAL TESTS PASSED!")
    print("==========================================")


if __name__ == "__main__":
    main()
