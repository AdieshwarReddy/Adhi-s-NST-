import sys
from pathlib import Path
import torch

# Add NST_Code to python path
BASE_DIR = Path(__file__).resolve().parent.parent
NST_CODE_DIR = BASE_DIR / "NST_Code"
sys.path.insert(0, str(NST_CODE_DIR))

from utils.models import VGGEncoder, Decoder


def verify_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {str(device).upper()}")

    vgg_weights_path = NST_CODE_DIR / "vgg_normalised.pth"
    decoder_weights_path = NST_CODE_DIR / "experiment" / "final_exp" / "decoder_final.pth"

    if not vgg_weights_path.exists():
        print(f"Error: VGG weights file not found at {vgg_weights_path}")
        return False
    if not decoder_weights_path.exists():
        print(f"Error: Decoder weights file not found at {decoder_weights_path}")
        return False

    try:
        # Load VGG Encoder
        encoder = VGGEncoder(str(vgg_weights_path)).to(device)
        encoder.eval()
        print("VGG: LOADED")
    except Exception as e:
        print(f"VGG: FAILED ({e})")
        return False

    try:
        # Load Decoder
        decoder = Decoder().to(device)
        decoder_state = torch.load(str(decoder_weights_path), map_location=device)
        decoder.load_state_dict(decoder_state)
        decoder.eval()
        print("Decoder: LOADED")
    except Exception as e:
        print(f"Decoder: FAILED ({e})")
        return False

    print("Model verification: SUCCESS")
    return True


if __name__ == "__main__":
    success = verify_models()
    sys.exit(0 if success else 1)
