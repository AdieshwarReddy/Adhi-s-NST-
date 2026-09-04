import io
import sys
from pathlib import Path
import torch
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
NST_CODE_DIR = BASE_DIR / "NST_Code"
sys.path.insert(0, str(NST_CODE_DIR))

from app import app


def test_flask_suite():
    print("==========================================")
    print("STARTING FLASK END-TO-END TEST SUITE")
    print("==========================================")
    client = app.test_client()

    # 1. GET /
    print("\n[Test 1] Testing GET / ...")
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    # Verify no premature error on initial GET
    assert b"Please select or upload" not in res.data, "Premature error on initial GET!"
    print("PASS: Homepage loaded cleanly without errors.")

    # 2. Missing inputs
    print("\n[Test 2] Testing POST without files (missing inputs)...")
    res = client.post('/', data={}, content_type='multipart/form-data')
    assert res.status_code == 200
    assert b"Please select or upload" in res.data
    print("PASS: Missing inputs correctly caught and error banner displayed.")

    # 3. Invalid file (fake non-image content)
    print("\n[Test 3] Testing POST with corrupted/invalid image file...")
    fake_file = (io.BytesIO(b"Not an image data at all"), "fake.png")
    valid_img_buf = io.BytesIO()
    Image.new("RGB", (100, 100), color="blue").save(valid_img_buf, format="JPEG")
    valid_img_buf.seek(0)
    valid_file = (valid_img_buf, "valid.jpg")

    res = client.post('/', data={
        'content': fake_file,
        'style': valid_file,
        'alpha': '1.0'
    }, content_type='multipart/form-data')
    assert res.status_code == 200
    assert b"corrupted or not a valid image" in res.data
    print("PASS: Corrupted/invalid image file correctly rejected.")

    # 4. Invalid alpha
    print("\n[Test 4] Testing invalid alpha values...")
    demo_dir = BASE_DIR / "Demo_IO_Images" / "i-p"
    with open(demo_dir / "i_p image.jpg", "rb") as f_c, open(demo_dir / "style 1.png", "rb") as f_s:
        res = client.post('/', data={
            'content': (f_c, "content.jpg"),
            'style': (f_s, "style.png"),
            'alpha': '2.5'
        }, content_type='multipart/form-data')
        assert res.status_code == 200
        assert b"Style Strength (Alpha) must be between 0.0 and 1.0" in res.data
    print("PASS: Out-of-bounds alpha value rejected.")

    # 5. Real End-to-End Style Transfer Generation
    print("\n[Test 5] Testing valid style transfer via Flask POST...")
    with open(demo_dir / "i_p image.jpg", "rb") as f_c, open(demo_dir / "style 1.png", "rb") as f_s:
        res = client.post('/', data={
            'content': (f_c, "content.jpg"),
            'style': (f_s, "style.png"),
            'alpha': '0.8'
        }, content_type='multipart/form-data')
        assert res.status_code == 200
        assert b"Stylized Result" in res.data, "Result section not found in HTML response!"
        assert b"Download Artwork" in res.data, "Download Artwork link not found!"
        
        # Extract stylized filename
        import re
        match = re.search(r'/uploads/(stylized_[a-f0-9]+\.png)', res.data.decode('utf-8'))
        assert match, "Could not find result image filename in response!"
        result_filename = match.group(1)
        print(f"PASS: Style transfer executed. Result filename: {result_filename}")

    # 6. Verify result image exists in uploads directory
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    result_path = upload_folder / result_filename
    assert result_path.exists(), f"Result file does not exist on disk: {result_path}"
    assert result_path.stat().st_size > 0, "Result file is empty!"
    print(f"PASS: Result file verified on disk ({result_path.stat().st_size} bytes).")

    # 7. Test download route
    print("\n[Test 6] Testing GET /download/<filename>...")
    download_res = client.get(f'/download/{result_filename}')
    assert download_res.status_code == 200, f"Download returned {download_res.status_code}"
    assert 'attachment' in download_res.headers.get('Content-Disposition', '')
    print("PASS: Download route returned 200 with attachment disposition.")

    print("\n==========================================")
    print("ALL FLASK TEST SUITE TESTS PASSED!")
    print("==========================================")
    return True


if __name__ == "__main__":
    success = test_flask_suite()
    sys.exit(0 if success else 1)
