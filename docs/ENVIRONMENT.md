# Environment Specification: Adhi's NST

This document records the exact, verified Python runtime and package versions for reproducing the local environment.

## Python Runtime
- **Python Version**: 3.12.1 (tags/v3.12.1:2305ca5, Dec 7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)]
- **Virtual Environment**: `.venv` created with `py -3.12 -m venv .venv`

## Key Packages & Versions
| Package | Installed Version | Requirements Specification | Status |
| :--- | :--- | :--- | :--- |
| **Python** | 3.12.1 | 3.10 - 3.12 | Verified |
| **PyTorch** (`torch`) | 2.2.2+cpu | `==2.2.2` | Verified |
| **Torchvision** (`torchvision`) | 0.17.2 | `==0.17.2` | Verified |
| **Pillow** (`PIL`) | 12.0.0 | `==12.0.0` | Verified |
| **Flask** | 3.1.2 | `==3.1.2` | Verified |
| **Flask-Bootstrap** | 3.3.7.1 | `==3.3.7.1` | Verified |
| **Flask-WTF** | 1.2.2 | `==1.2.2` | Verified |
| **WTForms** | 3.2.1 | `==3.2.1` | Verified |
| **Werkzeug** | 3.1.4 | `==3.1.4` | Verified |
| **NumPy** | 1.26.4 | `>=1.24,<2.0` | Verified |
| **tqdm** | 4.66.4 | `==4.66.4` | Verified |
| **Gunicorn** | 26.2.0 | `gunicorn` | Verified |

## Verification Command
```powershell
.\.venv\Scripts\python.exe -m pip check
```
**Output**: `No broken requirements found.`
