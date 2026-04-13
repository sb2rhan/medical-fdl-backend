"""
extractor.py  (re-export shim)
==============================
The full MedicalNet ResNet3D extractor lives in app/services/extractor.py.
This shim keeps backwards-compatibility with any code that previously
imported from the top-level `extractor` module.
"""
from app.services.extractor import *  # noqa: F401,F403
