"""
Vercel Serverless Function Entrypoint for Flask Application.
"""

import os
import sys

# Ensure parent directory is in sys.path so app and sibling modules import correctly
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app import app

# Vercel looks for the WSGI app variable (app or handler)
handler = app

if __name__ == "__main__":
    app.run(debug=True)
