"""Application package and paths shared by the launcher and runtime."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
