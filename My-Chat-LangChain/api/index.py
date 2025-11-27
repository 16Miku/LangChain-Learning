import os
import sys

# Add the backend directory to sys.path to allow imports within main.py to work
# This makes 'backend' the root for imports, matching local development environment
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from main import app
