import os
import sys

# Patch sqlite3 with pysqlite3-binary to satisfy chroma requirements on Vercel
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Add the backend directory to sys.path to allow imports within main.py to work
# This makes 'backend' the root for imports, matching local development environment
sys.path.append(os.path.join(os.path.dirname(__file__), '../backend'))

from main import app
