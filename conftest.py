"""
Pytest configuration file.
This file helps pytest find the src module by adding it to the Python path.
"""

import sys
from pathlib import Path

# Add the project root to Python path so 'src' can be imported
project_root = Path(__file__).parent
src_path = project_root / "src"

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))