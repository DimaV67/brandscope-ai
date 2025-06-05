"""
Entry point for running the application as a module.
"""
import sys
from pathlib import Path

# Add the src directory to the path to handle imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

try:
    from cli import cli
    
    if __name__ == '__main__':
        cli()
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've installed the package: pip install -e .")
    sys.exit(1)