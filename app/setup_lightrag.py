"""
Setup script to add local LightRAG to Python path
"""
import sys
import os
from pathlib import Path

def setup_lightrag_path():
    """Add local LightRAG directory to Python path"""
    
    # Get the project root directory (parent of app/)
    project_root = Path(__file__).parent.parent
    lightrag_path = project_root / "lightrag"
    
    # Add the PROJECT ROOT to Python path so lightrag can be imported as a package
    if lightrag_path.exists():
        project_root_str = str(project_root.absolute())
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
            print(f"✅ Added LightRAG to Python path: {project_root_str}")
        return True
    else:
        raise FileNotFoundError(f"❌ lightrag directory not found at: {lightrag_path}")

def verify_lightrag_import():
    """Verify that LightRAG can be imported"""
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import TokenTracker
        print("✅ LightRAG successfully imported from local source")
        return True
    except ImportError as e:
        print(f"❌ Failed to import LightRAG: {e}")
        return False

if __name__ == "__main__":
    setup_lightrag_path()
    verify_lightrag_import() 