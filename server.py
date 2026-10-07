import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    port = int(os.environ.get("PORT", 8080))
    print("=" * 60)
    print("🚀 SYNTHETIQ — AI-Based Synthetic Data Generator")
    print("=" * 60)
    print("Starting FastAPI Backend & Interactive Web UI...")
    print(f"Access the dashboard at: http://localhost:{port}")
    print(f"API Documentation at:    http://localhost:{port}/docs")
    print("=" * 60)
    
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=reload)
