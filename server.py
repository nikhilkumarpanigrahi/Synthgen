import os
import sys
import argparse
import json
import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(description="Run Synthgen or prepare its real datasets.")
    subparsers = parser.add_subparsers(dest="command")

    local_parser = subparsers.add_parser("import-local", help="Import a local class-folder dataset.")
    local_parser.add_argument("--domain", choices=["road_defects", "industrial_defects"], required=True)
    local_parser.add_argument("--source", required=True, help="Root directory containing the downloaded dataset.")

    isic_parser = subparsers.add_parser("download-isic", help="Download a small ISIC labelled subset.")
    isic_parser.add_argument("--per-class", type=int, default=20, help="Number of images per skin class.")

    return parser.parse_args()

if __name__ == "__main__":
    # Ensure current directory is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    args = parse_args()
    if args.command == "import-local":
        from backend.dataset_manager import import_local_dataset
        result = import_local_dataset(args.domain, args.source)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    if args.command == "download-isic":
        from backend.dataset_manager import download_isic_dataset
        result = download_isic_dataset(per_class=args.per_class)
        print(json.dumps(result, indent=2))
        sys.exit(0)
        
    port = int(os.environ.get("PORT", 8080))
    print("=" * 60)
    print("🚀 SYNTHETIQ — AI-Based Synthetic Data Generator")
    print("=" * 60)
    print("Starting FastAPI Backend & Interactive Web UI...")
    print(f"Access the dashboard at: http://localhost:{port}")
    print(f"API Documentation at:    http://localhost:{port}/docs")
    print("=" * 60)
    
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
