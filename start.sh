#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f ".venv/bin/python" ]; then
    echo "Using project virtual environment..."
    .venv/bin/python server.py
else
    echo "Virtual environment not found, trying python3..."
    python3 server.py
fi
