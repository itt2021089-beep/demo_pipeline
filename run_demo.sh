#!/usr/bin/env bash
cd "$(dirname "$0")"
if [ -x venv/bin/python ]; then
    venv/bin/python -m streamlit run demo/app.py
else
    python -m streamlit run demo/app.py
fi
