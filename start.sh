#!/bin/bash
# Start the FastAPI backend on port 8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start
sleep 5

# Start the Streamlit frontend on port 7860 (Hugging Face default)
python -m streamlit run frontend/app.py --server.port 7860 --server.address 0.0.0.0
