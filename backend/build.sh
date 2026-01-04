#!/bin/bash
# Render.com build script for Word Weaver Quest backend

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!"
