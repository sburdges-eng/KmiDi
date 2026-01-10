#!/bin/bash
# Setup script for Streamlit Cloud deployment

echo "Setting up KmiDi Streamlit Demo..."

# Install dependencies
pip install -r requirements.txt

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Copy example secrets if secrets.toml doesn't exist
if [ ! -f .streamlit/secrets.toml ]; then
    if [ -f .streamlit/secrets.toml.example ]; then
        cp .streamlit/secrets.toml.example .streamlit/secrets.toml
        echo "Created .streamlit/secrets.toml from example"
    fi
fi

# Make streamlit_app.py executable
chmod +x streamlit_app.py

echo "Setup complete!"
echo "Run with: streamlit run streamlit_app.py"
