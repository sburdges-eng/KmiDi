#!/bin/bash
echo "Starting Lariat Bible..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    open dist/LariatBible.app
else
    ./dist/LariatBible
fi
