#!/bin/bash
cd /Users/seanburdges/Documents/GitHub/iDAW

for i in $(seq 1 195); do
    FILE="reposean_archives/reposean${i}.zip"
    if [ -f "$FILE" ]; then
        echo "=== Pushing reposean${i}.zip ==="
        git add "$FILE"
        git commit -m "Add reposean${i}.zip"
        git push origin main
        if [ $? -ne 0 ]; then
            echo "Push failed for reposean${i}.zip, retrying..."
            git pull --rebase origin main
            git push origin main
        fi
        echo "Done with reposean${i}.zip"
        echo ""
    fi
done

echo "ALL DONE!"
