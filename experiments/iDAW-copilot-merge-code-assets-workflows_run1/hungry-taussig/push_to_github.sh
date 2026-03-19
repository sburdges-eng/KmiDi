#!/bin/bash
# Push to GitHub - The Lariat Bible

echo "🤠 The Lariat Bible - GitHub Setup"
echo "=================================="
echo ""
echo "Follow these steps to push your project to GitHub:"
echo ""
echo "1. Go to https://github.com/new"
echo "2. Create a new repository called 'lariat-bible'"
echo "3. Make it PRIVATE (contains business information)"
echo "4. DON'T initialize with README, .gitignore, or license"
echo "5. After creating, run these commands:"
echo ""
echo "git remote add origin https://github.com/YOUR-USERNAME/lariat-bible.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "Replace YOUR-USERNAME with your GitHub username"
echo ""
echo "=================================="
echo ""
echo "Your repository is ready to push!"
echo "Current status:"
git status
echo ""
echo "Current branch:"
git branch
echo ""
echo "Commit history:"
git log --oneline -5
