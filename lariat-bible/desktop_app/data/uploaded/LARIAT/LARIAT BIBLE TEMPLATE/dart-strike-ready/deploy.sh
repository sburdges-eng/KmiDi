#!/bin/bash
# Quick Deploy Script for Dart Strike PWA

echo "🎯 Dart Strike - Quick Deploy"
echo "=============================="
echo ""
echo "Choose deployment option:"
echo "1) Local Network (for testing on your iPhone)"
echo "2) Deploy to Netlify (free hosting)"
echo "3) Deploy to Vercel (free hosting)"
echo "4) Deploy to GitHub Pages"
echo "5) Create deployment package (zip file)"
echo ""
read -p "Enter choice (1-5): " choice

case $choice in
    1)
        echo "📱 Starting local server for iOS testing..."
        echo ""
        # Get local IP
        LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
        echo "✅ Server starting at:"
        echo "   http://localhost:8080"
        echo "   http://$LOCAL_IP:8080 (use this on your iPhone)"
        echo ""
        echo "On your iPhone:"
        echo "1. Open Safari"
        echo "2. Go to http://$LOCAL_IP:8080"
        echo "3. Tap Share > Add to Home Screen"
        echo ""
        echo "Press Ctrl+C to stop server"
        python3 -m http.server 8080
        ;;
        
    2)
        echo "☁️ Deploying to Netlify..."
        if ! command -v netlify &> /dev/null; then
            echo "Installing Netlify CLI..."
            npm install -g netlify-cli
        fi
        echo "Deploying..."
        netlify deploy --prod --dir=. --site dart-strike-$(date +%s)
        echo "✅ Deployment complete! Check the URL above."
        ;;
        
    3)
        echo "▲ Deploying to Vercel..."
        if ! command -v vercel &> /dev/null; then
            echo "Installing Vercel CLI..."
            npm install -g vercel
        fi
        vercel --prod
        echo "✅ Deployment complete!"
        ;;
        
    4)
        echo "🐙 Setting up GitHub Pages deployment..."
        echo ""
        echo "Steps to deploy to GitHub Pages:"
        echo "1. Create a new GitHub repository"
        echo "2. Run these commands:"
        echo ""
        echo "git init"
        echo "git add ."
        echo "git commit -m 'Initial commit'"
        echo "git branch -M main"
        echo "git remote add origin https://github.com/YOUR_USERNAME/dart-strike.git"
        echo "git push -u origin main"
        echo ""
        echo "3. Go to Settings > Pages in your GitHub repo"
        echo "4. Select 'Deploy from a branch'"
        echo "5. Choose 'main' branch and '/ (root)'"
        echo "6. Save and wait 2-3 minutes"
        echo "7. Access at: https://YOUR_USERNAME.github.io/dart-strike"
        ;;
        
    5)
        echo "📦 Creating deployment package..."
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ZIP_NAME="dart-strike-deploy-$TIMESTAMP.zip"
        
        # Create zip excluding unnecessary files
        zip -r "$ZIP_NAME" . \
            -x "*.git*" \
            -x "node_modules/*" \
            -x "dist/*" \
            -x "*.log" \
            -x "*.sh" \
            -x "generate_icons.py" \
            -x ".DS_Store" \
            -x "package-lock.json"
            
        echo "✅ Created $ZIP_NAME"
        echo ""
        echo "Upload this zip file to any web hosting service:"
        echo "- Netlify (drag & drop)"
        echo "- Vercel"
        echo "- Any web server"
        echo "- cPanel hosting"
        ;;
        
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "🎯 Need help? Check INSTALL.md for detailed instructions!"
