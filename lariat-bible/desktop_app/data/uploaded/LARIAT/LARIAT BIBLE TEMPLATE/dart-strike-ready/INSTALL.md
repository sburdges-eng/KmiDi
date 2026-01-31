# 📱 iOS & Mac Installation Guide for Dart Strike

## 🎯 Quick Start

### Option 1: Install on iPhone/iPad (PWA - Recommended for Mobile)

**No App Store needed! Install directly from Safari:**

1. **Open Safari** on your iPhone/iPad
2. **Visit the app**: Go to your hosted URL or local server
3. **Tap Share button** (square with arrow pointing up)
4. **Scroll down and tap "Add to Home Screen"**
5. **Name it** "Dart Strike" and tap "Add"
6. **Done!** The app icon appears on your home screen

**Features of iOS PWA:**
- Works offline after first load
- Full screen experience (no browser UI)
- App icon on home screen
- Fast loading
- No App Store approval needed

---

## 💻 Mac Desktop App Installation

### Option 1: Download Pre-built App (Easiest)

If you have the built `.dmg` file:

1. **Download** `Dart Strike.dmg`
2. **Double-click** to open the installer
3. **Drag** Dart Strike to Applications folder
4. **First time opening:**
   - Right-click on Dart Strike in Applications
   - Click "Open" (bypasses Gatekeeper)
   - Click "Open" in the dialog
5. **Enjoy!** The app is now installed

### Option 2: Build from Source

**Requirements:**
- Node.js 18+ installed
- npm or yarn package manager

**Steps:**

```bash
# 1. Clone or download the project
cd dart-strike

# 2. Install dependencies
npm install

# 3. Run development version
npm run electron

# OR build for distribution
npm run dist-mac

# 4. Find your app in the 'dist' folder
# - Dart Strike.dmg (installer)
# - Dart Strike.app (application)
```

---

## 🚀 Hosting for iOS PWA

To use the PWA on your iPhone, you need to host it. Here are your options:

### Option 1: Local Network (Testing)

```bash
# In the dart-strike folder
npm start
# OR
python3 -m http.server 8080
```

Then on your iPhone (same WiFi):
1. Open Safari
2. Go to `http://[your-computer-ip]:8080`
3. Add to Home Screen

### Option 2: Free Hosting Services

**GitHub Pages** (Free & Easy):
```bash
# Push to GitHub repo
# Enable GitHub Pages in Settings
# Access at: https://[username].github.io/dart-strike
```

**Netlify** (Drag & Drop):
1. Visit [netlify.com](https://netlify.com)
2. Drag your dart-strike folder to browser
3. Get instant URL
4. Share with anyone!

**Vercel** (One-click deploy):
```bash
npm i -g vercel
vercel
# Follow prompts
```

---

## 📲 Features by Platform

### iOS PWA Features:
- ✅ Offline play
- ✅ Home screen icon
- ✅ Full screen
- ✅ Touch optimized
- ✅ No App Store needed
- ✅ Auto-updates when online
- ⚠️ Limited iOS APIs (no push notifications)

### Mac Desktop Features:
- ✅ Native app experience
- ✅ Menu bar integration
- ✅ Keyboard shortcuts
- ✅ Window management
- ✅ Dock icon
- ✅ File system access
- ✅ Better performance

---

## 🛠 Troubleshooting

### iOS Issues:

**"Add to Home Screen" not working:**
- Must use Safari (not Chrome/Firefox)
- Clear Safari cache and try again
- Ensure you're on HTTPS or localhost

**App won't work offline:**
- Open app while online first
- Let it fully load once
- Service worker will cache for offline

### Mac Issues:

**"App can't be opened" security warning:**
1. Go to System Preferences > Security & Privacy
2. Click "Open Anyway"
OR
- Right-click app > Open

**Build fails:**
```bash
# Clear and rebuild
rm -rf node_modules dist
npm install
npm run dist-mac
```

---

## 🔧 Development Tips

### Testing iOS PWA locally:
```bash
# Get your local IP
ifconfig | grep "inet " | grep -v 127.0.0.1

# Start server
npm start

# On iPhone Safari, visit:
# http://[your-ip]:8080
```

### Debug iOS PWA:
1. Connect iPhone to Mac via cable
2. Open Safari on Mac
3. Develop menu > [Your iPhone] > [Web App]

### Custom iOS Icons:
- Place icons in `/icons` folder
- Sizes: 180x180 (main), 192x192, 512x512
- Format: PNG with transparency

---

## 📦 Project Structure

```
dart-strike/
├── index.html          # Main app
├── app.js             # Game logic
├── bowling-scorer.js  # Scoring engine
├── styles.css         # Styling
├── manifest.json      # PWA config
├── service-worker.js  # Offline support
├── icons/             # App icons
├── electron-main.js   # Mac app wrapper
├── package.json       # Dependencies
└── build-mac.sh      # Mac build script
```

---

## 🎮 Quick Commands

```bash
# Install dependencies
npm install

# Run web server (for iOS)
npm start

# Run Mac app (development)
npm run electron

# Build Mac app
npm run dist-mac

# Generate icons
python3 generate_icons.py
```

---

## 🚢 Deployment Checklist

### For iOS PWA:
- [ ] Test on real iPhone
- [ ] Verify offline mode works
- [ ] Check all icon sizes
- [ ] Test "Add to Home Screen"
- [ ] Deploy to HTTPS host

### For Mac App:
- [ ] Test on Intel & Apple Silicon
- [ ] Sign app (for distribution)
- [ ] Create DMG installer
- [ ] Test auto-updater
- [ ] Upload to releases

---

## 📱 Share Your App

### Share iOS PWA:
1. Host on any web server
2. Share the URL
3. Users add to home screen

### Share Mac App:
1. Build with `npm run dist-mac`
2. Share the `.dmg` file
3. Users install normally

---

## 💡 Pro Tips

1. **For iOS**: Use meta viewport tag for better mobile experience
2. **For Mac**: Add auto-updater for easy updates
3. **Both**: Keep file sizes small for fast loading
4. **PWA**: Update service worker version to force cache refresh
5. **Testing**: Always test on real devices, not just simulators

---

## 📞 Support

- Issues: Check console logs (Safari/Chrome DevTools)
- Updates: Increment version in package.json
- Help: Review the README.md for game features

Enjoy Dart Strike! 🎯🎳
