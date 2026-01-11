# PWA Wrapper for Mobile Access

> Progressive Web App configuration for iDAW mobile access.

## Overview

Transform the Streamlit app into a Progressive Web App (PWA) for:

- Offline access
- App-like experience on mobile
- Home screen installation
- Push notifications

## Implementation

### 1. Manifest File

```json
// static/manifest.json
{
  "name": "iDAW - Intelligent DAW",
  "short_name": "iDAW",
  "description": "AI-powered music composition assistant",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1a1a1a",
  "theme_color": "#4A90D9",
  "orientation": "any",
  "icons": [
    {
      "src": "/static/icons/icon-72.png",
      "sizes": "72x72",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-96.png",
      "sizes": "96x96",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-128.png",
      "sizes": "128x128",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-144.png",
      "sizes": "144x144",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/static/icons/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ],
  "categories": ["music", "productivity"],
  "screenshots": [
    {
      "src": "/static/screenshots/desktop.png",
      "sizes": "1280x720",
      "type": "image/png",
      "form_factor": "wide"
    },
    {
      "src": "/static/screenshots/mobile.png",
      "sizes": "720x1280",
      "type": "image/png",
      "form_factor": "narrow"
    }
  ]
}
```

### 2. Service Worker

```javascript
// static/sw.js
const CACHE_NAME = 'idaw-v1';
const STATIC_ASSETS = [
  '/',
  '/static/manifest.json',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/css/main.css'
];

// Install event
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
  );
});

// Fetch event with network-first strategy
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Clone and cache successful responses
        if (response.status === 200) {
          const responseClone = response.clone();
          caches.open(CACHE_NAME)
            .then((cache) => cache.put(event.request, responseClone));
        }
        return response;
      })
      .catch(() => {
        // Fallback to cache
        return caches.match(event.request);
      })
  );
});

// Activate event
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
});
```

### 3. HTML Head Tags

```html
<!-- Add to index.html -->
<head>
  <!-- PWA Meta Tags -->
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="theme-color" content="#4A90D9">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="iDAW">

  <!-- Manifest -->
  <link rel="manifest" href="/static/manifest.json">

  <!-- Icons -->
  <link rel="icon" href="/static/icons/icon-192.png">
  <link rel="apple-touch-icon" href="/static/icons/icon-192.png">

  <!-- Service Worker Registration -->
  <script>
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/static/sw.js')
        .then((reg) => console.log('SW registered:', reg.scope))
        .catch((err) => console.log('SW registration failed:', err));
    }
  </script>
</head>
```

### 4. Streamlit PWA Integration

```python
# pwa_components.py
import streamlit.components.v1 as components

def add_pwa_support():
    """Add PWA support to Streamlit app."""
    pwa_script = """
    <script>
    // Check if app is installed
    let deferredPrompt;
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        showInstallButton();
    });

    function showInstallButton() {
        const btn = document.createElement('button');
        btn.textContent = 'Install App';
        btn.className = 'pwa-install-btn';
        btn.onclick = () => {
            deferredPrompt.prompt();
            deferredPrompt.userChoice.then((choice) => {
                if (choice.outcome === 'accepted') {
                    console.log('User installed PWA');
                }
                deferredPrompt = null;
            });
        };
        document.body.appendChild(btn);
    }
    </script>
    <style>
    .pwa-install-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        background: #4A90D9;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        z-index: 9999;
    }
    </style>
    """
    components.html(pwa_script, height=0)
```

## Mobile Optimizations

### Responsive Layout
- Use `st.columns` for adaptive layouts
- Hide sidebar on mobile with CSS
- Touch-friendly button sizes (min 44x44px)

### Performance
- Lazy load heavy components
- Compress images
- Minimize JavaScript

### Offline Support
- Cache critical assets
- Show offline indicator
- Queue actions for sync

## Testing

### Lighthouse Audit
```bash
# Run Lighthouse PWA audit
lighthouse https://idaw.streamlit.app --only-categories=pwa
```

### Mobile Testing
- Chrome DevTools device mode
- Real device testing (iOS/Android)
- Safari Web Inspector

## Deployment

PWA works with any hosting that supports:

- HTTPS (required for service workers)
- Static file serving
- Custom headers

Supported platforms:

- Streamlit Cloud ✓
- Railway ✓
- Render ✓
- Vercel ✓
- Netlify ✓

---

*"The audience doesn't hear 'borrowed from Dorian.' They hear 'that part made me cry.'"*
