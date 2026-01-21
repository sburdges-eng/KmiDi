#!/usr/bin/env node
/**
 * Simple UI Test Runner
 * Opens browser and runs tests via console injection
 * 
 * Run: node scripts/test-ui-simple.js
 */

import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('🚀 Starting UI Tests...\n');
console.log('Opening browser at http://localhost:1420\n');
console.log('Please run the following in the browser console:\n');
console.log('='.repeat(60));

// Read and display the test script
const testScriptPath = path.join(__dirname, 'test-ui-all.js');
const testScript = fs.readFileSync(testScriptPath, 'utf8');

console.log('\nCopy and paste this into the browser console:\n');
console.log(testScript);
console.log('\n' + '='.repeat(60));
console.log('\nThen type: runUITests()\n');

// Try to open browser automatically
const platform = process.platform;
let openCommand;

if (platform === 'darwin') {
  openCommand = 'open';
} else if (platform === 'win32') {
  openCommand = 'start';
} else {
  openCommand = 'xdg-open';
}

console.log('Opening browser...\n');
const browser = spawn(openCommand, ['http://localhost:1420']);

browser.on('error', (err) => {
  console.error('Could not open browser automatically:', err.message);
  console.log('\nPlease manually open: http://localhost:1420');
});

browser.on('close', (code) => {
  console.log('\n✅ Browser opened. Run the test script in the console.');
});

// Also create a simple HTML file that can be bookmarked
const htmlTestPage = `<!DOCTYPE html>
<html>
<head>
  <title>UI Test - Copy Script</title>
  <style>
    body { font-family: monospace; padding: 20px; background: #1a1a1a; color: #0f0; }
    textarea { width: 100%; height: 400px; background: #000; color: #0f0; border: 1px solid #0f0; padding: 10px; }
    button { padding: 10px 20px; background: #0f0; color: #000; border: none; cursor: pointer; margin: 10px 5px; }
  </style>
</head>
<body>
  <h1>UI Test Script</h1>
  <p>1. Open <a href="http://localhost:1420" target="_blank">http://localhost:1420</a></p>
  <p>2. Press F12 to open DevTools</p>
  <p>3. Copy the script below and paste into Console</p>
  <p>4. Type: runUITests()</p>
  <textarea id="script" readonly>${testScript}</textarea>
  <button onclick="navigator.clipboard.writeText(document.getElementById('script').value); alert('Copied!')">Copy Script</button>
  <button onclick="window.open('http://localhost:1420', '_blank')">Open App</button>
</body>
</html>`;

fs.writeFileSync(path.join(__dirname, '..', 'test-ui-helper.html'), htmlTestPage);
console.log('\n📄 Helper page created: test-ui-helper.html');
console.log('   Open this file in your browser for easy copy-paste.\n');
