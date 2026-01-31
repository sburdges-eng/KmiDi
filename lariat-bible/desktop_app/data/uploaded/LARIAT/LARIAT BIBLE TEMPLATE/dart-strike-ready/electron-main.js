const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    icon: path.join(__dirname, 'icons/icon-512x512.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    },
    titleBarStyle: 'hiddenInset', // Mac-specific for better look
    backgroundColor: '#667eea',
    show: false
  });

  // Load the index.html
  mainWindow.loadFile('index.html');

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Create application menu
  const template = [
    {
      label: 'Dart Strike',
      submenu: [
        { label: 'About Dart Strike', role: 'about' },
        { type: 'separator' },
        { label: 'Preferences...', accelerator: 'Cmd+,', enabled: false },
        { type: 'separator' },
        { label: 'Hide Dart Strike', role: 'hide', accelerator: 'Cmd+H' },
        { label: 'Hide Others', role: 'hideothers', accelerator: 'Cmd+Option+H' },
        { label: 'Show All', role: 'unhide' },
        { type: 'separator' },
        { label: 'Quit Dart Strike', role: 'quit', accelerator: 'Cmd+Q' }
      ]
    },
    {
      label: 'Game',
      submenu: [
        { 
          label: 'New Game', 
          accelerator: 'Cmd+N',
          click: () => {
            mainWindow.webContents.executeJavaScript('game.newGame()');
          }
        },
        { 
          label: 'Add Player', 
          accelerator: 'Cmd+P',
          click: () => {
            mainWindow.webContents.executeJavaScript('game.showAddPlayerModal()');
          }
        },
        { type: 'separator' },
        { 
          label: 'Reset Pins', 
          accelerator: 'Cmd+R',
          click: () => {
            mainWindow.webContents.executeJavaScript('game.resetPins()');
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { label: 'Undo', role: 'undo' },
        { label: 'Redo', role: 'redo' },
        { type: 'separator' },
        { label: 'Cut', role: 'cut' },
        { label: 'Copy', role: 'copy' },
        { label: 'Paste', role: 'paste' },
        { label: 'Select All', role: 'selectall' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { label: 'Reload', role: 'reload', accelerator: 'Cmd+Shift+R' },
        { label: 'Force Reload', role: 'forcereload', accelerator: 'Cmd+Option+R' },
        { type: 'separator' },
        { label: 'Actual Size', role: 'resetzoom' },
        { label: 'Zoom In', role: 'zoomin', accelerator: 'Cmd+Plus' },
        { label: 'Zoom Out', role: 'zoomout', accelerator: 'Cmd+-' },
        { type: 'separator' },
        { label: 'Toggle Fullscreen', role: 'togglefullscreen', accelerator: 'Ctrl+Cmd+F' },
        { type: 'separator' },
        { label: 'Developer Tools', role: 'toggledevtools', accelerator: 'Cmd+Option+I' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { label: 'Minimize', role: 'minimize', accelerator: 'Cmd+M' },
        { label: 'Close', role: 'close', accelerator: 'Cmd+W' },
        { type: 'separator' },
        { label: 'Bring All to Front', role: 'front' }
      ]
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'How to Play',
          click: async () => {
            const { shell } = require('electron');
            await shell.openExternal('https://github.com/yourusername/dart-strike#how-to-play');
          }
        }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// App event handlers
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// Set application name
app.setName('Dart Strike');
