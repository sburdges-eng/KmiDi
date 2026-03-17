import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import AppConsole from './AppConsole';
import ErrorBoundary from './ErrorBoundary';
import './index.css';

// Default shell: AppConsole. App.tsx is legacy/alternate and is not mounted here.

const root = document.getElementById('root');

if (!root) {
  throw new Error('Missing #root element in index.html');
}

createRoot(root).render(
  <StrictMode>
    <ErrorBoundary>
      <AppConsole />
    </ErrorBoundary>
  </StrictMode>,
);
