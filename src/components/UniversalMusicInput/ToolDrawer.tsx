import { useState, type ReactNode } from 'react';

interface ToolDrawerProps {
  children: ReactNode;
}

export function ToolDrawer({ children }: ToolDrawerProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        className="umi-drawer-toggle"
        onClick={() => setIsOpen(v => !v)}
        title={isOpen ? 'Hide tools' : 'Show Mix/Transport/Lyrics tools'}
      >
        {isOpen ? '▼ Tools' : '▲ Tools'}
      </button>
      {isOpen && (
        <div className="umi-drawer">
          {children}
        </div>
      )}
    </>
  );
}
