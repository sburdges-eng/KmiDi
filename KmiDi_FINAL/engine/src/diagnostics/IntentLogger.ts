type FrameSnapshot = unknown;

interface IntentLogger {
  logUIBoundary(frame: FrameSnapshot): void;
  logError(stage: string, frame: FrameSnapshot, message: string): void;
}

const logger: IntentLogger = {
  logUIBoundary(frame) {
    // Intentionally lightweight: diagnostics only at the UI boundary.
    console.debug("[IntentIR][ui_boundary]", frame);
  },
  logError(stage, frame, message) {
    console.error("[IntentIR][error]", { stage, frame, message });
  },
};

export function getLogger(): IntentLogger {
  return logger;
}
