import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Guide } from "./GuideNav";

// Eagerly bundle all production workflow markdown files as raw strings.
// Keys look like: "/Production_Workflows/Filename.md"
const markdownFiles = import.meta.glob("/Production_Workflows/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

type Props = {
  guide: Guide | null;
};

export function GuideViewer({ guide }: Props) {
  if (!guide) {
    return (
      <div className="guide-viewer empty">
        Select a guide to preview its contents.
      </div>
    );
  }

  // Normalize path for lookup: try multiple formats to handle different path styles
  // Manifest paths are like "Production_Workflows/Filename.md"
  // Glob keys are like "/Production_Workflows/Filename.md"
  let content: string | undefined;
  
  // Try with leading slash first (expected format from glob)
  const keyWithSlash = guide.path.startsWith("/") ? guide.path : `/${guide.path}`;
  content = markdownFiles[keyWithSlash];
  
  // If not found, try without leading slash
  if (!content) {
    const keyWithoutSlash = guide.path.startsWith("/") ? guide.path.substring(1) : guide.path;
    content = markdownFiles[keyWithoutSlash];
  }
  
  // If still not found, try to extract just the filename and match by filename
  if (!content) {
    const filename = guide.path.split("/").pop() || guide.path;
    const matchingKey = Object.keys(markdownFiles).find(key => key.endsWith(filename));
    if (matchingKey) {
      content = markdownFiles[matchingKey];
    }
  }

  if (!content) {
    return (
      <div className="guide-viewer empty">
        Preview unavailable: markdown not bundled. Make sure
        <code>{` ${guide.path} `}</code>
        is within <code>Production_Workflows/</code>.
        <br />
        <small style={{ fontSize: "0.8em", color: "#999", marginTop: "8px", display: "block" }}>
          Available keys: {Object.keys(markdownFiles).slice(0, 3).join(", ")}...
        </small>
      </div>
    );
  }

  return (
    <div className="guide-viewer">
      <div className="guide-viewer-header">
        <div>
          <div className="guide-viewer-title">{guide.title}</div>
          <div className="guide-viewer-slug">{guide.slug}</div>
        </div>
        <div className="guide-viewer-meta">
          {guide.topics.map((t) => (
            <span key={t} className="topic-chip">
              {t}
            </span>
          ))}
        </div>
      </div>
      <div className="guide-viewer-body">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    </div>
  );
}

export default GuideViewer;
