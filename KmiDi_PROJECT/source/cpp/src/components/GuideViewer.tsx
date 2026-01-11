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

  const key = guide.path.startsWith("/") ? guide.path : `/${guide.path}`;
  const content = markdownFiles[key];

  if (!content) {
    return (
      <div className="guide-viewer empty">
        Preview unavailable: markdown not bundled. Make sure
        <code>{` ${guide.path} `}</code>
        is within <code>Production_Workflows/</code>.
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
