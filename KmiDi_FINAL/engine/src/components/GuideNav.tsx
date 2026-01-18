import { useMemo, useState } from "react";
import guides from "../manifest.json";

export type Guide = {
  title: string;
  slug: string;
  path: string;
  topics: string[];
};

type Props = {
  onSelect?: (guide: Guide) => void;
};

export function GuideNav({ onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [topicFilter, setTopicFilter] = useState<string | null>(null);

  const allTopics = useMemo(() => {
    const topics = new Set<string>();
    guides.forEach((guide: Guide) => guide.topics.forEach((t: string) => topics.add(t)));
    return Array.from(topics).sort((a, b) => a.localeCompare(b));
  }, []);

  const filteredGuides = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (guides as Guide[]).filter((guide) => {
      const matchesQuery =
        !q ||
        guide.title.toLowerCase().includes(q) ||
        guide.slug.toLowerCase().includes(q) ||
        guide.topics.some((t) => t.toLowerCase().includes(q));
      const matchesTopic = !topicFilter || guide.topics.includes(topicFilter);
      return matchesQuery && matchesTopic;
    });
  }, [query, topicFilter]);

  const handleCopyPath = async (path: string) => {
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(path);
      } else {
        throw new Error("Clipboard API unavailable");
      }
    } catch (err) {
      console.warn("Could not copy path", err);
      alert(`Path: ${path}`);
    }
  };

  return (
    <div className="guide-nav">
      <div className="guide-controls">
        <input
          className="guide-search"
          placeholder="Search guides by title or topic…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="guide-topics">
          <button
            className={!topicFilter ? "topic-pill active" : "topic-pill"}
            onClick={() => setTopicFilter(null)}
          >
            All
          </button>
          {allTopics.map((topic) => (
            <button
              key={topic}
              className={
                topicFilter === topic ? "topic-pill active" : "topic-pill"
              }
              onClick={() => setTopicFilter(topic)}
            >
              {topic}
            </button>
          ))}
        </div>
        <div className="guide-count">
          Showing {filteredGuides.length} of {(guides as Guide[]).length} guides
        </div>
      </div>

      <div className="guide-list">
        {filteredGuides.map((guide) => (
          <div key={guide.slug} className="guide-card">
            <div className="guide-card-header">
              <div>
                <div className="guide-title">{guide.title}</div>
                <div className="guide-slug">{guide.slug}</div>
              </div>
              <div className="guide-actions">
                <a
                  href={`/${encodeURI(guide.path)}`}
                  target="_blank"
                  rel="noreferrer"
                  className="guide-link"
                >
                  Open
                </a>
                <button
                  type="button"
                  className="copy-btn"
                  onClick={() => handleCopyPath(guide.path)}
                >
                  Copy path
                </button>
                {onSelect && (
                  <button
                    type="button"
                    className="preview-btn"
                    onClick={() => onSelect(guide)}
                  >
                    Preview
                  </button>
                )}
              </div>
            </div>
            <div className="guide-topics-row">
              {guide.topics.map((topic) => (
                <span key={topic} className="topic-chip">
                  {topic}
                </span>
              ))}
            </div>
          </div>
        ))}
        {filteredGuides.length === 0 && (
          <div className="guide-empty">No guides match that search.</div>
        )}
      </div>
    </div>
  );
}

export default GuideNav;

