import { useState, useCallback, useMemo } from 'react';
import { TaxonomyNodeChip } from './TaxonomyNodeChip';

interface TaxonomyNode {
  id: string;
  label: string;
  children?: TaxonomyNode[];
  keywords: string[];
}

interface TaxonomyTreeProps {
  /** The full taxonomy tree data */
  nodes: TaxonomyNode[];
  /** Weights for active nodes: nodeId → weight (0-1) */
  nodeWeights: Map<string, number>;
  /** Nodes detected by NLP (displayed differently) */
  nlpDetectedNodes: Set<string>;
  /** Pinned parameters */
  pinnedNodes: Set<string>;
  /** Search query for filtering */
  searchQuery: string;
  onToggleNode: (id: string) => void;
  onPinNode?: (id: string) => void;
}

export function TaxonomyTree({
  nodes, nodeWeights, nlpDetectedNodes, pinnedNodes,
  searchQuery, onToggleNode, onPinNode,
}: TaxonomyTreeProps) {
  return (
    <div className="umi-taxonomy-tree">
      {nodes.map(node => (
        <TreeBranch
          key={node.id}
          node={node}
          depth={0}
          nodeWeights={nodeWeights}
          nlpDetectedNodes={nlpDetectedNodes}
          pinnedNodes={pinnedNodes}
          searchQuery={searchQuery}
          onToggleNode={onToggleNode}
          onPinNode={onPinNode}
        />
      ))}
    </div>
  );
}

interface TreeBranchProps {
  node: TaxonomyNode;
  depth: number;
  nodeWeights: Map<string, number>;
  nlpDetectedNodes: Set<string>;
  pinnedNodes: Set<string>;
  searchQuery: string;
  onToggleNode: (id: string) => void;
  onPinNode?: (id: string) => void;
}

function TreeBranch({
  node, depth, nodeWeights, nlpDetectedNodes, pinnedNodes,
  searchQuery, onToggleNode, onPinNode,
}: TreeBranchProps) {
  const [isExpanded, setIsExpanded] = useState(depth === 0);

  const hasChildren = node.children && node.children.length > 0;
  const isLeaf = !hasChildren;

  // Check if this node or any descendant matches the search
  const matchesSearch = useMemo(() => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    return nodeMatchesSearch(node, q);
  }, [node, searchQuery]);

  // Auto-expand when search matches a descendant
  const shouldShow = matchesSearch;
  const shouldAutoExpand = searchQuery.length > 0 && matchesSearch;

  const toggle = useCallback(() => setIsExpanded(v => !v), []);

  // Count active children for the badge
  const activeChildCount = useMemo(() => {
    if (!hasChildren) return 0;
    return countActiveDescendants(node, nodeWeights);
  }, [node, nodeWeights, hasChildren]);

  if (!shouldShow) return null;

  const expanded = isExpanded || shouldAutoExpand;

  if (isLeaf) {
    return (
      <TaxonomyNodeChip
        id={node.id}
        label={node.label}
        weight={nodeWeights.get(node.id)}
        isNlpDetected={nlpDetectedNodes.has(node.id)}
        isPinned={pinnedNodes.has(node.id)}
        onToggle={onToggleNode}
        onPin={onPinNode}
      />
    );
  }

  return (
    <div className="umi-tree-branch" data-depth={depth}>
      <button
        type="button"
        className={`umi-tree-header${expanded ? ' expanded' : ''}`}
        onClick={toggle}
      >
        <span className="umi-tree-arrow">{expanded ? '▼' : '▶'}</span>
        <span className="umi-tree-label">{node.label}</span>
        {activeChildCount > 0 && (
          <span className="umi-tree-badge">{activeChildCount}</span>
        )}
      </button>
      {expanded && node.children && (
        <div className="umi-tree-children">
          {node.children.map(child => (
            <TreeBranch
              key={child.id}
              node={child}
              depth={depth + 1}
              nodeWeights={nodeWeights}
              nlpDetectedNodes={nlpDetectedNodes}
              pinnedNodes={pinnedNodes}
              searchQuery={searchQuery}
              onToggleNode={onToggleNode}
              onPinNode={onPinNode}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function nodeMatchesSearch(node: TaxonomyNode, query: string): boolean {
  if (node.label.toLowerCase().includes(query)) return true;
  if (node.keywords.some(k => k.toLowerCase().includes(query))) return true;
  if (node.children?.some(c => nodeMatchesSearch(c, query))) return true;
  return false;
}

function countActiveDescendants(node: TaxonomyNode, weights: Map<string, number>): number {
  let count = 0;
  if (!node.children) return weights.has(node.id) ? 1 : 0;
  for (const child of node.children) {
    count += countActiveDescendants(child, weights);
  }
  return count;
}
