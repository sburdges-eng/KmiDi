import { CommandPalette } from './CommandPalette';
import { TaxonomyTree } from './TaxonomyTree';
import { InterpretiveSlider } from './InterpretiveSlider';

interface TaxonomyNode {
  id: string;
  label: string;
  children?: TaxonomyNode[];
  keywords: string[];
}

interface SideAPaneProps {
  taxonomyNodes: TaxonomyNode[];
  nodeWeights: Map<string, number>;
  nlpDetectedNodes: Set<string>;
  pinnedNodes: Set<string>;
  searchQuery: string;
  interpretiveLevel: number;
  onSearchChange: (query: string) => void;
  onToggleNode: (id: string) => void;
  onPinNode: (id: string) => void;
  onInterpretiveLevelChange: (level: number) => void;
}

export function SideAPane({
  taxonomyNodes, nodeWeights, nlpDetectedNodes, pinnedNodes,
  searchQuery, interpretiveLevel,
  onSearchChange, onToggleNode, onPinNode, onInterpretiveLevelChange,
}: SideAPaneProps) {
  const activeCount = nodeWeights.size;

  return (
    <div className="umi-side-a">
      <div className="umi-side-header">
        <h2 className="umi-side-title">Musical Commands</h2>
      </div>
      <CommandPalette
        searchQuery={searchQuery}
        onSearchChange={onSearchChange}
        activeCount={activeCount}
      />
      <div className="umi-tree-scroll">
        <TaxonomyTree
          nodes={taxonomyNodes}
          nodeWeights={nodeWeights}
          nlpDetectedNodes={nlpDetectedNodes}
          pinnedNodes={pinnedNodes}
          searchQuery={searchQuery}
          onToggleNode={onToggleNode}
          onPinNode={onPinNode}
        />
      </div>
      <InterpretiveSlider
        value={interpretiveLevel}
        onChange={onInterpretiveLevelChange}
      />
    </div>
  );
}
