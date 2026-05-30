import { ChevronDown, Copy } from 'lucide-react'
import { QuerySource } from '../api'

type Props = {
  sources: QuerySource[]
  expanded: boolean
  onToggle: () => void
  onCopy: (text: string) => void
}

export function CitationList({ sources, expanded, onToggle, onCopy }: Props) {
  if (!sources.length) return null

  return (
    <div className="citation-block">
      <button type="button" className="citation-toggle" onClick={onToggle}>
        <span>Sources used ({sources.length})</span>
        <ChevronDown size={14} className={expanded ? 'citation-chevron open' : 'citation-chevron'} />
      </button>
      {expanded && (
        <div className="citation-list">
          {sources.map((src, idx) => (
            <details key={`${src.title}-${idx}`} className="citation-item" open={idx === 0}>
              <summary>
                <span className="citation-index">[{idx + 1}]</span>
                <span className="citation-title">{src.title}</span>
                <span className="citation-score">{(src.relevance_score * 100).toFixed(0)}%</span>
              </summary>
              <p>{src.content}</p>
              <button type="button" className="citation-copy" onClick={() => onCopy(src.content)}>
                <Copy size={12} />
                Copy excerpt
              </button>
            </details>
          ))}
        </div>
      )}
    </div>
  )
}
