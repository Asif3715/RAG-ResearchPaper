import { Settings, X } from 'lucide-react'
import { DismissLayer } from './DismissLayer'

type Props = {
  open: boolean
  onToggle: () => void
  searchMode: 'hybrid' | 'simple'
  onSearchMode: (mode: 'hybrid' | 'simple') => void
  topK: number
  onTopK: (value: number) => void
  rerank: boolean
  onRerank: (value: boolean) => void
}

export function AdvancedSettings({
  open,
  onToggle,
  searchMode,
  onSearchMode,
  topK,
  onTopK,
  rerank,
  onRerank,
}: Props) {
  return (
    <div className="advanced-settings">
      <button type="button" className="icon-button" onClick={onToggle} title="Search settings">
        <Settings size={16} />
      </button>
      {open && (
        <>
          <DismissLayer onDismiss={onToggle} />
          <div className="advanced-panel" role="dialog" aria-label="Advanced search settings">
            <div className="advanced-header">
              <span>Search settings</span>
              <button type="button" className="icon-button" onClick={onToggle}>
                <X size={16} />
              </button>
            </div>
            <div className="config-row">
              <span>Mode</span>
              <div className="segmented">
                <button type="button" className={searchMode === 'hybrid' ? 'seg-active' : ''} onClick={() => onSearchMode('hybrid')}>
                  Hybrid
                </button>
                <button type="button" className={searchMode === 'simple' ? 'seg-active' : ''} onClick={() => onSearchMode('simple')}>
                  Dense
                </button>
              </div>
            </div>
            <div className="config-row">
              <span>Top-K</span>
              <span className="mono-pill">{topK}</span>
            </div>
            <input type="range" min={1} max={10} value={topK} onChange={(e) => onTopK(Number(e.target.value))} />
            <label className="toggle-row">
              <span>Rerank results</span>
              <label className="toggle-switch">
                <input type="checkbox" checked={rerank} onChange={(e) => onRerank(e.target.checked)} />
                <span className="slider" />
              </label>
            </label>
          </div>
        </>
      )}
    </div>
  )
}
