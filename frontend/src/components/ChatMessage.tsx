import { Copy } from 'lucide-react'
import { QuerySource } from '../api'
import { CitationList } from './CitationList'
import { MarkdownContent } from './MarkdownContent'

export type Message = { role: 'user' | 'assistant'; content: string; sources?: QuerySource[] }

type Props = {
  message: Message
  citationsExpanded: boolean
  onToggleCitations: () => void
  onCopy: (text: string) => void
}

export function ChatMessage({ message, citationsExpanded, onToggleCitations, onCopy }: Props) {
  if (message.role === 'user') {
    return (
      <div className="message-row user">
        <div className="message-inner">
          <div className="message-bubble user">
            <div className="user-text">{message.content}</div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="message-row assistant">
      <div className="message-inner">
        <div className="assistant-label">
          <span className="assistant-avatar" aria-hidden>
            <span className="assistant-avatar-dot" />
          </span>
          <span>Answer</span>
        </div>
        <div className="message-body assistant">
          <MarkdownContent content={message.content} />
          <div className="message-actions">
            <button type="button" className="message-action-btn" onClick={() => onCopy(message.content)}>
              <Copy size={14} />
              Copy
            </button>
          </div>
          {message.sources && message.sources.length > 0 && (
            <CitationList
              sources={message.sources}
              expanded={citationsExpanded}
              onToggle={onToggleCitations}
              onCopy={onCopy}
            />
          )}
        </div>
      </div>
    </div>
  )
}

type StreamingProps = {
  content: string
  loading: boolean
}

export function StreamingMessage({ content, loading }: StreamingProps) {
  if (!content && !loading) return null

  return (
    <div className="message-row assistant">
      <div className="message-inner">
        <div className="assistant-label">
          <span className="assistant-avatar streaming" aria-hidden>
            <span className="assistant-avatar-dot" />
          </span>
          <span>{content ? 'Writing…' : 'Retrieving…'}</span>
        </div>
        <div className="message-body assistant">
          {content ? (
            <>
              <MarkdownContent content={content} />
              <span className="stream-cursor" aria-hidden />
            </>
          ) : (
            <div className="stream-placeholder">
              <span className="stream-shimmer" />
              Finding relevant passages in your papers…
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
