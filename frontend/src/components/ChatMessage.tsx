import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { QuerySource } from '../api'
import { CitationList } from './CitationList'

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
        <div className="message-bubble user">{message.content}</div>
      </div>
    )
  }

  return (
    <div className="message-row assistant">
      <div className="message-bubble assistant prose">
        <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
          {message.content}
        </ReactMarkdown>
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
      <div className="message-bubble assistant prose">
        {content ? (
          <>
            <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
              {content}
            </ReactMarkdown>
            <span className="stream-cursor" aria-hidden />
          </>
        ) : (
          <div className="stream-placeholder">Retrieving relevant passages…</div>
        )}
      </div>
    </div>
  )
}
