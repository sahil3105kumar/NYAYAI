import { useState, useEffect, useRef, useCallback } from 'react'
import { analyzeFull, sendChatMessage } from '../api'
import FormattedMessage from './FormattedMessage'

/**
 * View 3: Two-pane analysis workspace.
 *   Left  (50%) — InLegalBERT analysis cards (LSI, RR, CJPE)
 *   Right (50%) — Context-aware Graph RAG chatbot scoped to uploaded doc
 */
export default function AnalysisWorkspace({ extractedText, onBack }) {
  // ─── Analysis State ────────────────────────────────────────────────
  const [analysisData, setAnalysisData] = useState(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)
  const [analysisError, setAnalysisError] = useState(null)

  // ─── Chat State ────────────────────────────────────────────────────
  const [messages, setMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [threadId] = useState(() => `analysis_${Date.now()}`)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Run full analysis on mount if we have text
  useEffect(() => {
    if (!extractedText) return
    let cancelled = false

    async function run() {
      setAnalysisLoading(true)
      setAnalysisError(null)
      try {
        const data = await analyzeFull(extractedText)
        if (!cancelled) setAnalysisData(data)
      } catch (err) {
        if (!cancelled) setAnalysisError(err.message)
      } finally {
        if (!cancelled) setAnalysisLoading(false)
      }
    }
    run()
    return () => { cancelled = true }
  }, [extractedText])

  // ─── Chat handlers ─────────────────────────────────────────────────
  const handleSendChat = useCallback(async () => {
    const text = chatInput.trim()
    if (!text || chatLoading) return

    const userMsg = { role: 'user', content: text, id: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setChatInput('')
    setChatLoading(true)

    try {
      const { reply } = await sendChatMessage(text, threadId)
      setMessages(prev => [...prev, { role: 'assistant', content: reply, id: Date.now() + 1 }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'error', content: err.message, id: Date.now() + 1 }])
    } finally {
      setChatLoading(false)
    }
  }, [chatInput, chatLoading, threadId])

  const handleChatKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendChat()
    }
  }

  const ConfidenceBar = ({ value, label }) => (
    <div className="analysis-confidence-row">
      <span className="analysis-confidence-label">{label}</span>
      <div className="analysis-confidence-track">
        <div
          className="analysis-confidence-fill"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
      <span className="analysis-confidence-value">{Math.round(value * 100)}%</span>
    </div>
  )

  return (
    <div className="analysis-workspace">
      {/* Top Bar */}
      <header className="analysis-header">
        <button className="analysis-back-btn" onClick={onBack} id="analysis-back">
          ← Back to Inspector
        </button>
        <h2 className="analysis-header-title">Deep Legal Analysis & Case Chat</h2>
        <div className="analysis-header-badge">InLegalBERT + Graph RAG</div>
      </header>

      {/* Split Panes */}
      <div className="analysis-panes">
        {/* ── Left Pane: InLegalBERT Analysis ── */}
        <div className="analysis-left-pane">
          <div className="analysis-pane-header">
            <h3>InLegalBERT Analysis</h3>
          </div>

          {analysisLoading && (
            <div className="analysis-loading">
              <div className="analysis-spinner" />
              <p>Running InLegalBERT models…</p>
            </div>
          )}

          {analysisError && (
            <div className="analysis-error-card">
              <p>{analysisError}</p>
            </div>
          )}

          {analysisData && (
            <div className="analysis-cards">
              {/* LSI Card */}
              <div className="analysis-card analysis-card--lsi">
                <h4>Legal Statute Identification</h4>
                <p className="analysis-card-desc">Applicable BNS/IPC Sections</p>
                <div className="analysis-card-body">
                  {analysisData.lsi_predictions?.map((p, i) => (
                    <ConfidenceBar key={i} label={p.statute} value={p.confidence} />
                  ))}
                  {!analysisData.lsi_predictions?.length && (
                    <p className="analysis-empty">No statutes detected</p>
                  )}
                </div>
              </div>

              {/* RR Card */}
              <div className="analysis-card analysis-card--rr">
                <h4>Rhetorical Roles</h4>
                <p className="analysis-card-desc">Sentence-level structural analysis</p>
                <div className="analysis-card-body analysis-rr-list">
                  {analysisData.rr_predictions?.slice(0, 10).map((p, i) => (
                    <div key={i} className="analysis-rr-item">
                      <span className={`analysis-rr-badge analysis-rr-badge--${p.rhetorical_role.toLowerCase().replace(/\s+/g, '-')}`}>
                        {p.rhetorical_role}
                      </span>
                      <span className="analysis-rr-sentence">{p.sentence.slice(0, 80)}…</span>
                      <span className="analysis-rr-conf">{Math.round(p.confidence * 100)}%</span>
                    </div>
                  ))}
                  {!analysisData.rr_predictions?.length && (
                    <p className="analysis-empty">No roles classified</p>
                  )}
                </div>
              </div>

              {/* CJPE Card */}
              <div className="analysis-card analysis-card--cjpe">
                <h4>Case Judgment Prediction</h4>
                <p className="analysis-card-desc">Predicted outcome based on facts</p>
                <div className="analysis-card-body">
                  {analysisData.cjpe_prediction && (
                    <div className="analysis-cjpe-result">
                      <div className={`analysis-cjpe-outcome analysis-cjpe-outcome--${analysisData.cjpe_prediction.outcome.includes('Accepted') ? 'accepted' : 'rejected'}`}>
                        {analysisData.cjpe_prediction.outcome}
                      </div>
                      <ConfidenceBar label="Confidence" value={analysisData.cjpe_prediction.confidence} />
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {!analysisLoading && !analysisError && !analysisData && (
            <div className="analysis-empty-state">
              <p>No text available for analysis. Upload a PDF in the Inspector first.</p>
            </div>
          )}
        </div>

        {/* ── Right Pane: Graph RAG Chatbot ── */}
        <div className="analysis-right-pane">
          <div className="analysis-pane-header">
            <h3>Case Document Chat</h3>
            <span className="analysis-pane-badge">Graph RAG</span>
          </div>

          <div className="analysis-chat-messages">
            {messages.length === 0 && (
              <div className="analysis-chat-empty">
                <p>Ask questions about the uploaded case file.</p>
                <p className="analysis-chat-hint">This chatbot queries the Neo4j knowledge graph built from your document.</p>
              </div>
            )}
            {messages.map(msg => (
              <div key={msg.id} className={`chat-message chat-message--${msg.role}`}>
                <div className="chat-message-avatar">
                  {msg.role === 'user' ? 'U' : msg.role === 'error' ? '!' : 'N'}
                </div>
                <div className="chat-message-bubble">
                  <div className="chat-message-content">
                    {msg.role === 'assistant'
                      ? <FormattedMessage text={msg.content} />
                      : msg.content}
                  </div>
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="chat-message chat-message--assistant">
                <div className="chat-message-avatar">N</div>
                <div className="chat-message-bubble">
                  <div className="chat-typing-indicator">
                    <span /><span /><span />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="analysis-chat-input-area">
            <textarea
              className="chat-input"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={handleChatKeyDown}
              placeholder="Ask about this case file…"
              rows={1}
              disabled={chatLoading}
              id="analysis-chat-input"
            />
            <button
              className="chat-send-btn"
              onClick={handleSendChat}
              disabled={!chatInput.trim() || chatLoading}
              id="analysis-chat-send"
            >
              ›
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
