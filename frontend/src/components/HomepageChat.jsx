import { useState, useRef, useEffect, useCallback } from 'react'
import { sendChatMessage } from '../api'
import FormattedMessage from './FormattedMessage'

/**
 * View 1: Full-screen general legal chatbot landing page.
 * Provides general legal Q&A via the LangGraph agent and a prominent
 * navigation button to the PDF Error Inspector.
 */
export default function HomepageChat({ onNavigateToInspector }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [threadId] = useState(() => `session_${Date.now()}`)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text || isLoading) return

    const userMsg = { role: 'user', content: text, id: Date.now() }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    try {
      const { reply } = await sendChatMessage(text, threadId)
      const botMsg = { role: 'assistant', content: reply, id: Date.now() + 1 }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      const errMsg = { role: 'error', content: err.message, id: Date.now() + 1 }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setIsLoading(false)
    }
  }, [input, isLoading, threadId])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const suggestedQuestions = [
    "What are the key differences between IPC and BNS?",
    "Draft a bail application for a theft case",
    "Explain Section 302 of IPC",
    "What is the procedure for filing an FIR?",
  ]

  return (
    <div className="homepage-chat">
      <header className="chat-header">
        <div className="chat-header-brand">
          <span className="chat-brand-text">NyayAI</span>
        </div>
        <button
          className="chat-nav-btn"
          onClick={onNavigateToInspector}
          id="nav-to-inspector"
        >
          Go to PDF Error Inspector
        </button>
      </header>

      <div className="chat-body">
        {messages.length === 0 ? (
          <div className="chat-welcome">
            <h1 className="chat-welcome-title">NyayAI Legal Assistant</h1>
            <p className="chat-welcome-subtitle">
              Your AI-powered Indian legal companion. Ask questions about laws,
              draft legal documents, or analyze case files.
            </p>
            <div className="chat-suggestions">
              {suggestedQuestions.map((q, i) => (
                <button
                  key={i}
                  className="chat-suggestion-chip"
                  onClick={() => setInput(q)}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-messages">
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
            {isLoading && (
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
        )}
      </div>

      <div className="chat-input-area">
        <div className="chat-input-container">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a legal question…"
            rows={1}
            disabled={isLoading}
            id="chat-input"
          />
          <button
            className="chat-send-btn"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            id="chat-send"
          >
            ›
          </button>
        </div>
        <p className="chat-disclaimer">
          NyayAI provides AI-assisted legal information. Always consult a qualified lawyer for legal advice.
        </p>
      </div>
    </div>
  )
}
