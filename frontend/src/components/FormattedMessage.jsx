
export default function FormattedMessage({ text }) {
  if (!text) return null


  if (text.trim().startsWith('{') || text.trim().startsWith('[')) {
    try {
      const parsed = JSON.parse(text)
      return (
        <pre className="formatted-json">
          {JSON.stringify(parsed, null, 2)}
        </pre>
      )
    } catch {
      // Not valid JSON, render as text
    }
  }

  const lines = text.split('\n')
  const elements = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const key = i

    // Horizontal rule
    if (/^---+$/.test(line.trim())) {
      elements.push(<hr key={key} className="formatted-hr" />)
      continue
    }

    // Headings
    const h3Match = line.match(/^###\s+(.+)/)
    if (h3Match) {
      elements.push(<h4 key={key} className="formatted-h3">{renderInline(h3Match[1])}</h4>)
      continue
    }
    const h2Match = line.match(/^##\s+(.+)/)
    if (h2Match) {
      elements.push(<h3 key={key} className="formatted-h2">{renderInline(h2Match[1])}</h3>)
      continue
    }
    const h1Match = line.match(/^#\s+(.+)/)
    if (h1Match) {
      elements.push(<h2 key={key} className="formatted-h1">{renderInline(h1Match[1])}</h2>)
      continue
    }

    // List items (unordered: -, *)
    const ulMatch = line.match(/^\s*[-*]\s+(.+)/)
    if (ulMatch) {
      elements.push(
        <div key={key} className="formatted-li">
          <span className="formatted-bullet">•</span>
          <span>{renderInline(ulMatch[1])}</span>
        </div>
      )
      continue
    }

    // Numbered list items
    const olMatch = line.match(/^\s*(\d+)\.\s+(.+)/)
    if (olMatch) {
      elements.push(
        <div key={key} className="formatted-li">
          <span className="formatted-num">{olMatch[1]}.</span>
          <span>{renderInline(olMatch[2])}</span>
        </div>
      )
      continue
    }

    // Empty line = paragraph break
    if (line.trim() === '') {
      elements.push(<div key={key} className="formatted-break" />)
      continue
    }

    // Regular paragraph
    elements.push(<p key={key} className="formatted-p">{renderInline(line)}</p>)
  }

  return <div className="formatted-message">{elements}</div>
}

/**
 * Render inline formatting: **bold**, *italic*, `code`
 */
function renderInline(text) {
  const parts = []
  // Regex matches **bold**, *italic*, `code` in order
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`)/g
  let lastIndex = 0
  let match

  while ((match = pattern.exec(text)) !== null) {
    // Text before the match
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index))
    }

    if (match[2]) {
      // **bold**
      parts.push(<strong key={match.index}>{match[2]}</strong>)
    } else if (match[3]) {
      // *italic*
      parts.push(<em key={match.index}>{match[3]}</em>)
    } else if (match[4]) {
      // `code`
      parts.push(<code key={match.index} className="formatted-code">{match[4]}</code>)
    }

    lastIndex = match.index + match[0].length
  }

  // Remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex))
  }

  return parts.length > 0 ? parts : text
}
