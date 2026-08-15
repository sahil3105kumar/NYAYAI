
export default function HighlightOverlay({ errors, displayScale, activeErrorIndex, onSelect }) {
  if (!displayScale) return null

  return (
    <div className="highlight-overlay">
      {errors.map((error, i) => {
        const [x0, y0, x1, y1] = error.bbox
        const isActive = i === activeErrorIndex

      
        const left = Math.min(x0, x1) * displayScale
        const top = Math.min(y0, y1) * displayScale
        const width = Math.max(0, Math.abs(x1 - x0) * displayScale)
        const height = Math.max(0, Math.abs(y1 - y0) * displayScale)

        return (
          <button
            key={i}
            type="button"
            className={`highlight-box${isActive ? ' highlight-box--active' : ''}`}
            style={{
              left,
              top,
              width,
              height,
              '--highlight-color': error.highlight_color,
            }}
            title={error.suggestion || error.text}
            onClick={() => onSelect?.(i)}
          />
        )
      })}
    </div>
  )
}