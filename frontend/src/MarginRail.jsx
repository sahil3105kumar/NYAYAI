
export default function MarginRail({ errors, pageHeightPts, activeErrorIndex, onSelect }) {
  if (!pageHeightPts) return null

  return (
    <div className="margin-rail">
      {errors.map((error, i) => {
        const [, y0, , y1] = error.bbox
        const midY = (y0 + y1) / 2
        
        const topPercent = Math.min(100, Math.max(0, (midY / pageHeightPts) * 100))
        const isActive = i === activeErrorIndex

        return (
          <button
            key={i}
            type="button"
            className={`margin-tick${isActive ? ' margin-tick--active' : ''}`}
            style={{ top: `${topPercent}%`, '--tick-color': error.highlight_color }}
            title={`${error.error_type}: ${error.text}`}
            onClick={() => onSelect?.(i)}
          />
        )
      })}
    </div>
  )
}