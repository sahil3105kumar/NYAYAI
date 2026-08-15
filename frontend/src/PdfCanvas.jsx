import { useEffect, useRef, useState } from 'react'
import * as pdfjsLib from 'pdfjs-dist'
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.mjs?url'

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl


const MAX_RASTER_SCALE = 4


function safeDestroy(doc) {
  if (doc && typeof doc.destroy === 'function') {
    try {
      doc.destroy()
    } catch {
      // best-effort cleanup only - never let this crash the unmount
    }
  }
}


export default function PdfCanvas({ file, pageNumber, containerWidth, zoom, onPageRendered }) {
  const canvasRef = useRef(null)
  const [error, setError] = useState(null)
  const [doc, setDoc] = useState(null)

  useEffect(() => {
    if (!file) {
      setDoc(null)
      return
    }
    let cancelled = false
    let loadedDoc = null

    async function load() {
      try {
        const data = new Uint8Array(await file.arrayBuffer())
        loadedDoc = await pdfjsLib.getDocument({ data }).promise
        if (cancelled) {
          safeDestroy(loadedDoc)
          return
        }
        setDoc(loadedDoc)
      } catch (err) {
        if (!cancelled) setError(err.message)
      }
    }

    load()
    return () => {
      cancelled = true
      
      safeDestroy(loadedDoc)
    }
  }, [file])

  useEffect(() => {
    if (!doc) return
    let cancelled = false
    let renderTask = null

    async function render() {
      try {
        const page = await doc.getPage(pageNumber)
        if (cancelled) return

        const viewportAtScale1 = page.getViewport({ scale: 1 })

        // fitScale fills containerWidth exactly at zoom=1; falls back to
        // 1:1 point-to-CSS-pixel sizing if containerWidth isn't known yet
        // (first paint, before the ResizeObserver in App.jsx has measured
        // anything) rather than flashing a 0-width canvas.
        const fitScale = containerWidth
          ? containerWidth / viewportAtScale1.width
          : 1
        const displayScale = fitScale * (zoom || 1)

        const rasterScale = Math.min(displayScale, MAX_RASTER_SCALE)
        const viewport = page.getViewport({ scale: rasterScale })

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        canvas.width = viewport.width
        canvas.height = viewport.height
        canvas.style.width = `${viewportAtScale1.width * displayScale}px`
        canvas.style.height = `${viewportAtScale1.height * displayScale}px`

        renderTask = page.render({ canvasContext: ctx, viewport })
        await renderTask.promise
        if (cancelled) return

        onPageRendered?.({
          widthPts: viewportAtScale1.width,
          heightPts: viewportAtScale1.height,
          // CSS-displayed width divided by point width - what HighlightOverlay
          // and MarginRail multiply raw bbox coordinates by to land on the
          // right on-screen pixel regardless of fit-to-width sizing or zoom.
          displayScale,
          numPages: doc.numPages,
        })
      } catch (err) {
        if (!cancelled) setError(err.message)
      }
    }

    render()
    return () => {
      cancelled = true
      renderTask?.cancel()
    }
  }, [doc, pageNumber, containerWidth, zoom, onPageRendered])

  if (error) {
    return <div className="pdf-canvas-error">Couldn't render this page: {error}</div>
  }

  return <canvas ref={canvasRef} className="pdf-canvas" />
}