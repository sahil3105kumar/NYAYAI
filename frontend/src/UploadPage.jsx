import { useRef, useState } from 'react'


async function isPdfFile(file) {
  if (!file) return false

  const hasPdfMime = file.type === 'application/pdf'
  const hasPdfExtension = file.name?.toLowerCase().endsWith('.pdf') ?? false
  if (!hasPdfMime && !hasPdfExtension) return false

  try {
    const header = await file.slice(0, 5).text()
    return header === '%PDF-'
  } catch {
    return false
  }
}

export default function UploadPage({ onFileSelected, status, error, onGoHome }) {
  const inputRef = useRef(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const busy = status === 'PENDING' || status === 'STARTED'

  async function handleFiles(fileList) {
    const file = fileList?.[0]
    if (file && (await isPdfFile(file))) {
      onFileSelected(file)
    }
  }

  return (
    <div className="upload-page">
      {onGoHome && (
        <button className="upload-home-btn" onClick={onGoHome} id="upload-home">
          Back to NyayAI Home
        </button>
      )}

      <div className="upload-page-intro">
        <span className="upload-eyebrow">NyayAI</span>
        <h1>Check a document before it's filed.</h1>
        <p>
          Upload an FIR, contract, or court notice. NyayAI checks spelling, grammar, citation
          accuracy, and name consistency, entirely on this machine.
        </p>
      </div>

      <div
        className={`upload-dropzone${isDragOver ? ' upload-dropzone--active' : ''}${busy ? ' upload-dropzone--busy' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragOver(true)
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragOver(false)
          if (!busy) handleFiles(e.dataTransfer.files)
        }}
        onClick={() => !busy && inputRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
        {busy ? (
          <>
            <div className="upload-spinner" aria-hidden="true" />
            <p>{status === 'PENDING' ? 'Queued...' : 'Checking the document...'}</p>
          </>
        ) : (
          <>
            <p className="upload-dropzone-title">Drop a PDF here, or click to choose one</p>
            <p className="upload-dropzone-hint">PDF only, processed locally</p>
          </>
        )}
      </div>

      {error && (
        <p className="upload-error" role="alert">
          {error}
        </p>
      )}
    </div>
  )
}