/*
  Extended API client — adds helpers for the chat agent, PDF ingestion,
  and InLegalBERT analysis endpoints alongside the existing OCR flow.
*/

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// ─── Existing OCR Flow ───────────────────────────────────────────────

export async function uploadPdf(file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: formData })
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `upload failed (${res.status})`)
  }

  return { jobId: data.job_id }
}

export async function pollJobStatus(jobId) {
  const res = await fetch(`${API_BASE_URL}/status/${jobId}`)
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `status check failed (${res.status})`)
  }

  return { status: data.status }
}

export async function fetchResult(jobId) {
  const res = await fetch(`${API_BASE_URL}/result/${jobId}`)
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `fetching result failed (${res.status})`)
  }

  if (data.status === 'FAILURE') {
    throw new Error(data.error || 'processing failed')
  }

  // report is shaped exactly like renderer/report.py's build_report() output -
  // same shape mockData.js used to fake, so ErrorList/PdfCanvas/HighlightOverlay/
  // MarginRail don't need to know anything changed. the two download URLs are
  // folded in as extra fields, made absolute against the API origin.
  return {
    ...data.report,
    annotated_pdf_url: data.annotated_pdf_url ? `${API_BASE_URL}${data.annotated_pdf_url}` : null,
    report_html_url: data.report_html_url ? `${API_BASE_URL}${data.report_html_url}` : null,
  }
}


// ─── Chat Agent ──────────────────────────────────────────────────────

export async function sendChatMessage(message, threadId = 'default_session') {
  const res = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, thread_id: threadId }),
  })
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `chat failed (${res.status})`)
  }

  return { reply: data.reply, threadId: data.thread_id }
}


// ─── PDF → Neo4j Ingestion ──────────────────────────────────────────

export async function ingestPDFForGraph(pdfPath) {
  const res = await fetch(`${API_BASE_URL}/api/v1/chat/ingest`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pdf_path: pdfPath }),
  })
  const data = await res.json().catch(() => ({}))

  if (!res.ok) {
    throw new Error(data.detail || `ingestion failed (${res.status})`)
  }

  return data
}


// ─── InLegalBERT Analysis ────────────────────────────────────────────

export async function analyzeLSI(text) {
  const res = await fetch(`${API_BASE_URL}/analyze/lsi`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `LSI analysis failed (${res.status})`)
  return data
}

export async function analyzeRR(text) {
  const res = await fetch(`${API_BASE_URL}/analyze/rr`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `RR analysis failed (${res.status})`)
  return data
}

export async function analyzeCJPE(text) {
  const res = await fetch(`${API_BASE_URL}/analyze/cjpe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `CJPE analysis failed (${res.status})`)
  return data
}

export async function analyzeFull(text) {
  const res = await fetch(`${API_BASE_URL}/analyze/full`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `full analysis failed (${res.status})`)
  return data
}
