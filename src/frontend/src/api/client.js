const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'
const DEFAULT_TIMEOUT_MS = 30000

async function request(path, options = {}) {
  const { headers: customHeaders, timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...fetchOptions } = options
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs)
  const headers = {
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...(customHeaders || {}),
  }

  let response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers,
      ...fetchOptions,
      signal: signal || controller.signal,
    })
  } catch (error) {
    if (error.name === 'AbortError') {
      const timeoutError = new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`)
      timeoutError.status = 0
      timeoutError.code = 'REQUEST_TIMEOUT'
      timeoutError.isTimeout = true
      throw timeoutError
    }
    error.status = error.status ?? 0
    error.code = error.code || 'NETWORK_ERROR'
    throw error
  } finally {
    window.clearTimeout(timeout)
  }

  const text = await response.text()
  let body = null
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    body = { detail: text }
  }

  if (!response.ok) {
    const detail = body?.detail
    const message = Array.isArray(detail)
      ? detail.map((item) => item?.msg).filter(Boolean).join(' ')
      : detail || `Request failed with HTTP ${response.status}`
    const error = new Error(message)
    error.status = response.status
    throw error
  }

  return body
}

export function getHealth(options = {}) {
  return request('/api/health', options)
}

export function createRun(payload) {
  return request('/api/run', {
    method: 'POST',
    body: JSON.stringify(payload),
    timeoutMs: 15000,
  })
}

export function getRun(jobId) {
  return request(`/api/run/${encodeURIComponent(jobId)}`, { timeoutMs: 10000 })
}

export function getScalability() {
  return request('/api/scalability')
}

export function sendBugReport(payload) {
  return request('/api/bug-report', {
    method: 'POST',
    body: JSON.stringify(payload),
    timeoutMs: 15000,
  })
}
