const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request(path, options = {}) {
  const headers = {
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...(options.headers || {}),
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers,
    ...options,
  })

  const text = await response.text()
  const body = text ? JSON.parse(text) : null

  if (!response.ok) {
    const message = body?.detail || `Request failed with HTTP ${response.status}`
    throw new Error(message)
  }

  return body
}

export function getHealth() {
  return request('/api/health')
}

export function createRun(payload) {
  return request('/api/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function getRun(jobId) {
  return request(`/api/run/${jobId}`)
}

export function getScalability() {
  return request('/api/scalability')
}
