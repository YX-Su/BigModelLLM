import type { ChatResponse } from './types'

const BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'

export async function sendChat(sessionId: string, query: string): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, query }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || '请求失败')
  }
  return res.json()
}

export async function clearSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/session/${sessionId}`, { method: 'DELETE' })
}
