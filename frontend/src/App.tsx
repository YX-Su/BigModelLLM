import { useEffect, useRef, useState } from 'react'
import { clearSession, sendChat } from './api'
import TracePanel from './components/TracePanel'
import type { ChatMessage, ChatResponse } from './types'

const INTENT_CLASS: Record<string, string> = {
  explain: 'badge-explain',
  generate: 'badge-generate',
  optimize: 'badge-optimize',
  invalid: 'badge-invalid',
}

const SAMPLES = [
  '股票型基金配置是什么意思?',
  '为一位保守型、两年内要用钱的客户配一套投资组合方案',
  '现在的方案波动太大,帮我优化一下降低风险',
]

function newSessionId(): string {
  return 's-' + Math.random().toString(36).slice(2, 10)
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTrace, setActiveTrace] = useState<ChatResponse | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  async function submit(query: string) {
    const text = query.trim()
    if (!text || loading) return
    setInput('')
    setError('')
    setMessages((m) => [...m, { role: 'user', content: text }])
    setLoading(true)
    try {
      const res = await sendChat(sessionId, text)
      setMessages((m) => [...m, { role: 'assistant', content: res.answer, response: res }])
      setActiveTrace(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : '请求失败')
    } finally {
      setLoading(false)
    }
  }

  function startNewSession() {
    setSessionId(newSessionId())
    setMessages([])
    setActiveTrace(null)
    setError('')
  }

  async function resetSession() {
    await clearSession(sessionId).catch(() => undefined)
    setMessages([])
    setActiveTrace(null)
    setError('')
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>GraphRAG 投资组合配置助手</h1>
          <p className="subtitle">基于知识图谱与大模型推理的金融投资组合智能配置</p>
        </div>
        <div className="session-bar">
          <span className="session-id">会话 {sessionId}</span>
          <button onClick={resetSession}>清空上下文</button>
          <button onClick={startNewSession}>新建会话</button>
        </div>
      </header>

      <div className="main">
        <section className="chat">
          <div className="messages" ref={listRef}>
            {messages.length === 0 && (
              <div className="welcome">
                <p>你好,我是投资组合配置助手。你可以问我:</p>
                <div className="samples">
                  {SAMPLES.map((s) => (
                    <button key={s} className="sample" onClick={() => submit(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`bubble bubble-${msg.role}`}>
                {msg.role === 'assistant' && msg.response && (
                  <div className="bubble-meta">
                    <span className={`badge ${INTENT_CLASS[msg.response.intent] ?? ''}`}>
                      {msg.response.intent_label}
                    </span>
                    {msg.response.config_plan.length > 0 && (
                      <span className="plan-count">
                        配置项 {msg.response.config_plan.length} 项
                      </span>
                    )}
                  </div>
                )}
                <div className="bubble-text">{msg.content}</div>
                {msg.role === 'assistant' && msg.response && msg.response.config_plan.length > 0 && (
                  <div className="chips">
                    {msg.response.config_plan.map((code) => (
                      <span key={code} className="chip chip-plan">{code}</span>
                    ))}
                  </div>
                )}
                {msg.role === 'assistant' && msg.response?.validation && (
                  <div
                    className={`validation ${
                      msg.response.validation.passed ? 'validation-ok' : 'validation-fail'
                    }`}
                  >
                    {msg.response.validation.passed
                      ? '通过依赖与互斥规则校验'
                      : `规则校验发现 ${msg.response.validation.issues.length} 个问题`}
                  </div>
                )}
                {msg.role === 'assistant' && msg.response && (
                  <button className="trace-link" onClick={() => setActiveTrace(msg.response!)}>
                    查看检索链路
                  </button>
                )}
              </div>
            ))}

            {loading && <div className="bubble bubble-assistant loading">思考中…</div>}
          </div>

          {error && <div className="error">⚠ {error}</div>}

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault()
              submit(input)
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="输入配置相关问题,例如:债券基金和货币基金有什么区别?"
              disabled={loading}
            />
            <button type="submit" disabled={loading || !input.trim()}>
              发送
            </button>
          </form>
        </section>

        <TracePanel trace={activeTrace} />
      </div>
    </div>
  )
}
