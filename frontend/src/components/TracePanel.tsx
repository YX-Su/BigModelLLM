import type { ChatResponse } from '../types'

const REL_VERB: Record<string, string> = {
  BELONGS_TO: '属于',
  DEPENDS_ON: '依赖',
  MUTEX: '互斥于',
  ENHANCES: '增强',
  WEAKENS: '减弱',
}

export default function TracePanel({ trace }: { trace: ChatResponse | null }) {
  if (!trace) {
    return (
      <aside className="trace">
        <h2>GraphRAG 检索链路</h2>
        <p className="trace-empty">发送一个问题后,这里会展示实体识别、知识图谱子图与召回的参考片段。</p>
      </aside>
    )
  }

  return (
    <aside className="trace">
      <h2>GraphRAG 检索链路</h2>

      <section className="trace-block">
        <h3>① 命中实体(实体链接)</h3>
        {trace.seed_entities.length ? (
          <div className="chips">
            {trace.seed_entities.map((e) => (
              <span key={e} className="chip chip-entity">{e}</span>
            ))}
          </div>
        ) : (
          <p className="trace-empty">未命中实体</p>
        )}
      </section>

      <section className="trace-block">
        <h3>② 知识图谱子图({trace.relations.length} 条关系)</h3>
        {trace.relations.length ? (
          <ul className="triples">
            {trace.relations.map((r, i) => (
              <li key={i}>
                {r.rule_id && <span className="rule-id">{r.rule_id}</span>}
                <span className="triple-node">{r.source}</span>
                <span className="triple-verb">{REL_VERB[r.type] ?? r.type}</span>
                <span className="triple-node">{r.target}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="trace-empty">子图为空</p>
        )}
      </section>

      <section className="trace-block">
        <h3>③ 混合检索召回({trace.references.length} 段)</h3>
        {trace.references.length ? (
          <ul className="refs">
            {trace.references.map((c) => (
              <li key={c.chunk_id}>
                <div className="ref-head">
                  <span className="ref-title">{c.title || c.chunk_id}</span>
                  <span className="ref-score">融合 {c.score.toFixed(3)}</span>
                </div>
                <div className="ref-sub">
                  向量 {c.vec_score.toFixed(3)} · BM25 {c.bm25_score.toFixed(3)}
                </div>
                <p className="ref-text">{c.text}</p>
              </li>
            ))}
          </ul>
        ) : (
          <p className="trace-empty">无召回片段</p>
        )}
      </section>
    </aside>
  )
}
