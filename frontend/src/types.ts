// Types mirroring the backend API response models.

export interface Relation {
  source: string
  target: string
  type: string
  rule_id: string
}

export interface ScoredChunk {
  chunk_id: string
  title: string
  text: string
  doc: string
  score: number
  vec_score: number
  bm25_score: number
}

export interface ValidationIssue {
  rule_id: string
  severity: string
  message: string
}

export interface ValidationReport {
  passed: boolean
  issues: ValidationIssue[]
  suggested_plan: string[]
}

export interface ChatResponse {
  session_id: string
  intent: string
  intent_label: string
  answer: string
  config_plan: string[]
  validation: ValidationReport | null
  seed_entities: string[]
  relations: Relation[]
  references: ScoredChunk[]
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
}
