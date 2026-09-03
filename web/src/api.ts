// Typed client for the FIES API — shapes mirror
// src/financial_pipeline/api/schemas.py exactly.

export interface AskRequest {
  question: string
  year?: number | null
  month?: number | null
  category?: 'monthly' | 'quarterly' | 'unknown' | null
  top_k?: number
  model?: string | null
  user_name?: string | null
  user_email?: string | null
}

export interface Source {
  citation: number
  file_name: string | null
  period_year: number | null
  period_month: number | null
  category: string | null
  preview: string
}

export interface GuardrailReport {
  pre_passed: boolean
  post_passed: boolean
  blocked: boolean
  block_reason: string | null
  is_investment_advice: boolean
  answer_safe: boolean
  hallucination_risk: string // "low" | "medium" | "high" | "unknown"
  faithfulness_score: number // -1 = not computed
  citation_valid: boolean
  number_consistent: boolean
  abstention_detected: boolean
  warnings: string[]
}

export interface AskResponse {
  question: string
  answer: string
  sources: Source[]
  model: string
  latency_ms: number
  prompt_tokens: number
  completion_tokens: number
  retrieval_count: number
  guardrail: GuardrailReport | null
  thread_id: string
  query_id: string
}

export interface FeedbackRequest {
  query_id: string
  rating: number // 1-5
  comment?: string | null
}

export interface FeedbackResponse {
  query_id: string
  recorded: boolean
}

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function postJson<TResponse>(path: string, body: unknown): Promise<TResponse> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const data = await res.json()
      detail = data.detail ? JSON.stringify(data.detail) : detail
    } catch {
      // response body wasn't JSON — fall back to statusText
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<TResponse>
}

export function askQuestion(payload: AskRequest): Promise<AskResponse> {
  return postJson<AskResponse>('/api/ask', payload)
}

export function submitFeedback(payload: FeedbackRequest): Promise<FeedbackResponse> {
  return postJson<FeedbackResponse>('/api/feedback', payload)
}

export { ApiError }
