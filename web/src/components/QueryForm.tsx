import { useState } from 'react'
import type { FormEvent } from 'react'

interface Props {
  onSubmit: (question: string) => void
  busy: boolean
}

// Mirrors AskRequest.question's server-side constraint (min_length=3,
// max_length=1000 in api/schemas.py) so obviously-invalid submissions
// never round-trip to the API.
const MIN_LENGTH = 3
const MAX_LENGTH = 1000

export function QueryForm({ onSubmit, busy }: Props) {
  const [question, setQuestion] = useState('')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmed = question.trim()
    if (trimmed.length < MIN_LENGTH) return
    onSubmit(trimmed)
    setQuestion('')
  }

  const tooShort = question.trim().length > 0 && question.trim().length < MIN_LENGTH

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        maxLength={MAX_LENGTH}
        placeholder="Ask about Indian mutual funds — e.g. 'What was the AUM of large cap funds in 2024?'"
        rows={3}
        disabled={busy}
      />
      <div className="query-form-footer">
        <span className={`char-count${tooShort ? ' char-count-warn' : ''}`}>
          {question.length}/{MAX_LENGTH}
        </span>
        <button type="submit" disabled={busy || question.trim().length < MIN_LENGTH}>
          {busy ? 'Asking…' : 'Ask'}
        </button>
      </div>
    </form>
  )
}
