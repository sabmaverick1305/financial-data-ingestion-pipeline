import { useState } from 'react'
import { submitFeedback } from '../api'

interface Props {
  queryId: string
}

export function FeedbackWidget({ queryId }: Props) {
  const [rating, setRating] = useState<number | null>(null)
  const [comment, setComment] = useState('')
  const [status, setStatus] = useState<'idle' | 'submitting' | 'done' | 'error'>('idle')

  async function handleRate(value: number) {
    setRating(value)
  }

  async function handleSubmit() {
    if (rating === null) return
    setStatus('submitting')
    try {
      await submitFeedback({ query_id: queryId, rating, comment: comment.trim() || null })
      setStatus('done')
    } catch {
      setStatus('error')
    }
  }

  if (status === 'done') {
    return <div className="feedback-widget feedback-done">Thanks for the feedback.</div>
  }

  return (
    <div className="feedback-widget">
      <div className="feedback-stars">
        {[1, 2, 3, 4, 5].map((value) => (
          <button
            key={value}
            type="button"
            className={`star${rating !== null && value <= rating ? ' star-selected' : ''}`}
            onClick={() => handleRate(value)}
            aria-label={`Rate ${value} out of 5`}
          >
            ★
          </button>
        ))}
      </div>
      {rating !== null && (
        <>
          <textarea
            className="feedback-comment"
            placeholder="Optional: what was wrong or right about this answer?"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            maxLength={2000}
            rows={2}
          />
          <button type="button" onClick={handleSubmit} disabled={status === 'submitting'}>
            {status === 'submitting' ? 'Submitting…' : 'Submit feedback'}
          </button>
          {status === 'error' && <span className="feedback-error">Couldn't submit — try again.</span>}
        </>
      )}
    </div>
  )
}
