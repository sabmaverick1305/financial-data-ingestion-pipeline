import type { AskResponse } from '../api'

interface Props {
  response: AskResponse
}

export function AnswerCard({ response }: Props) {
  const g = response.guardrail
  const showWarningBanner = g && (g.blocked || (g.hallucination_risk !== 'low' && g.hallucination_risk !== 'unknown'))

  return (
    <div className="answer-card">
      {showWarningBanner && (
        <div className={`guardrail-banner${g!.blocked ? ' guardrail-banner-blocked' : ' guardrail-banner-risk'}`}>
          {g!.blocked
            ? `Blocked: ${g!.block_reason ?? 'policy guardrail'}`
            : `Guardrail flagged this answer as ${g!.hallucination_risk} risk — double-check before relying on it.`}
        </div>
      )}

      <p className="answer-text">{response.answer}</p>

      {response.sources.length > 0 && (
        <details className="sources">
          <summary>{response.sources.length} source{response.sources.length === 1 ? '' : 's'}</summary>
          <ol>
            {response.sources.map((s) => (
              <li key={s.citation}>
                <strong>[{s.citation}]</strong>{' '}
                {s.file_name ?? 'source'}
                {s.period_year ? ` (${s.period_year}${s.period_month ? `-${s.period_month}` : ''})` : ''}
                <div className="source-preview">{s.preview}</div>
              </li>
            ))}
          </ol>
        </details>
      )}

      <div className="answer-meta">
        {response.model} · {response.latency_ms}ms · {response.prompt_tokens + response.completion_tokens} tokens
      </div>
    </div>
  )
}
