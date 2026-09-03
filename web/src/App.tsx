import { useState } from 'react'
import { useIdentity } from './hooks/useIdentity'
import { IdentityGate } from './components/IdentityGate'
import { QueryForm } from './components/QueryForm'
import { AnswerCard } from './components/AnswerCard'
import { FeedbackWidget } from './components/FeedbackWidget'
import { askQuestion, ApiError, type AskResponse } from './api'
import './App.css'

interface HistoryEntry {
  question: string
  response: AskResponse | null
  error: string | null
}

function App() {
  const { identity, setIdentity } = useIdentity()
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [busy, setBusy] = useState(false)

  if (!identity) {
    return <IdentityGate onSubmit={setIdentity} />
  }

  async function handleAsk(question: string) {
    setBusy(true)
    // Optimistic entry so the question appears immediately while we wait.
    setHistory((prev) => [{ question, response: null, error: null }, ...prev])

    try {
      const response = await askQuestion({
        question,
        user_name: identity!.name,
        user_email: identity!.email,
      })
      setHistory((prev) => {
        const [, ...rest] = prev
        return [{ question, response, error: null }, ...rest]
      })
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Request failed — try again.'
      setHistory((prev) => {
        const [, ...rest] = prev
        return [{ question, response: null, error: message }, ...rest]
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>FIES Beta</h1>
        <span className="app-header-identity">{identity.name}</span>
      </header>

      <QueryForm onSubmit={handleAsk} busy={busy} />

      <div className="history">
        {history.length === 0 && (
          <p className="history-empty">Ask a question about Indian mutual funds to get started.</p>
        )}
        {history.map((entry, i) => (
          <div className="history-entry" key={i}>
            <p className="history-question">{entry.question}</p>
            {entry.response && (
              <>
                <AnswerCard response={entry.response} />
                <FeedbackWidget queryId={entry.response.query_id} />
              </>
            )}
            {entry.error && <p className="history-error">{entry.error}</p>}
            {!entry.response && !entry.error && <p className="history-pending">Thinking…</p>}
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
