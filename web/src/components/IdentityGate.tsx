import { useState } from 'react'
import type { FormEvent } from 'react'
import type { Identity } from '../hooks/useIdentity'

interface Props {
  onSubmit: (identity: Identity) => void
}

export function IdentityGate({ onSubmit }: Props) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const trimmedName = name.trim()
    const trimmedEmail = email.trim()
    if (!trimmedName || !trimmedEmail) return
    onSubmit({ name: trimmedName, email: trimmedEmail })
  }

  return (
    <div className="identity-gate">
      <form className="identity-gate-card" onSubmit={handleSubmit}>
        <h1>FIES Beta</h1>
        <p>Tell us who you are before you start — this just tags your questions so we can follow up on anything flagged during the beta.</p>
        <label>
          Name
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            autoFocus
          />
        </label>
        <label>
          Email
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </label>
        <button type="submit">Start asking questions</button>
      </form>
    </div>
  )
}
