import { useCallback, useState } from 'react'

export interface Identity {
  name: string
  email: string
}

const STORAGE_KEY = 'fies_beta_user'

function readStoredIdentity(): Identity | null {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw)
    if (typeof parsed?.name === 'string' && typeof parsed?.email === 'string') {
      return parsed
    }
  } catch {
    // corrupted value — treat as unset
  }
  return null
}

// No real auth — just a one-time label so query_log/triage can be traced
// back to a beta tester. See src/financial_pipeline/storage/query_log.py's
// user_name/user_email columns.
export function useIdentity() {
  const [identity, setIdentityState] = useState<Identity | null>(readStoredIdentity)

  const setIdentity = useCallback((next: Identity) => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    setIdentityState(next)
  }, [])

  const clearIdentity = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY)
    setIdentityState(null)
  }, [])

  return { identity, setIdentity, clearIdentity }
}
