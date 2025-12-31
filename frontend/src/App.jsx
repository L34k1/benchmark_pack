import { useEffect, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const initialState = {
  runId: null,
  status: null,
  progress: 0,
  findings: []
}

export default function App() {
  const [targetUrl, setTargetUrl] = useState('')
  const [maxDuration, setMaxDuration] = useState(300)
  const [state, setState] = useState(initialState)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let interval
    if (state.runId && state.status && ['queued', 'running'].includes(state.status)) {
      interval = setInterval(async () => {
        const response = await fetch(`${API_BASE}/runs/${state.runId}`)
        const data = await response.json()
        setState((prev) => ({ ...prev, status: data.status, progress: data.progress }))
        if (data.status === 'finished') {
          const findingsResponse = await fetch(`${API_BASE}/runs/${state.runId}/findings`)
          const findingsData = await findingsResponse.json()
          setState((prev) => ({ ...prev, findings: findingsData.findings }))
        }
      }, 4000)
    }
    return () => clearInterval(interval)
  }, [state.runId, state.status])

  const handleSubmit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`${API_BASE}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_url: targetUrl, max_duration_sec: maxDuration })
      })
      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.detail || 'Unable to start run')
      }
      const data = await response.json()
      setState({ ...initialState, runId: data.run_id, status: 'queued' })
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Web Security Scan (Passive)</h1>
        <p>Only scan targets you are authorized to test.</p>
      </header>
      <form onSubmit={handleSubmit}>
        <label>
          Target URL
          <input
            type="url"
            required
            placeholder="https://example.com"
            value={targetUrl}
            onChange={(event) => setTargetUrl(event.target.value)}
          />
        </label>
        <label>
          Max duration (sec)
          <input
            type="number"
            min="60"
            max="1800"
            value={maxDuration}
            onChange={(event) => setMaxDuration(Number(event.target.value))}
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? 'Starting...' : 'Lancer le scan'}
        </button>
      </form>
      {error && <div className="error">{error}</div>}
      {state.runId && (
        <section className="status">
          <h2>Run Status</h2>
          <p>ID: {state.runId}</p>
          <p>Status: {state.status}</p>
          <p>Progress: {state.progress}%</p>
          {state.status === 'finished' && (
            <a href={`${API_BASE}/runs/${state.runId}/report.html`} target="_blank" rel="noreferrer">
              Download HTML report
            </a>
          )}
        </section>
      )}
      <section>
        <h2>Findings</h2>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Summary</th>
              <th>Severity</th>
              <th>Check</th>
              <th>Category</th>
            </tr>
          </thead>
          <tbody>
            {state.findings.map((finding) => (
              <tr key={finding.id}>
                <td>{finding.id}</td>
                <td>{finding.summary}</td>
                <td>{finding.severity}</td>
                <td>{finding.check}</td>
                <td>{finding.category}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
