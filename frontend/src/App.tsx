import { useEffect, useState } from 'react'
import './App.css'

interface DatasetItem {
  question: string
  expected_answer: string
  actual_answer: string
}

interface EvaluationResult {
  question: string
  actual_answer: string
  token_f1_score: number
  semantic_similarity_score: number
  judge_score: number | null
  judge_reason: string
  verdict: string
}

interface RunResponse {
  experiment_id: number
  experiment_name: string
  model: string
  total_items: number
  passed: number
  failed: number
  average_token_f1: number
  average_semantic_similarity: number
  average_judge_score: number | null
  results: EvaluationResult[]
}

const emptyItem = (): DatasetItem => ({
  question: '',
  expected_answer: '',
  actual_answer: '',
})

function App() {
  const [page, setPage] = useState<'dashboard' | 'studio'>('dashboard')
  const [name, setName] = useState('')
  const [model, setModel] = useState('openai/gpt-oss-20b')
  const [generateAnswer, setGenerateAnswer] = useState(true)
  const [items, setItems] = useState<DatasetItem[]>([emptyItem()])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<RunResponse | null>(null)
  const [experiments, setExperiments] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  async function loadExperiments() {
    setLoading(true)
    try {
      const response = await fetch('/api/experiments')
      if (!response.ok) throw new Error('Could not load experiment history')
      setExperiments(await response.json())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'API connection failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadExperiments()
  }, [])

  function updateItem(
    index: number,
    field: keyof DatasetItem,
    value: string
  ) {
    setItems(current =>
      current.map((item, i) =>
        i === index ? { ...item, [field]: value } : item
      )
    )
  }

  function addItem() {
    if (items.length < 100) {
      setItems(current => [...current, emptyItem()])
    }
  }

  function removeItem(index: number) {
    if (items.length > 1) {
      setItems(current => current.filter((_, i) => i !== index))
    }
  }

  async function runEvaluation(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setRunning(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch('/api/experiments/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment_name: name,
          model,
          generate_answer: generateAnswer,
          items: items.map(item => ({
            question: item.question,
            expected_answer: item.expected_answer,
            ...(generateAnswer ? {} : {
              actual_answer: item.actual_answer
            })
          })),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Experiment execution failed')
      }

      setResult(data)
      await loadExperiments()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">E</div>
          <div>
            <h2>EvalForge</h2>
            <span>LLM Evaluation Platform</span>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>

        <nav className="navigation">
          <button
            className={`nav-item ${page === 'dashboard' ? 'active' : ''}`}
            onClick={() => setPage('dashboard')}
          >
            <span>▦</span> Overview
          </button>

          <button
            className={`nav-item ${page === 'studio' ? 'active' : ''}`}
            onClick={() => {
              setPage('studio')
              setResult(null)
              setError('')
            }}
          >
            <span>＋</span> Evaluation Studio
          </button>
        </nav>

        <div className="sidebar-bottom">
          <div className="status-indicator" />
          <div>
            <strong>System status</strong>
            <span>API connected</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              Workspace / {page === 'studio' ? 'Evaluation Studio' : 'Overview'}
            </div>
            <h1>{page === 'studio' ? 'Evaluation Studio' : 'Dashboard'}</h1>
          </div>
          <span className="environment">DEVELOPMENT</span>
        </header>

        {page === 'dashboard' ? (
          <>
            <section className="welcome-section">
              <div>
                <h2>Welcome to EvalForge</h2>
                <p>Monitor and improve the quality of your LLM applications.</p>
              </div>
              <button className="primary-button" onClick={() => setPage('studio')}>
                + New evaluation
              </button>
            </section>

            <section className="metrics-grid">
              <div className="metric-card">
                <div className="metric-top">Total experiments</div>
                <h2>{loading ? '—' : experiments.length}</h2>
                <p>Saved experiment runs</p>
              </div>

              <div className="metric-card">
                <div className="metric-top">Pass rate</div>
                <h2>
                  {loading ? '—' : (() => {
                    const total = experiments.reduce((s, e) => s + e.total_items, 0)
                    const passed = experiments.reduce((s, e) => s + e.passed, 0)
                    return total ? `${(passed / total * 100).toFixed(1)}%` : '0%'
                  })()}
                </h2>
                <p>Across saved evaluation items</p>
              </div>

              <div className="metric-card">
                <div className="metric-top">Evaluation items</div>
                <h2>
                  {loading ? '—' : experiments.reduce((s, e) => s + e.total_items, 0)}
                </h2>
                <p>Total items evaluated</p>
              </div>

              <div className="metric-card">
                <div className="metric-top">Average judge score</div>
                <h2>
                  {loading ? '—' : (() => {
                    const total = experiments.reduce((s, e) => s + e.total_items, 0)
                    return total
                      ? (experiments.reduce((s, e) => s + (e.average_judge_score ?? 0) * e.total_items, 0) / total).toFixed(2)
                      : '0.00'
                  })()}
                  <span className="out-of">/5</span>
                </h2>
                <p>Item-weighted average</p>
              </div>
            </section>

            <section className="content-panel">
              <div className="panel-header">
                <div>
                  <h2>Experiment history</h2>
                  <p>Your saved evaluation runs.</p>
                </div>
                <button className="secondary-button" onClick={loadExperiments}>
                  ↻ Refresh
                </button>
              </div>

              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>EXPERIMENT</th>
                      <th>MODEL</th>
                      <th>ITEMS</th>
                      <th>PASS / FAIL</th>
                      <th>TOKEN F1</th>
                      <th>JUDGE SCORE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experiments.map(e => (
                      <tr key={e.id}>
                        <td>
                          <strong>{e.experiment_name}</strong>
                          <span className="sub-text">#{e.id} · {e.created_at}</span>
                        </td>
                        <td>{e.model}</td>
                        <td>{e.total_items}</td>
                        <td>
                          <span className="passed-count">{e.passed} passed</span>
                          <span className="failed-count">{e.failed} failed</span>
                        </td>
                        <td>{Number(e.average_token_f1).toFixed(2)}</td>
                        <td>
                          {e.average_judge_score == null
                            ? '—'
                            : `${Number(e.average_judge_score).toFixed(1)}/5`}
                        </td>
                      </tr>
                    ))}
                    {!loading && experiments.length === 0 && (
                      <tr><td colSpan={6}>No experiments found.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        ) : (
          <>
            <section className="welcome-section">
              <div>
                <h2>Build a new evaluation</h2>
                <p>Define your test dataset and evaluate your model responses.</p>
              </div>
            </section>

            <form className="studio-form" onSubmit={runEvaluation}>
              <section className="content-panel studio-section">
                <div className="panel-header">
                  <div>
                    <h2>Experiment configuration</h2>
                    <p>Set up the experiment details.</p>
                  </div>
                </div>

                <div className="form-fields">
                  <label>
                    Experiment name
                    <input
                      value={name}
                      onChange={e => setName(e.target.value)}
                      placeholder="e.g. Python Knowledge Evaluation"
                      required
                    />
                  </label>

                  <label>
                    Model identifier
                    <input
                      value={model}
                      onChange={e => setModel(e.target.value)}
                      placeholder="e.g. openai/gpt-oss-20b"
                      required
                    />
                    <small>Enter a model identifier supported by your Groq account.</small>
                  </label>
                </div>
              </section>

              <section className="content-panel studio-section">
                <div className="panel-header">
                  <div>
                    <h2>Answer generation mode</h2>
                    <p>Choose how EvalForge obtains the model response.</p>
                  </div>
                </div>

                <div className="mode-selector">
                  <label>
                    <input
                      type="radio"
                      name="generationMode"
                      checked={generateAnswer}
                      onChange={() => setGenerateAnswer(true)}
                    />
                    <strong>Automatic generation</strong>
                    <small>Groq generates an answer for every question.</small>
                  </label>

                  <label>
                    <input
                      type="radio"
                      name="generationMode"
                      checked={!generateAnswer}
                      onChange={() => setGenerateAnswer(false)}
                    />
                    <strong>Manual evaluation</strong>
                    <small>Provide your own model responses.</small>
                  </label>
                </div>
              </section>

              <section className="content-panel studio-section">
                <div className="panel-header">
                  <div>
                    <h2>Evaluation dataset</h2>
                    <p>
                      Add questions and reference answers.
                      {generateAnswer
                        ? ' Groq will generate the actual responses.'
                        : ' Enter the actual model responses manually.'}
                      {' '}Maximum 100 items.
                    </p>
                  </div>
                  <span className="item-counter">{items.length} / 100</span>
                </div>

                <div className="dataset-list">
                  {items.map((item, index) => (
                    <div className="dataset-card" key={index}>
                      <div className="dataset-heading">
                        <strong>Evaluation item {index + 1}</strong>
                        {items.length > 1 && (
                          <button
                            type="button"
                            className="remove-button"
                            onClick={() => removeItem(index)}
                          >
                            Remove
                          </button>
                        )}
                      </div>

                      <label>
                        Question
                        <textarea
                          value={item.question}
                          onChange={e => updateItem(index, 'question', e.target.value)}
                          placeholder="Enter the evaluation question..."
                          required
                          rows={2}
                        />
                      </label>

                      <label>
                        Expected answer
                        <textarea
                          value={item.expected_answer}
                          onChange={e => updateItem(index, 'expected_answer', e.target.value)}
                          placeholder="Enter the reference answer..."
                          required
                          rows={3}
                        />
                      </label>

                      {!generateAnswer && (
                        <label>
                          Actual model response
                          <textarea
                            value={item.actual_answer}
                            onChange={e => updateItem(index, 'actual_answer', e.target.value)}
                            placeholder="Paste the response generated by your model..."
                            required
                            rows={3}
                          />
                        </label>
                      )}
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  className="secondary-button add-item-button"
                  onClick={addItem}
                  disabled={items.length >= 100}
                >
                  + Add another question
                </button>
              </section>

              {error && <div className="error-message">{error}</div>}

              <div className="run-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => setPage('dashboard')}
                >
                  Cancel
                </button>
                <button type="submit" className="primary-button" disabled={running}>
                  {running
                    ? (generateAnswer ? 'Generating and evaluating...' : 'Evaluating...')
                    : 'Run evaluation →'}
                </button>
              </div>
            </form>

            {result && (
              <section className="content-panel result-panel">
                <div className="panel-header">
                  <div>
                    <h2>Evaluation completed</h2>
                    <p>{result.experiment_name} · Experiment #{result.experiment_id}</p>
                  </div>
                  <span className="status-badge status-success">Saved</span>
                </div>

                <div className="detail-summary">
                  <div><span>Total items</span><strong>{result.total_items}</strong></div>
                  <div><span>Passed</span><strong className="passed-count">{result.passed}</strong></div>
                  <div><span>Failed</span><strong>{result.failed}</strong></div>
                  <div>
                    <span>Judge score</span>
                    <strong>
                      {result.average_judge_score == null
                        ? '—'
                        : `${result.average_judge_score.toFixed(2)}/5`}
                    </strong>
                  </div>
                </div>

                <div className="result-list">
                  {result.results.map((r, i) => (
                    <article className="result-card" key={i}>
                      <div className="result-heading">
                        <strong>Question {i + 1}</strong>
                        <span className={`status-badge ${r.verdict === 'PASS' ? 'status-success' : 'status-warning'}`}>
                          {r.verdict}
                        </span>
                      </div>

                      <p className="question-text">{r.question}</p>

                      <div className="generated-answer">
                        <span>Actual model response</span>
                        <p>{r.actual_answer || 'No answer returned.'}</p>
                      </div>

                      <div className="result-scores">
                        <span>Token F1: <strong>{r.token_f1_score.toFixed(2)}</strong></span>
                        <span>Semantic similarity: <strong>{r.semantic_similarity_score.toFixed(2)}</strong></span>
                        <span>Judge: <strong>{r.judge_score ?? '—'}/5</strong></span>
                      </div>

                      <div className="judge-reason">
                        <span>Judge reasoning</span>
                        <p>{r.judge_reason}</p>
                      </div>
                    </article>
                  ))}
                </div>

                <div className="run-actions">
                  <button
                    className="secondary-button"
                    onClick={() => {
                      setResult(null)
                      setName('')
                      setItems([emptyItem()])
                    }}
                  >
                    Create another experiment
                  </button>
                  <button className="primary-button" onClick={() => setPage('dashboard')}>
                    View dashboard
                  </button>
                </div>
              </section>
            )}
          </>
        )}

        <footer className="footer">
          <span>EvalForge © 2026</span>
          <span>Built for reliable AI evaluation</span>
        </footer>
      </main>
    </div>
  )
}

export default App
