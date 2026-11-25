import { useEffect, useState, type FormEvent, type ChangeEvent } from 'react'
import './App.css'

type Page = 'dashboard' | 'studio' | 'history' | 'analytics'

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

interface Experiment {
  id: number
  experiment_name: string
  model: string
  total_items: number
  passed: number
  failed: number
  average_token_f1: number
  average_semantic_similarity?: number
  average_judge_score: number | null
  created_at: string
}

const emptyItem = (): DatasetItem => ({
  question: '',
  expected_answer: '',
  actual_answer: '',
})

// Parses standard CSV files, including escaped quotes and quoted multiline fields.
function parseCSV(content: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let insideQuotes = false

  for (let i = 0; i < content.length; i++) {
    const char = content[i]

    if (char === '"') {
      if (insideQuotes && content[i + 1] === '"') {
        field += '"'
        i++
      } else {
        insideQuotes = !insideQuotes
      }
    } else if (char === ',' && !insideQuotes) {
      row.push(field)
      field = ''
    } else if ((char === '\n' || char === '\r') && !insideQuotes) {
      if (char === '\r' && content[i + 1] === '\n') i++

      row.push(field)

      if (row.some(value => value.trim() !== '')) {
        rows.push(row)
      }

      row = []
      field = ''
    } else {
      field += char
    }
  }

  if (insideQuotes) {
    throw new Error('CSV contains an unclosed quotation mark.')
  }

  row.push(field)

  if (row.some(value => value.trim() !== '')) {
    rows.push(row)
  }

  return rows
}

function App() {
  const [page, setPage] = useState<Page>('dashboard')
  const [name, setName] = useState('')
  const [model, setModel] = useState('openai/gpt-oss-20b')
  const [generateAnswer, setGenerateAnswer] = useState(true)
  const [items, setItems] = useState<DatasetItem[]>([emptyItem()])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [csvMessage, setCsvMessage] = useState('')
  const [result, setResult] = useState<RunResponse | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')

  async function loadExperiments() {
    setLoading(true)

    try {
      const response = await fetch('/api/experiments')

      if (!response.ok) {
        throw new Error('Could not load experiment history')
      }

      setExperiments(await response.json())
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'API connection failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadExperiments()
  }, [])

  function navigate(nextPage: Page) {
    setPage(nextPage)
    setError('')

    if (nextPage !== 'studio') {
      setResult(null)
    }
  }

  function updateItem(index: number, field: keyof DatasetItem, value: string) {
    setItems(current =>
      current.map((item, i) =>
        i === index ? { ...item, [field]: value } : item
      )
    )
  }

  function addItem() {
    if (items.length < 100) {
      setItems(current => [...current, emptyItem()])
      setCsvMessage('')
    }
  }

  function removeItem(index: number) {
    if (items.length > 1) {
      setItems(current => current.filter((_, i) => i !== index))
    }
  }

  function handleCSVUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    setError('')
    setCsvMessage('')

    const resetInput = () => {
      event.target.value = ''
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please upload a CSV file.')
      resetInput()
      return
    }

    const reader = new FileReader()

    reader.onload = () => {
      try {
        const content = String(reader.result || '')

        if (!content.trim()) {
          throw new Error('The uploaded CSV file is empty.')
        }

        const rows = parseCSV(content)

        if (rows.length < 2) {
          throw new Error('CSV must contain a header and at least one data row.')
        }

        const headers = rows[0].map(header =>
          header.trim().replace(/^\uFEFF/, '').toLowerCase()
        )

        const questionIndex = headers.indexOf('question')
        const answerIndex = headers.indexOf('expected_answer')
        const actualAnswerIndex = headers.indexOf('actual_answer')

        if (questionIndex === -1 || answerIndex === -1) {
          throw new Error(
            'CSV must contain question and expected_answer columns.'
          )
        }

        const dataRows = rows.slice(1)

        if (dataRows.length > 100) {
          throw new Error('Maximum 100 questions are allowed per experiment.')
        }

        const importedItems: DatasetItem[] = dataRows.map((columns, index) => {
          const question = (columns[questionIndex] || '').trim()
          const expectedAnswer = (columns[answerIndex] || '').trim()
          const actualAnswer =
            actualAnswerIndex === -1
              ? ''
              : (columns[actualAnswerIndex] || '').trim()

          if (!question || !expectedAnswer) {
            throw new Error(
              `Row ${index + 2} is missing a question or expected answer.`
            )
          }

          if (actualAnswerIndex !== -1 && !actualAnswer) {
            throw new Error(
              `Row ${index + 2} is missing an actual answer.`
            )
          }

          return {
            question,
            expected_answer: expectedAnswer,
            actual_answer: actualAnswer,
          }
        })

        if (importedItems.length === 0) {
          throw new Error('No valid dataset rows were found.')
        }

        setItems(importedItems)
        setResult(null)

        setCsvMessage(
          actualAnswerIndex !== -1
            ? `Imported ${importedItems.length} questions with actual answers from ${file.name}.`
            : `Imported ${importedItems.length} questions from ${file.name}.`
        )

        setError('')
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Could not process the CSV file.'
        )
      } finally {
        resetInput()
      }
    }

    reader.onerror = () => {
      setError('Failed to read the CSV file.')
      resetInput()
    }

    reader.readAsText(file)
  }

  async function runEvaluation(event: FormEvent<HTMLFormElement>) {
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
            ...(generateAnswer ? {} : { actual_answer: item.actual_answer }),
          })),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Experiment execution failed. Check your input and try again.'
        )
      }

      setResult(data)
      await loadExperiments()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setRunning(false)
    }
  }

  const totalItems = experiments.reduce((sum, e) => sum + e.total_items, 0)
  const totalPassed = experiments.reduce((sum, e) => sum + e.passed, 0)
  const totalFailed = experiments.reduce((sum, e) => sum + e.failed, 0)
  const passRate = totalItems ? (totalPassed / totalItems) * 100 : 0

  const judgeExperiments = experiments.filter(e => e.average_judge_score != null)
  const judgeWeight = judgeExperiments.reduce((sum, e) => sum + e.total_items, 0)

  const averageJudge = judgeWeight
    ? judgeExperiments.reduce(
        (sum, e) => sum + (e.average_judge_score ?? 0) * e.total_items,
        0
      ) / judgeWeight
    : 0

  const averageF1 = totalItems
    ? experiments.reduce(
        (sum, e) => sum + Number(e.average_token_f1) * e.total_items,
        0
      ) / totalItems
    : 0

  const filteredExperiments = experiments.filter(e =>
    `${e.experiment_name} ${e.model}`.toLowerCase().includes(search.toLowerCase())
  )

  const pageTitles: Record<Page, string> = {
    dashboard: 'Overview',
    studio: 'Evaluation Studio',
    history: 'Experiment History',
    analytics: 'Analytics',
  }

  function formatDate(value: string) {
    if (!value) return 'Date unavailable'

    const date = new Date(value)

    return Number.isNaN(date.getTime())
      ? value
      : date.toLocaleString([], {
          dateStyle: 'medium',
          timeStyle: 'short',
        })
  }

  function MetricCard({
    icon,
    label,
    value,
    note,
    accent,
  }: {
    icon: string
    label: string
    value: string
    note: string
    accent: string
  }) {
    return (
      <article className="metric-card">
        <div className="metric-card-top">
          <span>{label}</span>
          <span className={`metric-icon ${accent}`}>{icon}</span>
        </div>
        <div className="metric-value">{loading ? '—' : value}</div>
        <div className="metric-note">{note}</div>
      </article>
    )
  }

  function ExperimentTable() {
    return (
      <div className="table-scroll">
        <table className="experiment-table">
          <thead>
            <tr>
              <th>EXPERIMENT</th>
              <th>MODEL</th>
              <th>ITEMS</th>
              <th>PASS / FAIL</th>
              <th>TOKEN F1</th>
              <th>JUDGE</th>
            </tr>
          </thead>
          <tbody>
            {filteredExperiments.map(experiment => (
              <tr key={experiment.id}>
                <td>
                  <strong className="experiment-name">
                    {experiment.experiment_name}
                  </strong>
                  <span className="table-subtext">
                    #{experiment.id} · {formatDate(experiment.created_at)}
                  </span>
                </td>
                <td>
                  <span className="model-pill">{experiment.model}</span>
                </td>
                <td>{experiment.total_items}</td>
                <td>
                  <span className="pass-text">{experiment.passed} passed</span>
                  <span className="fail-text">{experiment.failed} failed</span>
                </td>
                <td>{Number(experiment.average_token_f1).toFixed(2)}</td>
                <td>
                  {experiment.average_judge_score == null
                    ? '—'
                    : `${Number(experiment.average_judge_score).toFixed(1)}/5`}
                </td>
              </tr>
            ))}

            {!loading && filteredExperiments.length === 0 && (
              <tr>
                <td colSpan={6} className="empty-table">
                  {search
                    ? 'No matching experiments found.'
                    : 'No experiments yet. Run your first evaluation.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">E</div>
          <div className="brand-copy">
            <h2>EvalForge</h2>
            <span>LLM Evaluation Platform</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>

        <nav className="navigation">
          <button
            className={`nav-item ${page === 'dashboard' ? 'active' : ''}`}
            onClick={() => navigate('dashboard')}
          >
            <span className="nav-icon">▦</span>
            <span>Overview</span>
          </button>

          <button
            className={`nav-item ${page === 'studio' ? 'active' : ''}`}
            onClick={() => navigate('studio')}
          >
            <span className="nav-icon">✳</span>
            <span>Evaluation Studio</span>
          </button>

          <button
            className={`nav-item ${page === 'history' ? 'active' : ''}`}
            onClick={() => navigate('history')}
          >
            <span className="nav-icon">◷</span>
            <span>Experiment History</span>
          </button>

          <button
            className={`nav-item ${page === 'analytics' ? 'active' : ''}`}
            onClick={() => navigate('analytics')}
          >
            <span className="nav-icon">▥</span>
            <span>Analytics</span>
          </button>
        </nav>

        <div className="sidebar-spacer" />

        <div className="sidebar-project">
          <div className="project-orb">✦</div>
          <div>
            <strong>EvalForge Engine</strong>
            <span>Evaluation workspace</span>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <div>
            <strong>System status</strong>
            <span>API connected</span>
          </div>
          <span className="status-live">LIVE</span>
        </div>

        <div className="sidebar-footer">EVALFORGE · 2026</div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">
              Workspace <span>/</span> {pageTitles[page]}
            </div>
            <h1>{pageTitles[page]}</h1>
          </div>

          <div className="topbar-actions">
            <span className="environment-tag">
              <span /> DEVELOPMENT
            </span>
            <button className="avatar-button" title="EvalForge workspace">
              E
            </button>
          </div>
        </header>

        {page === 'dashboard' && (
          <>
            <section className="welcome-section">
              <div>
                <div className="eyebrow">YOUR AI QUALITY WORKSPACE</div>
                <h2>
                  Welcome back to <span>EvalForge.</span>
                </h2>
                <p>
                  Measure, understand, and improve the quality of your language
                  models.
                </p>
              </div>

              <button
                className="primary-button"
                onClick={() => navigate('studio')}
              >
                <span>＋</span> New evaluation
              </button>
            </section>

            {error && <div className="error-message">{error}</div>}

            <section className="metrics-grid">
              <MetricCard
                icon="◇"
                label="Total experiments"
                value={String(experiments.length)}
                note="Saved experiment runs"
                accent="purple"
              />
              <MetricCard
                icon="↗"
                label="Pass rate"
                value={`${passRate.toFixed(1)}%`}
                note={`${totalPassed} passed out of ${totalItems} items`}
                accent="green"
              />
              <MetricCard
                icon="◎"
                label="Evaluation items"
                value={String(totalItems)}
                note={`${totalFailed} items did not pass`}
                accent="blue"
              />
              <MetricCard
                icon="✧"
                label="Average judge score"
                value={`${averageJudge.toFixed(2)}/5`}
                note="Item-weighted LLM judge score"
                accent="orange"
              />
            </section>

            <section className="dashboard-grid">
              <article className="panel performance-panel">
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">MODEL QUALITY</div>
                    <h3>Evaluation performance</h3>
                    <p>Token F1 score across your saved experiments</p>
                  </div>
                  <span className="panel-icon">⌁</span>
                </div>

                {experiments.length > 0 ? (
                  <div className="chart-area">
                    <div className="chart-y-labels">
                      <span>1.0</span>
                      <span>0.75</span>
                      <span>0.5</span>
                      <span>0.25</span>
                      <span>0</span>
                    </div>

                    <div className="bar-chart">
                      <div className="chart-grid-lines">
                        <i />
                        <i />
                        <i />
                        <i />
                        <i />
                      </div>

                      {experiments.slice(0, 12).reverse().map(experiment => {
                        const score = Math.max(
                          0,
                          Math.min(
                            1,
                            Number(experiment.average_token_f1) || 0
                          )
                        )

                        return (
                          <div className="bar-column" key={experiment.id}>
                            <div className="bar-track">
                              <div
                                className="bar-fill"
                                style={{
                                  height: `${Math.max(score * 100, 3)}%`,
                                }}
                                title={`F1: ${score.toFixed(2)}`}
                              />
                            </div>
                            <span>#{experiment.id}</span>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="chart-empty">
                    <div className="empty-chart-icon">⌁</div>
                    <strong>Your evaluation chart will appear here</strong>
                    <span>
                      Run an experiment to start tracking model quality.
                    </span>
                    <button
                      className="text-button"
                      onClick={() => navigate('studio')}
                    >
                      Create your first evaluation →
                    </button>
                  </div>
                )}

                <div className="chart-legend">
                  <span /> Token F1 score
                  <strong>Average: {averageF1.toFixed(2)}</strong>
                </div>
              </article>

              <article className="panel activity-panel">
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">WORKSPACE FEED</div>
                    <h3>Recent activity</h3>
                    <p>Your latest evaluation runs</p>
                  </div>
                  <span className="panel-icon">•••</span>
                </div>

                <div className="activity-list">
                  {experiments.slice(0, 5).map(experiment => (
                    <div className="activity-item" key={experiment.id}>
                      <div
                        className={`activity-symbol ${
                          experiment.failed === 0
                            ? 'activity-good'
                            : 'activity-warn'
                        }`}
                      >
                        {experiment.failed === 0 ? '✓' : '!'}
                      </div>
                      <div className="activity-copy">
                        <strong>{experiment.experiment_name}</strong>
                        <span>
                          {experiment.total_items} items · {experiment.model}
                        </span>
                      </div>
                      <span className="activity-result">
                        {experiment.passed}/{experiment.total_items}
                      </span>
                    </div>
                  ))}

                  {!loading && experiments.length === 0 && (
                    <div className="activity-empty">
                      Your recent runs will appear here.
                    </div>
                  )}
                </div>

                <button
                  className="activity-link"
                  onClick={() => navigate('history')}
                >
                  View all experiments <span>→</span>
                </button>
              </article>
            </section>

            <section className="panel history-panel">
              <div className="panel-heading">
                <div>
                  <div className="eyebrow">YOUR WORK</div>
                  <h3>Recent experiments</h3>
                  <p>Review your latest saved evaluation runs</p>
                </div>
                <button className="secondary-button" onClick={loadExperiments}>
                  ↻ Refresh
                </button>
              </div>
              <ExperimentTable />
            </section>
          </>
        )}

        {page === 'studio' && (
          <>
            <section className="welcome-section page-intro">
              <div>
                <div className="eyebrow">EVALUATION WORKSPACE</div>
                <h2>
                  Build a new <span>evaluation.</span>
                </h2>
                <p>
                  Configure your dataset and measure how well your model
                  performs.
                </p>
              </div>
            </section>

            <form className="studio-form" onSubmit={runEvaluation}>
              <section className="panel form-panel">
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">STEP 01</div>
                    <h3>Experiment configuration</h3>
                    <p>Name your run and select the model to evaluate.</p>
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
                    <small>
                      Enter a model identifier supported by your Groq account.
                    </small>
                  </label>
                </div>
              </section>

              <section className="panel form-panel">
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">STEP 02</div>
                    <h3>Answer generation mode</h3>
                    <p>Choose how EvalForge gets the model response.</p>
                  </div>
                </div>

                <div className="mode-selector">
                  <label
                    className={
                      generateAnswer ? 'mode-card selected' : 'mode-card'
                    }
                  >
                    <input
                      type="radio"
                      name="generationMode"
                      checked={generateAnswer}
                      onChange={() => setGenerateAnswer(true)}
                    />
                    <span className="mode-symbol">✦</span>
                    <strong>Automatic generation</strong>
                    <small>
                      Groq generates an answer for every question.
                    </small>
                  </label>

                  <label
                    className={
                      !generateAnswer ? 'mode-card selected' : 'mode-card'
                    }
                  >
                    <input
                      type="radio"
                      name="generationMode"
                      checked={!generateAnswer}
                      onChange={() => setGenerateAnswer(false)}
                    />
                    <span className="mode-symbol">⌘</span>
                    <strong>Manual evaluation</strong>
                    <small>Provide your own model responses.</small>
                  </label>
                </div>
              </section>

              <section className="panel form-panel">
                <div className="panel-heading dataset-panel-heading">
                  <div>
                    <div className="eyebrow">STEP 03</div>
                    <h3>Evaluation dataset</h3>
                    <p>
                      Add questions and reference answers.{' '}
                      {generateAnswer
                        ? 'Groq will generate the actual responses.'
                        : 'Enter actual model responses manually.'}{' '}
                      Maximum 100 items.
                    </p>
                  </div>
                  <span className="item-counter">{items.length} / 100</span>
                </div>

                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: '12px',
                    marginBottom: '20px',
                  }}
                >
                  <label
                    className="secondary-button"
                    style={{
                      position: 'relative',
                      overflow: 'hidden',
                      display: 'inline-flex',
                      alignItems: 'center',
                      cursor: 'pointer',
                    }}
                  >
                    ↑ Import CSV
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      onChange={handleCSVUpload}
                      aria-label="Import evaluation dataset from CSV"
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        opacity: 0,
                        cursor: 'pointer',
                      }}
                    />
                  </label>

                  <span style={{ fontSize: '12px', opacity: 0.7 }}>
                    Required columns: question, expected_answer
                  </span>
                </div>

                {csvMessage && (
                  <div
                    role="status"
                    style={{
                      padding: '12px 14px',
                      marginBottom: '16px',
                      borderRadius: '8px',
                      background: 'rgba(34, 197, 94, 0.10)',
                      color: '#4ade80',
                      fontSize: '13px',
                    }}
                  >
                    ✓ {csvMessage}
                  </div>
                )}

                <div className="dataset-list">
                  {items.map((item, index) => (
                    <div className="dataset-card" key={index}>
                      <div className="dataset-heading">
                        <strong>
                          <span>{String(index + 1).padStart(2, '0')}</span>{' '}
                          Evaluation item
                        </strong>

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
                          onChange={e =>
                            updateItem(index, 'question', e.target.value)
                          }
                          placeholder="Enter the evaluation question..."
                          required
                          rows={2}
                        />
                      </label>

                      <label>
                        Expected answer
                        <textarea
                          value={item.expected_answer}
                          onChange={e =>
                            updateItem(
                              index,
                              'expected_answer',
                              e.target.value
                            )
                          }
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
                            onChange={e =>
                              updateItem(
                                index,
                                'actual_answer',
                                e.target.value
                              )
                            }
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
                  className="add-item-button"
                  onClick={addItem}
                  disabled={items.length >= 100}
                >
                  ＋ Add another question
                </button>
              </section>

              {error && <div className="error-message">{error}</div>}

              <div className="run-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={() => navigate('dashboard')}
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  className="primary-button"
                  disabled={running}
                >
                  {running ? '◌ Running evaluation...' : 'Run evaluation →'}
                </button>
              </div>
            </form>

            {result && (
              <section className="panel result-panel">
                <div className="panel-heading">
                  <div>
                    <div className="eyebrow">RUN COMPLETED</div>
                    <h3>{result.experiment_name}</h3>
                    <p>
                      Experiment #{result.experiment_id} · {result.model}
                    </p>
                  </div>
                  <span className="status-badge status-success">SAVED</span>
                </div>

                <div className="detail-summary">
                  <div>
                    <span>Total items</span>
                    <strong>{result.total_items}</strong>
                  </div>
                  <div>
                    <span>Passed</span>
                    <strong className="pass-text">{result.passed}</strong>
                  </div>
                  <div>
                    <span>Failed</span>
                    <strong className="fail-text">{result.failed}</strong>
                  </div>
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
                  {result.results.map((r, index) => (
                    <article className="result-card" key={index}>
                      <div className="result-heading">
                        <strong>Question {index + 1}</strong>
                        <span
                          className={`status-badge ${
                            r.verdict === 'PASS'
                              ? 'status-success'
                              : 'status-warning'
                          }`}
                        >
                          {r.verdict}
                        </span>
                      </div>

                      <p className="question-text">{r.question}</p>

                      <div className="generated-answer">
                        <span>ACTUAL MODEL RESPONSE</span>
                        <p>{r.actual_answer || 'No answer returned.'}</p>
                      </div>

                      <div className="result-scores">
                        <span>
                          Token F1{' '}
                          <strong>{r.token_f1_score.toFixed(2)}</strong>
                        </span>
                        <span>
                          Semantic similarity{' '}
                          <strong>
                            {r.semantic_similarity_score.toFixed(2)}
                          </strong>
                        </span>
                        <span>
                          Judge <strong>{r.judge_score ?? '—'}/5</strong>
                        </span>
                      </div>

                      <div className="judge-reason">
                        <span>JUDGE REASONING</span>
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
                      setCsvMessage('')
                      setError('')
                    }}
                  >
                    Create another experiment
                  </button>

                  <button
                    className="primary-button"
                    onClick={() => navigate('dashboard')}
                  >
                    View dashboard
                  </button>
                </div>
              </section>
            )}
          </>
        )}

        {page === 'history' && (
          <>
            <section className="welcome-section page-intro">
              <div>
                <div className="eyebrow">EXPERIMENT ARCHIVE</div>
                <h2>
                  Your evaluation <span>history.</span>
                </h2>
                <p>Browse and review your saved model evaluation runs.</p>
              </div>

              <button
                className="primary-button"
                onClick={() => navigate('studio')}
              >
                ＋ New evaluation
              </button>
            </section>

            <section className="metrics-grid compact-metrics">
              <MetricCard
                icon="◇"
                label="Total experiments"
                value={String(experiments.length)}
                note="Saved runs"
                accent="purple"
              />
              <MetricCard
                icon="◎"
                label="Total items"
                value={String(totalItems)}
                note="Questions evaluated"
                accent="blue"
              />
              <MetricCard
                icon="✓"
                label="Passed items"
                value={String(totalPassed)}
                note="Successful evaluations"
                accent="green"
              />
              <MetricCard
                icon="!"
                label="Failed items"
                value={String(totalFailed)}
                note="Items to investigate"
                accent="orange"
              />
            </section>

            <section className="panel history-panel">
              <div className="panel-heading history-toolbar">
                <div>
                  <div className="eyebrow">ALL RUNS</div>
                  <h3>Experiment history</h3>
                  <p>{experiments.length} saved experiment runs</p>
                </div>

                <div className="history-actions">
                  <input
                    className="search-input"
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    placeholder="Search experiments..."
                  />
                  <button
                    className="secondary-button"
                    onClick={loadExperiments}
                  >
                    ↻ Refresh
                  </button>
                </div>
              </div>

              <ExperimentTable />
            </section>
          </>
        )}

        {page === 'analytics' && (
          <>
            <section className="welcome-section page-intro">
              <div>
                <div className="eyebrow">PERFORMANCE INSIGHTS</div>
                <h2>
                  Understand your model <span>quality.</span>
                </h2>
                <p>
                  Explore the evaluation metrics collected across your saved
                  experiments.
                </p>
              </div>

              <button
                className="secondary-button"
                onClick={loadExperiments}
              >
                ↻ Refresh data
              </button>
            </section>

            <section className="metrics-grid">
              <MetricCard
                icon="↗"
                label="Overall pass rate"
                value={`${passRate.toFixed(1)}%`}
                note={`${totalPassed} of ${totalItems} items passed`}
                accent="green"
              />
              <MetricCard
                icon="⌁"
                label="Average Token F1"
                value={averageF1.toFixed(2)}
                note="Item-weighted average"
                accent="purple"
              />
              <MetricCard
                icon="✧"
                label="Average judge score"
                value={`${averageJudge.toFixed(2)}/5`}
                note="Item-weighted average"
                accent="orange"
              />
              <MetricCard
                icon="◎"
                label="Total evaluated"
                value={String(totalItems)}
                note="Across all experiment runs"
                accent="blue"
              />
            </section>

            <section className="panel analytics-panel">
              <div className="panel-heading">
                <div>
                  <div className="eyebrow">QUALITY BREAKDOWN</div>
                  <h3>Experiment Token F1 scores</h3>
                  <p>
                    Compare the token overlap score from each saved run.
                  </p>
                </div>
              </div>

              {experiments.length ? (
                <div className="analytics-bars">
                  {experiments.slice(0, 12).map(experiment => (
                    <div
                      className="analytics-bar-row"
                      key={experiment.id}
                    >
                      <span>
                        #{experiment.id} · {experiment.experiment_name}
                      </span>
                      <div className="analytics-bar-track">
                        <div
                          style={{
                            width: `${
                              Math.max(
                                0,
                                Math.min(
                                  100,
                                  Number(experiment.average_token_f1) * 100
                                )
                              )
                            }%`,
                          }}
                        />
                      </div>
                      <strong>
                        {Number(experiment.average_token_f1).toFixed(2)}
                      </strong>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="chart-empty">
                  <strong>No analytics available yet</strong>
                  <span>
                    Run an evaluation to populate your analytics.
                  </span>
                </div>
              )}
            </section>
          </>
        )}

        <footer className="app-footer">
          <span>EvalForge © 2026</span>
          <span>
            Built for reliable AI evaluation <b>✦</b>
          </span>
        </footer>
      </main>
    </div>
  )
}

export default App
