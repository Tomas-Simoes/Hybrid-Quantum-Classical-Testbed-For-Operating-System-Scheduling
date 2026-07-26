import { useState } from 'react'
import { sendBugReport } from '../api/client.js'

const contacts = [
  {
    label: 'GitHub',
    value: 'Tomas-Simoes',
    href: 'https://github.com/Tomas-Simoes',
    symbol: 'GH',
    detail: 'Repository, experiments, and implementation work.',
  },
  {
    label: 'LinkedIn',
    value: 'tomas-simoes',
    href: 'https://www.linkedin.com/in/tomas-simoes/',
    symbol: 'in',
    detail: 'Academic and professional profile.',
  },
  {
    label: 'Email',
    value: 'tomas20simoes@gmail.com',
    href: 'mailto:tomas20simoes@gmail.com',
    symbol: '@',
    detail: 'Direct contact for questions about the project.',
  },
]

function currentPageUrl() {
  if (typeof window === 'undefined') return ''
  return window.location.href
}

function createInitialReport() {
  return {
    message: '',
    steps: '',
    page_url: currentPageUrl(),
    name: '',
    email: '',
    website: '',
    form_started_at: Date.now(),
  }
}

export function ContactSection() {
  const [report, setReport] = useState(createInitialReport)
  const [submitState, setSubmitState] = useState('idle')
  const [feedbackMessage, setFeedbackMessage] = useState('')
  const isSubmitting = submitState === 'sending'

  function updateReport(field, value) {
    setReport((current) => ({
      ...current,
      [field]: value,
    }))
  }

  async function handleReportSubmit(event) {
    event.preventDefault()
    setSubmitState('sending')
    setFeedbackMessage('')

    try {
      await sendBugReport({
        ...report,
        subject: 'Website bug report',
        severity: 'medium',
        contact_consent: Boolean(report.email.trim()),
        form_started_at: report.form_started_at || Date.now(),
      })
      setReport(createInitialReport())
      setSubmitState('sent')
      setFeedbackMessage('Report received. Thank you.')
    } catch (error) {
      setSubmitState('error')
      setFeedbackMessage(error.message || 'The report could not be submitted.')
    }
  }

  return (
    <section className="section-shell contacts-section" id="contacts" aria-labelledby="contacts-title">
      <div className="contacts-layout">
        <div className="contact-signal" aria-hidden="true">
          <span className="contact-core">
            <span className="contact-core-label">TS</span>
          </span>
          <span className="contact-ring ring-a" />
          <span className="contact-ring ring-b" />
          <span className="contact-ring ring-c" />
          <span className="contact-ring ring-d" />
          <span className="contact-ring ring-e" />
          <span className="contact-particle particle-a" />
          <span className="contact-particle particle-b" />
          <span className="contact-particle particle-c" />
          <span className="contact-particle particle-d" />
          <span className="contact-particle particle-e" />
          <span className="contact-particle particle-f" />
        </div>

        <div className="contacts-content">
          <p className="eyebrow mono">CONTACTS</p>
          <h2 id="contacts-title">Reach the developer</h2>

          <div className="contact-links">
            {contacts.map((contact) => (
              <a className="contact-link" href={contact.href} key={contact.label} rel="noreferrer" target="_blank" title={`${contact.label}: ${contact.value}`}>
                <span className="contact-symbol mono">{contact.symbol}</span>
                <span>
                  <b>{contact.label}</b>
                  <strong className="mono" title={contact.value}>{contact.value}</strong>
                  <small>{contact.detail}</small>
                </span>
              </a>
            ))}
          </div>

          <form className="bug-report-form" onSubmit={handleReportSubmit}>
            <div className="bug-report-form-heading">
              <p className="config-heading mono">Bug report</p>
            </div>

            <label htmlFor="bug-message">
              <span>What happened?</span>
              <textarea
                id="bug-message"
                minLength="20"
                maxLength="4000"
                required
                rows="4"
                value={report.message}
                onChange={(event) => updateReport('message', event.target.value)}
              />
            </label>

            <label htmlFor="bug-email">
              <span>Email for reply</span>
              <input
                id="bug-email"
                type="email"
                maxLength="254"
                autoComplete="email"
                value={report.email}
                onChange={(event) => updateReport('email', event.target.value)}
              />
            </label>

            <label className="bug-report-honeypot" htmlFor="bug-website" aria-hidden="true">
              <span>Website</span>
              <input
                id="bug-website"
                type="text"
                tabIndex="-1"
                autoComplete="off"
                value={report.website}
                onChange={(event) => updateReport('website', event.target.value)}
              />
            </label>

            <div className="bug-report-action-row">
              <button className="run-button bug-report-submit" type="submit" disabled={isSubmitting} aria-busy={isSubmitting}>
                {isSubmitting ? 'Sending' : 'Send report'}
              </button>
            </div>

            {feedbackMessage && (
              <p className={`bug-report-feedback ${submitState === 'error' ? 'is-error' : 'is-success'} mono`} role="status" aria-live="polite">
                {feedbackMessage}
              </p>
            )}
          </form>
        </div>
      </div>
    </section>
  )
}
