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

export function ContactSection() {
  return (
    <section className="section-shell contacts-section" id="contacts" aria-labelledby="contacts-title">
      <div className="contacts-layout">
        <div className="contact-signal" aria-hidden="true">
          <span className="contact-core">TS</span>
          <span className="contact-ring ring-a" />
          <span className="contact-ring ring-b" />
          <span className="contact-node node-github">GH</span>
          <span className="contact-node node-linkedin">in</span>
          <span className="contact-node node-email">@</span>
        </div>

        <div className="contacts-content">
          <p className="eyebrow mono">CONTACTS</p>
          <h2 id="contacts-title">Reach the builder</h2>
          <p>
            This project is a dissertation artifact for hybrid quantum-classical operating-system
            scheduling. These are the cleanest places to inspect the work or get in touch.
          </p>

          <div className="contact-links">
            {contacts.map((contact) => (
              <a className="contact-link" href={contact.href} key={contact.label} rel="noreferrer" target="_blank">
                <span className="contact-symbol mono">{contact.symbol}</span>
                <span>
                  <b>{contact.label}</b>
                  <strong className="mono">{contact.value}</strong>
                  <small>{contact.detail}</small>
                </span>
              </a>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
