import { useEffect, useState } from 'react'
import { getHealth } from '../api/client.js'

const navItems = [
  { label: 'Home', target: 'home', activeTarget: 'top', href: '#top' },
  { label: 'Tutorial', target: 'tutorial', href: '#tutorial' },
  { label: 'Contacts', target: 'contacts', href: '#contacts' },
  { label: 'Chamber', target: 'chamber', href: '#chamber' },
]

export function Nav({ activeTarget = 'top', onNavigate }) {
  const [health, setHealth] = useState('checking')

  function handleNavigate(event, target) {
    event.preventDefault()
    onNavigate?.(target)
  }

  useEffect(() => {
    let alive = true

    async function checkHealth() {
      try {
        await getHealth()
        if (alive) setHealth('live')
      } catch {
        if (alive) setHealth('offline')
      }
    }

    checkHealth()

    return () => {
      alive = false
    }
  }, [])

  return (
    <nav className="nav-glass" aria-label="Primary">
      <a className="wordmark" href="#top" aria-label="The Chamber home" onClick={(event) => handleNavigate(event, 'home')}>
        <span className="wordmark-mark">TC</span>
        <span>The Chamber</span>
      </a>

      <div className="nav-actions">
        {navItems.map((item) => (
          <a
            className={`nav-link ${activeTarget === (item.activeTarget || item.target) ? 'active' : ''}`}
            href={item.href}
            key={item.target}
            onClick={(event) => handleNavigate(event, item.target)}
          >
            {item.label}
          </a>
        ))}
      </div>

      <div className={`status-pill status-${health}`}>
        <span className="status-dot" aria-hidden="true" />
        <span className="mono">{health === 'live' ? 'LIVE' : health.toUpperCase()}</span>
      </div>
    </nav>
  )
}
