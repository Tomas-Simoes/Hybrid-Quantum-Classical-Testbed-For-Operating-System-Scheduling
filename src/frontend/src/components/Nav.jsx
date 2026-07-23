const navItems = [
  { label: 'Home', target: 'home', activeTarget: 'top', href: '#top' },
  { label: 'Chamber', target: 'chamber', href: '#chamber' },
  { label: 'Tutorial', target: 'tutorial', href: '#tutorial' },
  { label: 'Results', target: 'results', href: '#results' },
  { label: 'Contact', target: 'contacts', href: '#contacts' },
]

export function Nav({ activeTarget = 'top', onNavigate }) {
  function handleNavigate(event, target) {
    event.preventDefault()
    onNavigate?.(target)
  }

  return (
    <nav className="nav-glass" aria-label="Primary">
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
    </nav>
  )
}
