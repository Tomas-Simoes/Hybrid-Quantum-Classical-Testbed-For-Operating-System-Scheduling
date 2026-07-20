import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { AlgorithmTutorial } from './components/AlgorithmTutorial.jsx'
import { BundleLattice } from './components/BundleLattice.jsx'
import { ContactSection } from './components/ContactSection.jsx'
import { Hero } from './components/Hero.jsx'
import { Nav } from './components/Nav.jsx'
import { ResultsView } from './components/ResultsView.jsx'
import { RunConsole } from './components/RunConsole.jsx'
import { ScalabilityChart } from './components/ScalabilityChart.jsx'
import { extractAssignments, extractConvergence, extractEnergy } from './lib/results.js'
import { useSectionSnap } from './lib/useSectionSnap.js'

const HOME_TARGETS = new Set(['top', 'tutorial', 'contacts'])
const NAV_OFFSET = 64

function initialPage() {
  if (typeof window === 'undefined') return 'home'
  return window.location.hash === '#chamber' ? 'chamber' : 'home'
}

function hashTarget() {
  if (typeof window === 'undefined') return 'top'
  const target = window.location.hash.replace('#', '')
  return HOME_TARGETS.has(target) ? target : 'top'
}

function scrollToTarget(target) {
  const element = document.getElementById(target)
  if (!element) return

  window.scrollTo({
    top: Math.max(0, element.offsetTop - NAV_OFFSET),
    behavior: 'smooth',
  })
}

function App() {
  const [page, setPage] = useState(initialPage)
  const [pendingTarget, setPendingTarget] = useState(hashTarget)
  const [activeTarget, setActiveTarget] = useState(() => (initialPage() === 'chamber' ? 'chamber' : hashTarget()))

  useSectionSnap(page === 'home')

  const [runState, setRunState] = useState({
    status: 'idle',
    jobId: null,
    job: null,
    effectiveConfig: { num_processes: 8, mixer_type: 'xy' },
  })

  const assignments = useMemo(() => extractAssignments(runState.job), [runState.job])
  const convergence = useMemo(() => extractConvergence(runState.job), [runState.job])
  const energy = useMemo(() => extractEnergy(runState.job), [runState.job])

  useEffect(() => {
    function syncFromLocation() {
      const target = window.location.hash.replace('#', '')
      if (target === 'chamber') {
        setPage('chamber')
        setActiveTarget('chamber')
        window.requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: 'smooth' }))
        return
      }

      const nextTarget = HOME_TARGETS.has(target) ? target : 'top'
      setPage('home')
      setActiveTarget(nextTarget)
      setPendingTarget(nextTarget)
    }

    window.addEventListener('hashchange', syncFromLocation)
    window.addEventListener('popstate', syncFromLocation)

    return () => {
      window.removeEventListener('hashchange', syncFromLocation)
      window.removeEventListener('popstate', syncFromLocation)
    }
  }, [])

  useEffect(() => {
    if (page !== 'home' || !pendingTarget) return undefined

    const frame = window.requestAnimationFrame(() => {
      scrollToTarget(pendingTarget)
      setActiveTarget(pendingTarget)
      setPendingTarget(null)
    })

    return () => window.cancelAnimationFrame(frame)
  }, [page, pendingTarget])

  useEffect(() => {
    if (page !== 'home') return undefined

    const sectionIds = ['top', 'tutorial', 'contacts']

    function syncActiveSection() {
      const current = sectionIds.reduce((active, id) => {
        const section = document.getElementById(id)
        if (!section) return active
        return section.offsetTop <= window.scrollY + NAV_OFFSET + 8 ? id : active
      }, 'top')

      setActiveTarget(current)
    }

    syncActiveSection()
    window.addEventListener('scroll', syncActiveSection, { passive: true })

    return () => window.removeEventListener('scroll', syncActiveSection)
  }, [page])

  function navigate(target) {
    if (target === 'chamber') {
      window.history.pushState(null, '', '#chamber')
      setPage('chamber')
      setActiveTarget('chamber')
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    const homeTarget = target === 'home' ? 'top' : target
    window.history.pushState(null, '', `#${homeTarget}`)
    setActiveTarget(homeTarget)

    if (page === 'home') {
      scrollToTarget(homeTarget)
      return
    }

    setPage('home')
    setPendingTarget(homeTarget)
  }

  const lattice = (
    <BundleLattice
      status={runState.status}
      assignments={assignments}
      convergence={convergence}
      processCount={runState.effectiveConfig?.num_processes || 8}
    />
  )

  return (
    <div className={`app-shell page-${page}`}>
      <Nav activeTarget={activeTarget} onNavigate={navigate} />
      <main className={page === 'chamber' ? 'chamber-main' : undefined}>
        {page === 'home' ? (
          <>
            <Hero lattice={lattice} />
            <AlgorithmTutorial />
            <ContactSection />
          </>
        ) : (
          <>
            <RunConsole
              runState={runState}
              onRunStateChange={setRunState}
              energy={energy}
              convergence={convergence}
            />
            <ResultsView job={runState.job} />
            <ScalabilityChart />
          </>
        )}
      </main>
      <footer className="site-footer">
        <span>Hybrid quantum-classical OS scheduling dissertation artifact.</span>
        <a href="https://github.com/Tomas-Simoes/Hybrid-Quantum-Classical-Testbed-For-Operating-System-Scheduling">
          Source
        </a>
      </footer>
    </div>
  )
}

export default App
