import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { AlgorithmTutorial } from './components/AlgorithmTutorial.jsx'
import { BundleLattice } from './components/BundleLattice.jsx'
import { ContactSection } from './components/ContactSection.jsx'
import { Hero } from './components/Hero.jsx'
import { Nav } from './components/Nav.jsx'
import { ResearchResultsSection } from './components/ResearchResultsSection.jsx'
import { ResultsView } from './components/ResultsView.jsx'
import { RunConsole } from './components/RunConsole.jsx'
import { extractAssignments, extractConvergence, extractEnergy } from './lib/results.js'
import { useSectionReveal } from './lib/useSectionReveal.js'

const HOME_TARGETS = new Set(['top', 'tutorial', 'results', 'contacts'])
const NAV_OFFSET = 64
const CHAMBER_TRANSITION_MS = 820

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

function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function App() {
  const [page, setPage] = useState(initialPage)
  const [pendingTarget, setPendingTarget] = useState(hashTarget)
  const [activeTarget, setActiveTarget] = useState(() => (initialPage() === 'chamber' ? 'chamber' : hashTarget()))
  const [pageTransition, setPageTransition] = useState(null)
  const lastScrolledJobRef = useRef(null)
  const pageRef = useRef(initialPage())
  const transitionTimeoutRef = useRef(null)

  useSectionReveal(page)

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
    pageRef.current = page
  }, [page])

  useEffect(() => {
    if (page !== 'chamber') return undefined

    let secondFrame = null
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: 'auto' }))
    })

    return () => {
      window.cancelAnimationFrame(firstFrame)
      if (secondFrame) window.cancelAnimationFrame(secondFrame)
    }
  }, [page])

  useEffect(() => {
    return () => {
      if (transitionTimeoutRef.current) {
        window.clearTimeout(transitionTimeoutRef.current)
      }
    }
  }, [])

  function clearPageTransition() {
    if (transitionTimeoutRef.current) {
      window.clearTimeout(transitionTimeoutRef.current)
      transitionTimeoutRef.current = null
    }

    setPageTransition(null)
  }

  function enterChamber({ updateHistory = false } = {}) {
    if (updateHistory && window.location.hash !== '#chamber') {
      window.history.pushState(null, '', '#chamber')
    }

    setActiveTarget('chamber')

    if (pageRef.current !== 'home' || prefersReducedMotion()) {
      clearPageTransition()
      setPage('chamber')
      window.scrollTo({ top: 0, behavior: 'smooth' })
      return
    }

    clearPageTransition()
    setPageTransition('enter-chamber')

    transitionTimeoutRef.current = window.setTimeout(() => {
      transitionTimeoutRef.current = null
      setPage('chamber')
      setPageTransition(null)
      window.requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: 'auto' }))
    }, CHAMBER_TRANSITION_MS)
  }

  useEffect(() => {
    function syncFromLocation() {
      const target = window.location.hash.replace('#', '')
      if (target === 'chamber') {
        enterChamber()
        return
      }

      const nextTarget = HOME_TARGETS.has(target) ? target : 'top'
      clearPageTransition()
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

    const sectionIds = ['top', 'tutorial', 'results', 'contacts']

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

  useEffect(() => {
    if (page !== 'chamber') return undefined
    if (runState.status !== 'done' || !runState.job?.result || !runState.jobId) return undefined
    if (lastScrolledJobRef.current === runState.jobId) return undefined

    lastScrolledJobRef.current = runState.jobId
    const frame = window.requestAnimationFrame(() => scrollToTarget('run-results'))

    return () => window.cancelAnimationFrame(frame)
  }, [page, runState.status, runState.job, runState.jobId])

  function navigate(target) {
    if (target === 'chamber') {
      enterChamber({ updateHistory: true })
      return
    }

    const homeTarget = target === 'home' ? 'top' : target
    clearPageTransition()
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
    <div className={`app-shell page-${page} ${pageTransition ? `transition-${pageTransition}` : ''}`}>
      <Nav activeTarget={activeTarget} onNavigate={navigate} />
      <main className={page === 'chamber' ? 'chamber-main' : undefined}>
        {page === 'home' ? (
          <>
            <Hero lattice={lattice} />
            <AlgorithmTutorial onNavigate={navigate} />
            <ResearchResultsSection />
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
          </>
        )}
      </main>
      <footer className="site-footer">
        <span>Hybrid quantum-classical OS scheduling dissertation artifact.</span>
        <a href="https://github.com/Tomas-Simoes/Hybrid-Quantum-Classical-Testbed-For-Operating-System-Scheduling">
          Source
        </a>
      </footer>
      {pageTransition === 'enter-chamber' ? (
        <div className="chamber-transition" aria-hidden="true">
          <span className="chamber-transition-grid" />
          <span className="chamber-transition-ring chamber-ring-outer" />
          <span className="chamber-transition-ring chamber-ring-inner" />
          <span className="chamber-transition-core" />
          <span className="chamber-transition-label mono">CHAMBER</span>
        </div>
      ) : null}
    </div>
  )
}

export default App
