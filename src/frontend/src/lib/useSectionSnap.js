import { useEffect } from 'react'

const NAV_OFFSET = 64
const SNAP_COOLDOWN_MS = 760
const WHEEL_THRESHOLD = 24
const TOUCH_THRESHOLD = 48

function interactiveTarget(target) {
  return target?.closest?.('input, textarea, select, button, a')
}

function getSections() {
  return Array.from(document.querySelectorAll('main > section'))
}

function currentSectionIndex(sections) {
  const position = window.scrollY + NAV_OFFSET + 4
  return sections.reduce((current, section, index) => (section.offsetTop <= position ? index : current), 0)
}

function scrollToSection(section) {
  window.scrollTo({
    top: Math.max(0, section.offsetTop - NAV_OFFSET),
    behavior: 'smooth',
  })
}

export function useSectionSnap(enabled = true) {
  useEffect(() => {
    if (!enabled) return undefined

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reducedMotion) return undefined

    let lastSnap = 0
    let touchStartY = null

    function snap(direction) {
      const now = window.performance.now()
      if (now - lastSnap < SNAP_COOLDOWN_MS) return

      const sections = getSections()
      if (!sections.length) return

      const currentIndex = currentSectionIndex(sections)
      const targetIndex = Math.max(0, Math.min(sections.length - 1, currentIndex + direction))
      const target = sections[targetIndex]
      if (!target || targetIndex === currentIndex) return

      lastSnap = now
      scrollToSection(target)
    }

    function onWheel(event) {
      if (interactiveTarget(event.target) || Math.abs(event.deltaY) < WHEEL_THRESHOLD) return
      event.preventDefault()
      snap(event.deltaY > 0 ? 1 : -1)
    }

    function onKeyDown(event) {
      if (interactiveTarget(event.target)) return

      if (event.key === 'ArrowDown' || event.key === 'PageDown' || event.key === ' ') {
        event.preventDefault()
        snap(1)
      }

      if (event.key === 'ArrowUp' || event.key === 'PageUp') {
        event.preventDefault()
        snap(-1)
      }
    }

    function onTouchStart(event) {
      touchStartY = event.touches[0]?.clientY ?? null
    }

    function onTouchEnd(event) {
      if (touchStartY === null || interactiveTarget(event.target)) return
      const touchEndY = event.changedTouches[0]?.clientY ?? touchStartY
      const delta = touchStartY - touchEndY
      touchStartY = null
      if (Math.abs(delta) < TOUCH_THRESHOLD) return
      event.preventDefault()
      snap(delta > 0 ? 1 : -1)
    }

    window.addEventListener('wheel', onWheel, { passive: false })
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('touchstart', onTouchStart, { passive: true })
    window.addEventListener('touchend', onTouchEnd, { passive: false })

    return () => {
      window.removeEventListener('wheel', onWheel)
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('touchstart', onTouchStart)
      window.removeEventListener('touchend', onTouchEnd)
    }
  }, [enabled])
}
