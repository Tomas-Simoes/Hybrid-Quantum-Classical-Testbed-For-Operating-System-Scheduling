import { useEffect } from 'react'

const SECTION_SELECTOR = 'main > section'

export function useSectionReveal(refreshKey = true) {
  useEffect(() => {
    if (!refreshKey || typeof window === 'undefined') return undefined

    const sections = Array.from(document.querySelectorAll(SECTION_SELECTOR))
    if (!sections.length) return undefined

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const supportsObserver = 'IntersectionObserver' in window

    sections.forEach((section) => {
      section.classList.add('reveal-section')
    })

    if (prefersReducedMotion || !supportsObserver) {
      sections.forEach((section) => section.classList.add('section-visible'))

      return () => {
        sections.forEach((section) => {
          section.classList.remove('reveal-section', 'section-visible')
        })
      }
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return

          entry.target.classList.add('section-visible')
          observer.unobserve(entry.target)
        })
      },
      {
        rootMargin: '0px 0px -18% 0px',
        threshold: 0.14,
      },
    )

    sections.forEach((section) => observer.observe(section))

    return () => {
      observer.disconnect()
      sections.forEach((section) => {
        section.classList.remove('reveal-section', 'section-visible')
      })
    }
  }, [refreshKey])
}
