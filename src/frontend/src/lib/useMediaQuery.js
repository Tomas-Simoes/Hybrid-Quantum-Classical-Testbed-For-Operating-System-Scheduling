import { useEffect, useState } from 'react'

export function useMediaQuery(query) {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(query).matches
  })

  useEffect(() => {
    if (typeof window === 'undefined') return undefined

    const mediaQuery = window.matchMedia(query)
    const syncMatches = () => setMatches(mediaQuery.matches)

    syncMatches()
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', syncMatches)
    } else {
      mediaQuery.addListener(syncMatches)
    }

    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', syncMatches)
      } else {
        mediaQuery.removeListener(syncMatches)
      }
    }
  }, [query])

  return matches
}
