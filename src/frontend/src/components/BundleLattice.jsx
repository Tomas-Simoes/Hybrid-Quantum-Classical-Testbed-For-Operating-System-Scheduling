import { useEffect, useRef } from 'react'

const GOLD = '#C9A24B'
const GOLD_LIGHT = '#F2D985'
const GOLD_DEEP = '#6D5323'

function hexPath(ctx, x, y, radius, rotation = Math.PI / 6) {
  ctx.beginPath()
  for (let index = 0; index < 6; index += 1) {
    const angle = rotation + (Math.PI * 2 * index) / 6
    const px = x + Math.cos(angle) * radius
    const py = y + Math.sin(angle) * radius
    if (index === 0) ctx.moveTo(px, py)
    else ctx.lineTo(px, py)
  }
  ctx.closePath()
}

function drawHex(ctx, x, y, radius, options = {}) {
  const {
    alpha = 1,
    fill = GOLD,
    stroke = 'rgba(237, 230, 217, 0.1)',
    lineWidth = 1,
    rotation = Math.PI / 6,
    shadow = 0,
  } = options

  ctx.save()
  ctx.globalAlpha *= alpha
  ctx.shadowColor = fill
  ctx.shadowBlur = shadow

  const gradient = ctx.createLinearGradient(x - radius, y - radius, x + radius, y + radius)
  gradient.addColorStop(0, GOLD_LIGHT)
  gradient.addColorStop(0.55, fill)
  gradient.addColorStop(1, GOLD_DEEP)

  hexPath(ctx, x, y, radius, rotation)
  ctx.fillStyle = gradient
  ctx.fill()

  if (stroke) {
    ctx.shadowBlur = 0
    ctx.strokeStyle = stroke
    ctx.lineWidth = lineWidth
    ctx.stroke()
  }

  ctx.restore()
}

function rotatePoint(x, y, angle) {
  return {
    x: x * Math.cos(angle) - y * Math.sin(angle),
    y: x * Math.sin(angle) + y * Math.cos(angle),
  }
}

function drawSoftDisc(ctx, x, y, radius, color, alpha, blur) {
  ctx.save()
  ctx.globalAlpha = alpha
  ctx.filter = `blur(${blur}px)`

  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius)
  gradient.addColorStop(0, color)
  gradient.addColorStop(0.45, 'rgba(201, 162, 75, 0.36)')
  gradient.addColorStop(1, 'rgba(201, 162, 75, 0)')

  ctx.fillStyle = gradient
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function drawAmbientArcs(ctx, cx, cy, scale, time, alpha) {
  ctx.save()
  ctx.globalCompositeOperation = 'screen'
  ctx.lineWidth = 0.8 * scale

  const orbitCy = cy
  const orbits = [
    { rx: 178, ry: 42, tilt: 0.02, speed: -0.000068, phase: 4.8 },
    { rx: 164, ry: 38, tilt: 0.72, speed: 0.0001, phase: 0.8 },
    { rx: 164, ry: 38, tilt: -0.72, speed: -0.000092, phase: 2.6 },
  ]

  orbits.forEach((orbit, index) => {
    const pulse = 0.5 + Math.sin(time * 0.00032 + index * 1.1) * 0.5
    const phase = time * orbit.speed + orbit.phase
    const glintX = Math.cos(phase) * orbit.rx * scale
    const glintY = Math.sin(phase) * orbit.ry * scale

    ctx.save()
    ctx.translate(cx, orbitCy)
    ctx.rotate(orbit.tilt)
    ctx.strokeStyle = `rgba(242, 217, 133, ${alpha * (0.022 + pulse * 0.026)})`
    ctx.beginPath()
    ctx.ellipse(0, 0, orbit.rx * scale, orbit.ry * scale, 0, 0, Math.PI * 2)
    ctx.stroke()

    ctx.fillStyle = `rgba(242, 217, 133, ${alpha * (0.09 + pulse * 0.06)})`
    ctx.beginPath()
    ctx.arc(glintX, glintY, (1.2 + pulse * 0.9) * scale, 0, Math.PI * 2)
    ctx.fill()
    ctx.restore()
  })

  ctx.restore()
}

function drawDust(ctx, cx, cy, scale, time) {
  ctx.save()
  ctx.globalCompositeOperation = 'screen'

  for (let index = 0; index < 18; index += 1) {
    const angle = index * 2.399 + time * 0.000045
    const radius = (92 + (index % 6) * 28) * scale
    const drift = Math.sin(time * 0.00034 + index) * 7 * scale
    const x = cx + Math.cos(angle) * (radius + drift)
    const y = cy + Math.sin(angle) * (radius * 0.46 + drift * 0.35)
    const pulse = 0.5 + Math.sin(time * 0.0007 + index * 0.83) * 0.5

    ctx.fillStyle = `rgba(242, 217, 133, ${0.035 + pulse * 0.08})`
    ctx.beginPath()
    ctx.arc(x, y, (0.75 + pulse * 1.25) * scale, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.restore()
}

function honeycombOffsets(radius, gap = 0) {
  const x = Math.sqrt(3) * radius + gap
  const y = 1.5 * radius + gap
  return [
    { x: 0, y: 0 },
    { x, y: 0 },
    { x: x / 2, y },
    { x: -x / 2, y },
    { x: -x, y: 0 },
    { x: -x / 2, y: -y },
    { x: x / 2, y: -y },
  ]
}

function drawHoneycomb(ctx, cx, cy, radius, time, options = {}) {
  const {
    alpha = 1,
    blur = 0,
    gap = 0,
    rotation = 0,
    shadow = 0,
    scale = 1,
    muted = false,
    includeCenter = true,
  } = options
  const offsets = honeycombOffsets(radius * scale, gap * scale).filter((_, index) => includeCenter || index > 0)

  ctx.save()
  if (blur) ctx.filter = `blur(${blur}px)`

  offsets.forEach((offset, index) => {
    const point = rotatePoint(offset.x, offset.y, rotation)
    const angle = Math.atan2(offset.y, offset.x)
    const sweep = 0.5 + Math.sin(time * 0.00082 - angle * 1.7 + rotation * 9) * 0.5
    const stagger = 0.5 + Math.sin(time * 0.00048 + index * 1.47) * 0.5
    const wave = sweep * 0.76 + stagger * 0.24
    const cellAlpha = muted
      ? alpha * (0.16 + wave * 0.42)
      : alpha * (index === 0 ? 0.32 + wave * 0.22 : 0.18 + wave * 0.82)
    const fillShift = muted ? 'rgba(201, 162, 75, 0.74)' : index === 0 ? '#806424' : GOLD

    drawHex(ctx, cx + point.x, cy + point.y, radius * scale, {
      alpha: cellAlpha,
      fill: fillShift,
      stroke: muted ? 'rgba(237, 230, 217, 0.035)' : 'rgba(237, 230, 217, 0.08)',
      lineWidth: 1,
      rotation: Math.PI / 6 + rotation,
      shadow: muted ? shadow : shadow * (0.28 + wave * 0.92),
    })
  })

  ctx.restore()
}

function drawScene(ctx, width, height, time, active) {
  ctx.clearRect(0, 0, width, height)

  const scale = Math.min(width / 900, height / 430)
  const cx = width / 2
  const cy = height * 0.43
  const breath = 0.5 + Math.sin(time * 0.00058) * 0.5
  const runLift = active ? 1.12 : 1
  const clusterRotation = time * 0.000032 * runLift
  const underRotation = -time * 0.000021 * runLift
  const pulseScale = 1 + breath * 0.055

  drawSoftDisc(ctx, cx, cy, 238 * scale * pulseScale, 'rgba(201, 162, 75, 0.34)', 0.22 + breath * 0.1, 38 * scale)
  drawSoftDisc(ctx, cx, cy + 42 * scale, 172 * scale, 'rgba(201, 162, 75, 0.22)', 0.09 + breath * 0.04, 44 * scale)
  drawAmbientArcs(ctx, cx, cy, scale, time, 1)
  drawDust(ctx, cx, cy, scale, time)

  ctx.save()
  ctx.globalCompositeOperation = 'screen'
  drawHoneycomb(ctx, cx, cy, 50 * scale, time, {
    alpha: 0.24,
    blur: 12 * scale,
    gap: 0,
    rotation: underRotation,
    shadow: 0,
    scale: 1.08,
    muted: true,
  })
  ctx.restore()

  drawHoneycomb(ctx, cx, cy, 50 * scale, time, {
    alpha: 0.76,
    gap: 0,
    rotation: clusterRotation,
    shadow: 20 * scale,
  })

}

export function BundleLattice({ status }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined

    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return undefined

    let frame = 0
    let logicalWidth = 0
    let logicalHeight = 0
    let lastTime = 0
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    function resize() {
      const rect = canvas.getBoundingClientRect()
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      logicalWidth = Math.max(1, rect.width)
      logicalHeight = Math.max(1, rect.height)
      canvas.width = Math.floor(logicalWidth * dpr)
      canvas.height = Math.floor(logicalHeight * dpr)
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    function render(time = lastTime) {
      lastTime = time
      drawScene(ctx, logicalWidth, logicalHeight, time, status === 'running')
      if (!prefersReducedMotion) {
        frame = window.requestAnimationFrame(render)
      }
    }

    const observer = new ResizeObserver(() => {
      resize()
      drawScene(ctx, logicalWidth, logicalHeight, lastTime, status === 'running')
    })

    observer.observe(canvas)
    resize()
    render()

    return () => {
      observer.disconnect()
      window.cancelAnimationFrame(frame)
    }
  }, [status])

  return (
    <div className={`lattice-shell lattice-${status}`}>
      <canvas
        ref={canvasRef}
        className="quantum-orb-canvas"
        aria-label="Breathing gold hexagonal quantum scheduler mark"
        role="img"
      />
    </div>
  )
}
