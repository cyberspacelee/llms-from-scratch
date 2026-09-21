import React, { useState } from 'react'

export default function DotProductExplorer() {
  const [angle, setAngle] = useState(60)
  const [aNorm, setANorm] = useState(2)
  const [bNorm, setBNorm] = useState(3)
  const score = aNorm * bNorm * Math.cos(angle * Math.PI / 180)
  // Keep the largest slider value inside the 320×230 viewBox.
  const size = 40
  const point = (r: number, deg: number) => [160 + size * r * Math.cos(deg * Math.PI / 180), 150 - size * r * Math.sin(deg * Math.PI / 180)]
  const a = point(aNorm, 0)
  const b = point(bNorm, angle)
  const projection = point(bNorm * Math.cos(angle * Math.PI / 180), 0)
  const fmt = (n: number) => n.toFixed(3)
  return <section className="astro-lab" aria-label="点积交互实验">
    <div className="astro-lab-head"><strong>长度、夹角与点积</strong><span>MDX + React island</span></div>
    <div className="astro-controls">
      <label>a 的长度 <output>{aNorm.toFixed(1)}</output><input type="range" min="0.5" max="3" step="0.1" value={aNorm} onChange={e => setANorm(Number(e.target.value))} /></label>
      <label>b 的长度 <output>{bNorm.toFixed(1)}</output><input type="range" min="0.5" max="3" step="0.1" value={bNorm} onChange={e => setBNorm(Number(e.target.value))} /></label>
      <label>夹角 θ <output>{angle}°</output><input type="range" min="-180" max="180" value={angle} onChange={e => setAngle(Number(e.target.value))} /></label>
    </div>
    <svg viewBox="0 0 320 230" role="img" aria-label={`夹角 ${angle} 度，点积 ${fmt(score)}`}>
      <line x1="20" y1="150" x2="300" y2="150" stroke="var(--line)" /><line x1="160" y1="25" x2="160" y2="205" stroke="var(--line)" />
      <line x1="160" y1="150" x2={a[0]} y2={a[1]} stroke="var(--green)" strokeWidth="4" />
      <line x1="160" y1="150" x2={b[0]} y2={b[1]} stroke="var(--red)" strokeWidth="4" strokeDasharray="7 5" />
      <line x1={b[0]} y1={b[1]} x2={projection[0]} y2={projection[1]} stroke="var(--blue)" strokeDasharray="4 4" />
      <text x={a[0] + 7} y={a[1] - 8} fill="var(--green)">a</text><text x={b[0] + 7} y={b[1] - 8} fill="var(--red)">b</text>
    </svg>
    <p className="astro-readout">aᵀb = {aNorm.toFixed(1)} × {bNorm.toFixed(1)} × cos({angle}°) = <strong>{fmt(score)}</strong></p>
  </section>
}
