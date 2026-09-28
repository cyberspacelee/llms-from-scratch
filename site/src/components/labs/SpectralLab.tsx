import { useId, useState } from 'react'
import { conditioning, rotate, spectralTransform, type Vec2 } from '../../lib/chapter-labs-model'
import { Arrow, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'

export default function SpectralLab() {
  const [angle, setAngle] = useState(37)
  const [direction, setDirection] = useState(65)
  const [large, setLarge] = useState(3)
  const [exponent, setExponent] = useState(4)
  const id = useId()
  const x = rotate([1, 0], direction)
  const result = spectralTransform(x, angle, [1, large])
  const sensitivity = conditioning(exponent, 1e-6)
  const point = ([a, b]: Vec2) => [Math.round((180 + a * 45) * 100) / 100, Math.round((180 - b * 45) * 100) / 100]
  const vector = (v: Vec2, color: string, name: string) => {
    const [a, b] = point(v)
    return <g><line x1="180" y1="180" x2={a} y2={b} className={color} markerEnd={`url(#${id})`} /><text x={a + 8} y={b - 8}>{name}</text></g>
  }
  return <LabFrame title="旋转坐标轴，再沿特征方向缩放" hint="A = Q diag(1, λ₂) Qᵀ；‖x‖ = 1">
    <Controls>
      <Range label="特征轴角度 θ" min={0} max={90} value={angle} onChange={setAngle} format={v => `${v}°`} />
      <Range label="输入方向 φ" min={0} max={180} value={direction} onChange={setDirection} format={v => `${v}°`} />
      <Range label="第二特征值 λ₂" min={1} max={3} step={0.1} value={large} onChange={setLarge} />
    </Controls>
    <svg viewBox="0 0 360 360" className={`${pen.canvas} max-w-100!`} role="img" aria-label={`输入向量和变换结果，第二特征值 ${large}`}>
      <defs><Arrow id={id} className="fill-ink" /></defs>
      <line x1="20" y1="180" x2="340" y2="180" className={pen.grid} /><line x1="180" y1="20" x2="180" y2="340" className={pen.grid} />
      <circle cx="180" cy="180" r="45" className={pen.guide} />
      <ellipse cx="180" cy="180" rx="45" ry={large * 45} transform={`rotate(${-angle} 180 180)`} className={pen.grid} />
      {[0, 90].map((offset, i) => {
        const a = point(rotate([-3.2, 0], angle + offset)), b = point(rotate([3.2, 0], angle + offset))
        return <g key={offset}><line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className={pen.guide} /><text x={b[0]} y={b[1] - 8} textAnchor="middle">q{i + 1}</text></g>
      })}
      {vector(x, pen.a, 'x')}{vector(result.output, pen.b, 'Ax')}
    </svg>
    <Readout>Qᵀx = ({fmt(result.coordinates[0])}, {fmt(result.coordinates[1])}) → ΛQᵀx = ({fmt(result.scaled[0])}, {fmt(result.scaled[1])}) → Ax = ({fmt(result.output[0])}, {fmt(result.output[1])})</Readout>
    <div className="mt-6 border-t border-rule pt-4">
      <Range label="病态矩阵 A = diag(1, 10⁻ᵏ)，指数 k" min={0} max={6} value={exponent} onChange={setExponent} />
      <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
        <div><div className="text-muted">右端相对扰动</div><p className="mt-1 font-mono">δb = (0, 10⁻⁶)</p></div>
        <div><div className="text-muted">解的相对扰动</div><p className="mt-1 font-mono">δx₂ = {sensitivity.outputError.toExponential(1)}</p></div>
      </div>
      <svg viewBox="0 0 360 125" className={`${pen.canvas} max-w-100!`} role="img" aria-label={`相对误差对数刻度，输入一百万分之一，输出 ${sensitivity.outputError.toExponential(1)}`}>
        <text x="46" y="16" className={pen.muted}>相对误差（对数刻度）</text>
        <text x="4" y="49">δb</text><text x="4" y="79">δx</text>
        {[0, 1, 2, 3, 4, 5, 6].map(k => <line key={k} x1={46 + k * 44} x2={46 + k * 44} y1="30" y2="90" className={pen.grid} />)}
        <circle cx="46" cy="45" r="5" className="fill-accent" />
        <line x1="46" x2={46 + exponent * 44} y1="75" y2="75" className={pen.b} />
        <circle cx={46 + exponent * 44} cy="75" r="5" className="fill-accent2" />
        <text x="46" y="111" textAnchor="middle">10⁻⁶</text><text x="178" y="111" textAnchor="middle">10⁻³</text><text x="310" y="111" textAnchor="middle">1</text>
      </svg>
      <Readout>κ₂(A) = {sensitivity.condition.toExponential(1)} · 相同的 δb，输出放大 {sensitivity.condition.toLocaleString('en-US')} 倍 · x = (1, 0)</Readout>
    </div>
  </LabFrame>
}
