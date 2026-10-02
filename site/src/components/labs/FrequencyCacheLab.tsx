import SvgCanvas from './SvgCanvas'
import { useState } from 'react'
import { Toggle, Controls, LabFrame, Range, Readout, fmt, pen } from './Lab'
import { frequencyCache } from '../../lib/advanced-interactive-model'

export default function FrequencyCacheLab() {
  const [query, setQuery] = useState(5), [key, setKey] = useState(4), [scale, setScale] = useState(2), [recompute, setRecompute] = useState(false)
  const r = frequencyCache(query, Math.min(key, query), scale)
  const end = (angle: number) => [140 + Math.cos(angle) * 85, 115 - Math.sin(angle) * 85]
  const mixed = end(r.oldAngle), correct = end(r.newAngle)
  return <LabFrame title="频率改了，旧缓存仍是旧规则">
    <Controls><Range label="查询位置 i" value={query} min={1} max={8} onChange={setQuery} /><Range label="历史键 j" value={Math.min(key, query)} min={0} max={query} onChange={setKey} /><Range label="位置插值因子" value={scale} min={1} max={4} step={0.25} onChange={setScale} /><Toggle label="用新规则重算完整前缀" checked={recompute} onChange={value => setRecompute(value)} /></Controls>
    <SvgCanvas viewBox="0 0 280 230" className={pen.canvas} style={{maxWidth: 352}} role="img" aria-label={`混用分数 ${r.mixedScore}，新规则参照 ${r.correctScore}`}><circle cx="140" cy="115" r="85" className={pen.axis} /><path d={`M140 115L${correct.join(' ')}`} className={pen.a} /><path d={`M140 115L${(recompute ? correct : mixed).join(' ')}`} className={pen.b} /><text x="8" y="225">实线：新规则；虚线：实际缓存路径</text></SvgCanvas>
    <Readout>ω=π/4；旧键+新查询角差={fmt(r.oldAngle)}；完整新规则角差={fmt(r.newAngle)}<br />实际分数={fmt(recompute ? r.correctScore : r.mixedScore, 6)}；参照={fmt(r.correctScore, 6)}；误差={fmt(recompute ? 0 : r.mixedScore - r.correctScore, 6)}<br />本图用 q=k=(1,0) 隔离单块角度；多层模型须重算前缀隐藏状态，不能只重旋转旧键</Readout>
  </LabFrame>
}
