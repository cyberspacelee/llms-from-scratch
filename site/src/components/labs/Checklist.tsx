import { useEffect, useState } from 'react'

type Item = { title: string; detail: string; level: string }

/** A self-check list whose ticks are kept in this browser only. */
export default function Checklist({ items, storageKey }: { items: Item[]; storageKey: string }) {
  const [done, setDone] = useState<boolean[]>(() => items.map(() => false))

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) ?? '[]')
      if (Array.isArray(saved)) setDone(items.map((_, index) => Boolean(saved[index])))
    } catch {}
  }, [storageKey, items])

  function toggle(index: number) {
    const next = done.map((value, position) => (position === index ? !value : value))
    setDone(next)
    try { localStorage.setItem(storageKey, JSON.stringify(next)) } catch {}
  }

  return (
    <ul className="not-prose my-6 list-none border-t border-rule p-0">
      {items.map((item, index) => (
        <li key={item.title}>
          <label className="grid cursor-pointer grid-cols-[1.5rem_minmax(0,1fr)_auto] items-start gap-3 border-b border-rule py-3">
            <input type="checkbox" checked={done[index]} onChange={() => toggle(index)} className="mt-1.5 size-4 accent-accent" />
            <span>
              <strong className={`block font-semibold ${done[index] ? 'text-muted line-through' : ''}`}>{item.title}</strong>
              <span className="text-sm text-muted">{item.detail}</span>
            </span>
            <small className="rounded-sm bg-sunken px-2 py-0.5 text-xs whitespace-nowrap text-muted">{item.level}</small>
          </label>
        </li>
      ))}
    </ul>
  )
}
