import { renderToString } from 'katex'

export default function Formula({ children }: { children: string }) {
  return <span className="[&_.katex]:text-[1.05em]" dangerouslySetInnerHTML={{ __html: renderToString(children, { throwOnError: true, strict: 'error' }) }} />
}
