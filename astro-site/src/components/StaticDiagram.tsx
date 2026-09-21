export default function StaticDiagram({ src, alt }: { src: string; alt: string }) {
  return <figure className="math-figure"><img src={src} alt={alt} width="640" height="440" loading="lazy" /></figure>
}
