import Callout from '../components/mdx/Callout.astro'
import ChapterList from '../components/mdx/ChapterList.astro'
import CodeFile from '../components/mdx/CodeFile.astro'
import Figure from '../components/mdx/Figure.astro'
import KeyEq from '../components/mdx/KeyEq.astro'
import Panel from '../components/mdx/Panel.astro'
import Panels from '../components/mdx/Panels.astro'
import SourceNote from '../components/mdx/SourceNote.astro'
import StatGrid from '../components/mdx/StatGrid.astro'
import Steps from '../components/mdx/Steps.astro'

/** Components every lesson can use without importing them. Interactive labs are imported per lesson. */
export const mdxComponents = { Callout, ChapterList, CodeFile, Figure, KeyEq, Panel, Panels, SourceNote, StatGrid, Steps }
