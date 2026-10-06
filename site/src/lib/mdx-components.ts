import Callout from '../components/mdx/Callout.astro'
import ChapterList from '../components/mdx/ChapterList.astro'
import CodeFile from '../components/mdx/CodeFile.astro'
import Definition from '../components/mdx/Definition.astro'
import Figure from '../components/mdx/Figure.astro'
import KeyEq from '../components/mdx/KeyEq.astro'
import Panel from '../components/mdx/Panel.astro'
import Panels from '../components/mdx/Panels.astro'
import StatGrid from '../components/mdx/StatGrid.astro'
import Steps from '../components/mdx/Steps.astro'
import Arrow from '../components/diagram/Arrow.astro'
import Bars from '../components/diagram/Bars.astro'
import Box from '../components/diagram/Box.astro'
import Diagram from '../components/diagram/Diagram.astro'
import Lanes from '../components/diagram/Lanes.astro'
import Legend from '../components/diagram/Legend.astro'
import Matrix from '../components/diagram/Matrix.astro'
import Plot from '../components/diagram/Plot.astro'
import Region from '../components/diagram/Region.astro'
import Text from '../components/diagram/Text.astro'

/** Components every chapter can use without importing them. Interactive labs are imported per chapter. */
export const mdxComponents = {
  Callout, ChapterList, CodeFile, Definition, Figure, KeyEq, Panel, Panels, StatGrid, Steps,
  Diagram, Box, Arrow, Text, Matrix, Region, Legend, Plot, Lanes, Bars,
}
