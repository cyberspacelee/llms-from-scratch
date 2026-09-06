/* The embedded Markdown is repository-authored, not an untrusted input channel. */
(() => {
  const source = document.getElementById('lesson-source').textContent;
  const article = document.getElementById('article');
  const status = document.getElementById('load-status');

  if (!window.markdownit || !window.texmath || !window.katex || !window.hljs) {
    status.className = 'warning';
    status.textContent = 'CDN 资源加载失败，以下保留 Markdown 原文。请检查网络后刷新。';
    const pre = document.createElement('pre');
    pre.textContent = source;
    article.append(pre);
    return;
  }

  try {
    const md = markdownit({
      html: true,
      highlight(code, language) {
        return hljs.getLanguage(language)
          ? hljs.highlight(code, {language}).value
          : '';
      },
    }).use(texmath, {
      engine: katex,
      delimiters: 'dollars',
      katexOptions: {throwOnError: true, trust: false, strict: 'error'},
    });
    article.innerHTML = md.render(source);
    article.querySelectorAll('pre code').forEach(code => code.classList.add('hljs'));
    article.querySelectorAll('h2').forEach((section, index) => {
      section.id = 'section-' + (index + 1);
      const link = document.createElement('a');
      link.href = '#' + section.id;
      link.textContent = section.textContent;
      document.getElementById('contents').append(link);
    });
    document.documentElement.dataset.rendered = 'true';
  } catch (error) {
    status.className = 'warning';
    status.textContent = '渲染失败：' + error.message;
    article.textContent = source;
    console.error(error);
    return;
  }

  const frequencies = Array.from({length: 4}, (_, m) => 10000 ** (-2 * m / 8));
  const format = (value, digits = 3) => {
    const rounded = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
    return rounded.toFixed(digits);
  };
  const heading = (title, kicker) =>
    '<div class="lab-heading"><p class="lab-title">' + title +
    '</p><span class="lab-kicker">' + kicker + '</span></div>';

  function initRotationLab() {
    const lab = document.getElementById('rotation-lab');
    if (!lab) return;

    lab.innerHTML = heading('同样的位移，不同的起点', 'rotation · matrix') +
      '<div class="lab-controls">' +
        '<label>起点 p <output id="rotation-p-value">2</output>' +
          '<input aria-label="起点 p" id="rotation-p" type="range" min="0" max="20" value="2">' +
        '</label>' +
        '<label>位移 Δ <output id="rotation-delta-value">3</output>' +
          '<input aria-label="位移 Δ" id="rotation-delta" type="range" min="-8" max="8" value="3">' +
        '</label>' +
      '</div>' +
      '<svg class="lab-svg" viewBox="0 0 840 330" role="img" aria-labelledby="rotation-svg-title rotation-svg-desc"></svg>' +
      '<div class="readout" aria-live="polite"></div>';

    const svg = lab.querySelector('svg');
    const readout = lab.querySelector('.readout');
    const pInput = lab.querySelector('#rotation-p');
    const deltaInput = lab.querySelector('#rotation-delta');
    let renderedWidth = 0;

    function draw() {
      const p = Number(pInput.value);
      const delta = Number(deltaInput.value);
      const theta = Math.PI / 8;
      const qAngle = p * theta;
      const kAngle = (p + delta) * theta;
      const phi = delta * theta;
      const cos = Math.cos(phi);
      const sin = Math.sin(phi);
      const width = Math.max(320, Math.round(svg.clientWidth));
      const compact = width < 620;
      const height = compact ? 570 : 330;
      const cx = compact ? width / 2 : 170;
      const cy = compact ? 165 : 165;
      const radius = compact ? Math.min(108, width / 2 - 42) : 112;
      renderedWidth = width;
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      const point = angle => ({
        x: cx + radius * Math.cos(angle),
        y: cy - radius * Math.sin(angle),
      });
      const q = point(qAngle);
      const k = point(kAngle);
      const arcRadius = 52;
      const arcStart = {
        x: cx + arcRadius * Math.cos(qAngle),
        y: cy - arcRadius * Math.sin(qAngle),
      };
      const arcEnd = {
        x: cx + arcRadius * Math.cos(kAngle),
        y: cy - arcRadius * Math.sin(kAngle),
      };
      const sweep = delta >= 0 ? 0 : 1;
      const arc = delta === 0 ? '' :
        '<path d="M ' + arcStart.x + ' ' + arcStart.y +
        ' A ' + arcRadius + ' ' + arcRadius + ' 0 0 ' + sweep + ' ' +
        arcEnd.x + ' ' + arcEnd.y + '" class="position-guide"/>';
      const values = [cos, -sin, sin, cos];
      const cellWidth = compact ? Math.min(120, (width - 76) / 2) : 98;
      const cellHeight = 50;
      const cellGap = 7;
      const matrixWidth = 2 * cellWidth + cellGap;
      const matrixX = compact ? (width - matrixWidth) / 2 : 530;
      const matrixY = compact ? 390 : 120;
      const matrixCenter = matrixX + matrixWidth / 2;
      const matrixLabelX = compact ? matrixX : 440;
      const matrixLabelY = compact ? 355 : 77;
      const matrixBottom = matrixY + 2 * cellHeight + 8;
      const qLabelX = Math.max(8, Math.min(width - 46, q.x + 10));
      const kLabelX = Math.max(8, Math.min(width - 74, k.x + 10));
      const matrixCells = values.map((value, index) => {
        const column = index % 2;
        const row = Math.floor(index / 2);
        const x = matrixX + column * (cellWidth + cellGap);
        const y = matrixY + row * (cellHeight + 8);
        return '<rect x="' + x + '" y="' + y + '" width="' + cellWidth +
          '" height="' + cellHeight + '" fill="var(--soft)"/>' +
          '<text x="' + (x + cellWidth / 2) + '" y="' + (y + 31) +
          '" class="mono" text-anchor="middle">' + format(value) + '</text>';
      }).join('');

      svg.innerHTML =
        '<title id="rotation-svg-title">两个位置向量的共同旋转与相对旋转矩阵</title>' +
        '<desc id="rotation-svg-desc">左侧为单位圆上的查询和键向量，右侧为只由位移决定的二乘二旋转矩阵。</desc>' +
        '<defs>' +
          '<marker id="arrow-q" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--green)"/>' +
          '</marker>' +
          '<marker id="arrow-k" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--red)"/>' +
          '</marker>' +
        '</defs>' +
        '<circle cx="' + cx + '" cy="' + cy + '" r="' + radius + '" class="axis"/>' +
        '<line x1="' + (cx - radius - 22) + '" y1="' + cy + '" x2="' +
        (cx + radius + 22) + '" y2="' + cy + '" class="axis"/>' +
        '<line x1="' + cx + '" y1="' + (cy - radius - 22) + '" x2="' + cx +
        '" y2="' + (cy + radius + 22) + '" class="axis"/>' +
        '<text x="' + (cx + radius + 18) + '" y="' + (cy - 9) +
        '" class="muted" text-anchor="end">cos</text>' +
        '<text x="' + (cx + 9) + '" y="' + (cy - radius - 13) + '" class="muted">sin</text>' +
        arc +
        '<line x1="' + cx + '" y1="' + cy + '" x2="' + q.x + '" y2="' + q.y + '" class="vector-q" marker-end="url(#arrow-q)"/>' +
        '<line x1="' + cx + '" y1="' + cy + '" x2="' + k.x + '" y2="' + k.y + '" class="vector-k" marker-end="url(#arrow-k)"/>' +
        '<circle cx="' + q.x + '" cy="' + q.y + '" r="5" fill="var(--green)"/>' +
        '<circle cx="' + k.x + '" cy="' + k.y + '" r="5" fill="var(--red)"/>' +
        '<text x="' + qLabelX + '" y="' + Math.max(18, q.y - 9) +
        '" style="fill:var(--green)">q(p)</text>' +
        '<text x="' + kLabelX + '" y="' + Math.min(cy + radius + 34, k.y + 18) +
        '" style="fill:var(--red)">k(p + Δ)</text>' +
        '<text x="' + matrixLabelX + '" y="' + matrixLabelY + '" class="mono">R(Δθ) =</text>' +
        '<path d="M ' + (matrixX - 8) + ' ' + (matrixY - 8) +
        ' h -10 v ' + (matrixBottom - matrixY + 16) + ' h 10 M ' +
        (matrixX + matrixWidth + 8) + ' ' + (matrixY - 8) +
        ' h 10 v ' + (matrixBottom - matrixY + 16) + ' h -10" class="matrix-bracket"/>' +
        matrixCells +
        '<text x="' + matrixCenter + '" y="' + (compact ? 525 : 270) +
        '" class="mono muted" text-anchor="middle">Δθ = ' + format(phi) + ' rad</text>' +
        '<text x="' + matrixCenter + '" y="' + (compact ? 552 : 298) +
        '" class="mono" text-anchor="middle">qᵀR(Δθ)k = ' + format(cos, 6) + '</text>';

      lab.querySelector('#rotation-p-value').value = p;
      lab.querySelector('#rotation-delta-value').value = delta;
      readout.textContent =
        'p = ' + p + ' · j = ' + (p + delta) + ' · Δ = ' + delta +
        ' · 夹角只由 Δ 决定 · cos(Δθ) = ' + format(cos, 6);
    }

    [pInput, deltaInput].forEach(input => input.addEventListener('input', draw));
    new ResizeObserver(() => {
      const width = Math.max(320, Math.round(svg.clientWidth));
      if (width !== renderedWidth) draw();
    }).observe(svg);
    draw();
  }

  function initFrequencyLab() {
    const lab = document.getElementById('frequency-lab');
    if (!lab) return;

    const options = frequencies.map((omega, m) =>
      '<option value="' + m + '">m = ' + m + ' · d' + (2 * m) +
      '/d' + (2 * m + 1) + ' · ω = ' + omega + '</option>'
    ).join('');
    lab.innerHTML = heading('一个频率对，两种观察方式', 'function · phase') +
      '<div class="lab-controls">' +
        '<label>维度对 m' +
          '<select id="frequency-pair" aria-label="选择位置编码维度对">' + options + '</select>' +
        '</label>' +
        '<label>观察位置 p <output id="frequency-position-value">12</output>' +
          '<input id="frequency-position" aria-label="观察位置 p" type="range" min="0" max="64" value="12">' +
        '</label>' +
      '</div>' +
      '<svg class="lab-svg" viewBox="0 0 840 360" role="img" aria-labelledby="frequency-svg-title frequency-svg-desc"></svg>' +
      '<div class="readout" aria-live="polite"></div>';

    const svg = lab.querySelector('svg');
    const pairInput = lab.querySelector('#frequency-pair');
    const positionInput = lab.querySelector('#frequency-position');
    const readout = lab.querySelector('.readout');
    let renderedWidth = 0;

    function draw() {
      const m = Number(pairInput.value);
      const p = Number(positionInput.value);
      const omega = frequencies[m];
      const sinValue = Math.sin(p * omega);
      const cosValue = Math.cos(p * omega);
      const width = Math.max(320, Math.round(svg.clientWidth));
      const compact = width < 620;
      const height = compact ? 610 : 360;
      const plot = compact
        ? {left: 43, right: width - 12, top: 57, bottom: 305}
        : {left: 62, right: 574, top: 35, bottom: 294};
      const xScale = value =>
        plot.left + (value / 64) * (plot.right - plot.left);
      const yScale = value =>
        plot.top + ((1.15 - value) / 2.3) * (plot.bottom - plot.top);
      const functionPath = fn => Array.from({length: 257}, (_, index) => {
        const position = index / 4;
        return (index ? 'L' : 'M') + ' ' + xScale(position).toFixed(2) +
          ' ' + yScale(fn(position)).toFixed(2);
      }).join(' ');
      const positionX = xScale(p);
      const circle = compact
        ? {x: width / 2, y: 466, radius: 82}
        : {x: width - 134, y: 162, radius: 84};
      const vectorX = circle.x + circle.radius * sinValue;
      const vectorY = circle.y - circle.radius * cosValue;
      const sinLegendX = compact ? plot.left + 31 : 375;
      const cosLegendX = compact ? plot.left + 158 : 475;
      const xTicks = [0, 16, 32, 48, 64].map(value =>
        '<line x1="' + xScale(value) + '" y1="' + plot.top +
        '" x2="' + xScale(value) + '" y2="' + plot.bottom + '" class="grid-line"/>' +
        '<text x="' + xScale(value) + '" y="' + (plot.bottom + 23) +
        '" class="mono muted" text-anchor="middle">' + value + '</text>'
      ).join('');
      const yTicks = [-1, 0, 1].map(value =>
        '<line x1="' + plot.left + '" y1="' + yScale(value) +
        '" x2="' + plot.right + '" y2="' + yScale(value) +
        '" class="' + (value === 0 ? 'axis-strong' : 'grid-line') + '"/>' +
        '<text x="' + (plot.left - 12) + '" y="' + (yScale(value) + 4) +
        '" class="mono muted" text-anchor="end">' + value + '</text>'
      ).join('');
      renderedWidth = width;
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);

      svg.innerHTML =
        '<title id="frequency-svg-title">正弦余弦函数图与当前位置的单位圆相位</title>' +
        '<desc id="frequency-svg-desc">左侧绘制所选维度对在零到六十四个 token 内的正弦和余弦，右侧显示当前位置对应的二维向量。</desc>' +
        '<defs>' +
          '<clipPath id="frequency-clip"><rect x="' + plot.left + '" y="' + plot.top +
          '" width="' + (plot.right - plot.left) + '" height="' + (plot.bottom - plot.top) + '"/></clipPath>' +
          '<marker id="phase-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="var(--blue)"/>' +
          '</marker>' +
        '</defs>' +
        '<rect data-chart-frame x="' + plot.left + '" y="' + plot.top +
        '" width="' + (plot.right - plot.left) + '" height="' + (plot.bottom - plot.top) +
        '" fill="none" stroke="var(--line)"/>' +
        xTicks + yTicks +
        '<g clip-path="url(#frequency-clip)">' +
          '<path d="' + functionPath(value => Math.sin(value * omega)) + '" class="series-sin"/>' +
          '<path d="' + functionPath(value => Math.cos(value * omega)) + '" class="series-cos"/>' +
          '<line x1="' + positionX + '" y1="' + plot.top + '" x2="' + positionX +
          '" y2="' + plot.bottom + '" class="position-guide"/>' +
        '</g>' +
        '<circle cx="' + positionX + '" cy="' + yScale(sinValue) + '" r="5" fill="var(--green)"/>' +
        '<circle cx="' + positionX + '" cy="' + yScale(cosValue) + '" r="5" fill="var(--red)"/>' +
        '<line x1="' + sinLegendX + '" y1="36" x2="' + (sinLegendX + 28) + '" y2="36" class="series-sin"/>' +
        '<text x="' + (sinLegendX + 36) + '" y="41">sin(pω)</text>' +
        '<line x1="' + cosLegendX + '" y1="36" x2="' + (cosLegendX + 28) + '" y2="36" class="series-cos"/>' +
        '<text x="' + (cosLegendX + 36) + '" y="41">cos(pω)</text>' +
        '<text x="' + ((plot.left + plot.right) / 2) +
        '" y="' + (plot.bottom + 48) + '" class="muted" text-anchor="middle">位置 p / token</text>' +
        '<text x="14" y="' + ((plot.top + plot.bottom) / 2) +
        '" class="muted" text-anchor="middle" transform="rotate(-90 14 ' +
        ((plot.top + plot.bottom) / 2) + ')">函数值</text>' +
        '<circle cx="' + circle.x + '" cy="' + circle.y + '" r="' + circle.radius + '" class="axis"/>' +
        '<line x1="' + (circle.x - circle.radius - 20) + '" y1="' + circle.y +
        '" x2="' + (circle.x + circle.radius + 20) +
        '" y2="' + circle.y + '" class="axis"/>' +
        '<line x1="' + circle.x + '" y1="' + (circle.y - circle.radius - 20) +
        '" x2="' + circle.x + '" y2="' + (circle.y + circle.radius + 20) + '" class="axis"/>' +
        '<line x1="' + circle.x + '" y1="' + circle.y + '" x2="' + vectorX +
        '" y2="' + vectorY + '" stroke="var(--blue)" stroke-width="3" marker-end="url(#phase-arrow)"/>' +
        '<line x1="' + vectorX + '" y1="' + circle.y + '" x2="' + vectorX +
        '" y2="' + vectorY + '" class="series-cos" style="stroke-width:2"/>' +
        '<line x1="' + circle.x + '" y1="' + circle.y + '" x2="' + vectorX +
        '" y2="' + circle.y + '" class="series-sin" style="stroke-width:2"/>' +
        '<circle cx="' + vectorX + '" cy="' + vectorY + '" r="5" fill="var(--blue)"/>' +
        '<text x="' + (circle.x + circle.radius + 18) + '" y="' + (circle.y - 8) +
        '" class="muted" text-anchor="end">sin</text>' +
        '<text x="' + (circle.x + 8) + '" y="' + (circle.y - circle.radius - 15) +
        '" class="muted">cos</text>' +
        '<text x="' + circle.x + '" y="' + (compact ? 574 : 301) +
        '" class="mono" text-anchor="middle">u(p) = (sin, cos)</text>' +
        '<text x="' + circle.x + '" y="' + (compact ? 598 : 325) +
        '" class="mono muted" text-anchor="middle">ω' +
        m + ' = ' + omega + '</text>';

      lab.querySelector('#frequency-position-value').value = p;
      readout.textContent =
        'PE[' + p + ', ' + (2 * m) + '] = sin(' + format(p * omega) +
        ') = ' + format(sinValue, 6) + ' · PE[' + p + ', ' + (2 * m + 1) +
        '] = cos(' + format(p * omega) + ') = ' + format(cosValue, 6);
    }

    [pairInput, positionInput].forEach(input => input.addEventListener('input', draw));
    new ResizeObserver(() => {
      const width = Math.max(320, Math.round(svg.clientWidth));
      if (width !== renderedWidth) draw();
    }).observe(svg);
    draw();
  }

  function blend(from, to, amount) {
    const channel = index =>
      Math.round(from[index] + (to[index] - from[index]) * amount);
    return 'rgb(' + channel(0) + ' ' + channel(1) + ' ' + channel(2) + ')';
  }

  function heatColor(value) {
    const paper = [250, 251, 249];
    const target = value >= 0 ? [8, 124, 112] : [197, 72, 52];
    return blend(paper, target, Math.abs(value) * 0.86);
  }

  function positionalRow(position) {
    return frequencies.flatMap(omega => [
      Math.sin(position * omega),
      Math.cos(position * omega),
    ]);
  }

  function initEncodingMatrixLab() {
    const lab = document.getElementById('encoding-matrix-lab');
    if (!lab) return;

    lab.innerHTML = heading('位置 × 维度：把波形堆成矩阵', '32 × 8 · heatmap') +
      '<div class="lab-controls">' +
        '<label>选中位置 p <output id="matrix-position-value">12</output>' +
          '<input id="matrix-position" aria-label="选择矩阵中的位置行" type="range" min="0" max="31" value="12">' +
        '</label>' +
      '</div>' +
      '<svg class="lab-svg" viewBox="0 0 840 510" role="img" aria-labelledby="matrix-svg-title matrix-svg-desc"></svg>' +
      '<div class="readout" aria-live="polite"></div>';

    const svg = lab.querySelector('svg');
    const positionInput = lab.querySelector('#matrix-position');
    const readout = lab.querySelector('.readout');
    let renderedWidth = 0;

    function draw() {
      const selected = Number(positionInput.value);
      const width = Math.max(320, Math.round(svg.clientWidth));
      const compact = width < 620;
      const height = compact ? 530 : 510;
      const xStart = compact ? 49 : 132;
      const xEnd = width - 8;
      const yStart = 66;
      const cellWidth = (xEnd - xStart) / 8;
      const cellHeight = 9.5;
      const labelClass = compact ? 'mono small' : 'mono';
      renderedWidth = width;
      svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
      const columnLabels = Array.from({length: 8}, (_, dimension) => {
        const fn = dimension % 2 === 0 ? 'sin' : 'cos';
        const pair = Math.floor(dimension / 2);
        const x = xStart + dimension * cellWidth + cellWidth / 2;
        return '<text x="' + x + '" y="28" class="' + labelClass +
          '" text-anchor="middle">d' +
          dimension + '</text><text x="' + x +
          '" y="48" class="' + labelClass + ' muted" text-anchor="middle">' +
          fn + (compact ? '' : ' ω' + pair) + '</text>';
      }).join('');
      const rowTicks = Array.from({length: 8}, (_, index) => index * 4).map(position => {
        const y = yStart + position * cellHeight + cellHeight / 2 + 4;
        return '<text x="' + (xStart - 14) + '" y="' + y +
          '" class="mono small muted" text-anchor="end">' + position + '</text>';
      }).join('');
      const cells = Array.from({length: 32}, (_, position) =>
        positionalRow(position).map((value, dimension) =>
          '<rect data-position="' + position + '" x="' +
          (xStart + dimension * cellWidth) + '" y="' + (yStart + position * cellHeight) +
          '" width="' + (cellWidth - 1) + '" height="' + (cellHeight - 1) +
          '" fill="' + heatColor(value) + '"><title>p=' + position + ', d' +
          dimension + ', ' + (dimension % 2 === 0 ? 'sin' : 'cos') +
          ' = ' + format(value, 6) + '</title></rect>'
        ).join('')
      ).join('');
      const selectedValues = positionalRow(selected);
      const selectedCells = selectedValues.map((value, dimension) => {
        const x = xStart + dimension * cellWidth;
        const textColor = value >= 0 ? 'var(--green)' : 'var(--red)';
        return '<rect x="' + x + '" y="403" width="' + (cellWidth - 1) +
          '" height="45" fill="var(--soft)"/><text x="' + (x + cellWidth / 2) +
          '" y="431" class="' + (compact ? 'mono small' : 'mono') +
          '" text-anchor="middle" style="fill:' + textColor + '">' +
          format(value, compact ? 1 : 3) + '</text>';
      }).join('');
      const selectedY = yStart + selected * cellHeight;
      const squaredNorm = selectedValues.reduce((sum, value) => sum + value ** 2, 0);

      svg.innerHTML =
        '<title id="matrix-svg-title">三十二个位置、八个维度的位置编码矩阵</title>' +
        '<desc id="matrix-svg-desc">每一行是一个位置，每两列是一对正弦余弦频率。绿色表示正值，红色表示负值，蓝框标出当前选中的位置。</desc>' +
        '<defs><linearGradient id="heat-legend" x1="0" x2="1">' +
          '<stop offset="0%" stop-color="rgb(197 72 52)"/>' +
          '<stop offset="50%" stop-color="rgb(250 251 249)"/>' +
          '<stop offset="100%" stop-color="rgb(8 124 112)"/>' +
        '</linearGradient></defs>' +
        columnLabels +
        '<text x="13" y="223" class="muted" text-anchor="middle" transform="rotate(-90 13 223)">位置 p</text>' +
        rowTicks + cells +
        '<rect x="' + (xStart - 3) + '" y="' + (selectedY - 2) +
        '" width="' + (cellWidth * 8 + 5) + '" height="' + (cellHeight + 3) +
        '" class="selected-row"/>' +
        '<line x1="' + (compact ? 14 : 80) + '" y1="382" x2="' + xEnd +
        '" y2="382" class="axis"/>' +
        '<text x="' + (xStart - 14) + '" y="431" class="mono small" text-anchor="end">PE</text>' +
        selectedCells +
        '<text x="' + xStart + '" y="487" class="muted">−1</text>' +
        '<rect x="' + (xStart + 28) + '" y="475" width="' + (compact ? 110 : 180) +
        '" height="12" fill="url(#heat-legend)"/>' +
        '<text x="' + (xStart + (compact ? 148 : 218)) + '" y="487" class="muted">+1</text>' +
        '<text x="' + (compact ? width / 2 : xEnd) + '" y="' + (compact ? 518 : 487) +
        '" class="mono muted" text-anchor="' + (compact ? 'middle' : 'end') +
        '">每个 sin/cos 对长度² = 1</text>';

      lab.querySelector('#matrix-position-value').value = selected;
      readout.textContent =
        'PE(' + selected + ') = [' + selectedValues.map(value => format(value)).join(', ') +
        '] · ‖PE(' + selected + ')‖² = ' + format(squaredNorm, 4);
    }

    positionInput.addEventListener('input', draw);
    svg.addEventListener('click', event => {
      const cell = event.target.closest('[data-position]');
      if (!cell) return;
      positionInput.value = cell.dataset.position;
      draw();
    });
    new ResizeObserver(() => {
      const width = Math.max(320, Math.round(svg.clientWidth));
      if (width !== renderedWidth) draw();
    }).observe(svg);
    draw();
  }

  function initCacheLab() {
    const cache = document.getElementById('cache-lab');
    if (!cache) return;

    cache.innerHTML = heading('当前查询与历史键的位置', 'cache · relative offset') +
      '<label>解码位置 t <output id="t-value">4</output>' +
        '<input id="t" aria-label="解码位置 t" type="range" min="0" max="8" value="4">' +
      '</label><div class="cache-grid"></div>' +
      '<p class="legend">绿色：历史缓存；红色：当前 token；灰色：未来，不可见。</p>' +
      '<div class="readout" aria-live="polite"></div>';

    function draw() {
      const t = Number(cache.querySelector('#t').value);
      cache.querySelector('#t-value').value = t;
      cache.querySelector('.cache-grid').innerHTML = Array.from({length: 9}, (_, j) =>
        '<div class="cache-cell ' + (j < t ? 'past' : j === t ? 'current' : 'future') +
        '">j = ' + j + '<br>' + (j <= t ? 'Δ = ' + (j - t) : 'masked') + '</div>'
      ).join('');
      cache.querySelector('.readout').textContent =
        'Q: [B,H,1,D] · K/V: [B,H,' + (t + 1) +
        ',D] · scores: [B,H,1,' + (t + 1) + ']';
    }

    cache.querySelector('input').addEventListener('input', draw);
    draw();
  }

  initRotationLab();
  initFrequencyLab();
  initEncodingMatrixLab();
  initCacheLab();
})();
