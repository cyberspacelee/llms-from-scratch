/* Shared reading behavior: theme, table scroll, section progress. */
(() => {
  const root = document.documentElement;
  if (root.dataset.readerBound === 'true') return;
  root.dataset.readerBound = 'true';

  const mode = () => localStorage.getItem('theme') || 'system';

  const applyTheme = (value) => {
    const dark = value === 'dark' || (value !== 'light' && matchMedia('(prefers-color-scheme: dark)').matches);
    root.classList.toggle('dark', dark);
    root.dataset.theme = value;
    root.style.colorScheme = dark ? 'dark' : 'light';
    const button = document.querySelector('.theme-toggle');
    if (!button) return;
    const label = value === 'light' ? '浅色' : value === 'dark' ? '深色' : '系统';
    button.textContent = label;
    button.setAttribute('aria-label', `配色：${label}。点击切换浅色、深色或跟随系统`);
  };

  applyTheme(mode());

  document.querySelector('.theme-toggle')?.addEventListener('click', () => {
    const order = ['system', 'light', 'dark'];
    const next = order[(Math.max(order.indexOf(mode()), 0) + 1) % order.length];
    if (next === 'system') localStorage.removeItem('theme');
    else localStorage.setItem('theme', next);
    applyTheme(next);
  });

  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (mode() === 'system') applyTheme('system');
  });

  const wrapTables = () => {
    document.querySelectorAll('table').forEach((table) => {
      if (table.parentElement?.classList.contains('table-scroll')) return;
      const wrap = document.createElement('div');
      wrap.className = 'table-scroll';
      wrap.tabIndex = 0;
      wrap.setAttribute('role', 'region');
      wrap.setAttribute('aria-label', '可横向滚动的表格');
      table.replaceWith(wrap);
      wrap.append(table);
    });
  };

  let spyReady = false;

  const chapterLinks = () => [...document.querySelectorAll('a.toc-link')];

  const sections = () => {
    const seen = new Set();
    const found = [];
    chapterLinks().forEach((link) => {
      const id = (link.getAttribute('href') || '').slice(1);
      if (!id || seen.has(id)) return;
      const section = document.getElementById(id);
      if (!section) return;
      seen.add(id);
      found.push(section);
    });
    return found;
  };

  const reveal = (id) => {
    const escaped = window.CSS && CSS.escape ? CSS.escape(id) : id.replace(/"/g, '\\"');
    document.querySelectorAll('.sidebar, .mobile-toc-list').forEach((scroller) => {
      const link = scroller.querySelector(`a.toc-link[href="#${escaped}"]`);
      if (!link) return;
      const linkBox = link.getBoundingClientRect();
      const view = scroller.getBoundingClientRect();
      if (linkBox.top < view.top) scroller.scrollTop += linkBox.top - view.top;
      else if (linkBox.bottom > view.bottom) scroller.scrollTop += linkBox.bottom - view.bottom;
    });
  };

  const boot = () => {
    wrapTables();
    if (spyReady || root.dataset.progress === 'off') return;
    const found = sections();
    if (!found.length) return;
    spyReady = true;
    found.forEach((section) => {
      if (!section.hasAttribute('tabindex')) section.tabIndex = -1;
    });
    const bar = document.getElementById('read-bar');
    const label = document.getElementById('read-label');
    let currentId = '';
    const mark = (id) => {
      if (!id || id === currentId) return;
      currentId = id;
      const index = found.findIndex((section) => section.id === id);
      chapterLinks().forEach((link) => {
        const on = link.getAttribute('href') === `#${id}`;
        if (on) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
      if (index >= 0 && label) label.textContent = `${index + 1} / ${found.length}`;
      if (index >= 0 && bar) bar.style.width = `${((index + 1) / found.length) * 100}%`;
      reveal(id);
    };
    const update = () => {
      const line = window.scrollY + Math.min(window.innerHeight * 0.28, 220);
      let active = found[0];
      found.forEach((section) => {
        const top = section.getBoundingClientRect().top + window.scrollY;
        if (top <= line) active = section;
      });
      if (active) mark(active.id);
    };
    document.addEventListener('click', (event) => {
      const link = event.target instanceof Element ? event.target.closest('a.toc-link') : null;
      if (!link) return;
      const id = (link.getAttribute('href') || '').slice(1);
      const target = document.getElementById(id);
      if (!target) return;
      target.tabIndex = -1;
      target.focus({ preventScroll: true });
    });
    window.addEventListener('scroll', update, { passive: true });
    update();
  };

  document.addEventListener('lesson-ready', boot);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
