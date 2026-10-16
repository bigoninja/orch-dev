"""A tiny read-only progress dashboard: ``orch dashboard``.

Serves one local HTML page that re-fetches ``/api/state`` every few seconds. The
data comes straight from the run's ``state.json`` — the same snapshot every
pipeline node writes — so the page is always as fresh as the run itself.

The page shows a header (run + escalation banner), the task table, and a kanban
board: Queued → Working → In review → Needs you → Ready to merge.

Strictly read-only: it never touches the graph, the lock, or the target repo.
"""

from __future__ import annotations

import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from orch.artifacts import RunDir

_TERMINAL_OK = {"integrated"}

_LOGO = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 295 90"><g fill="#ffffff" transform="matrix(5.2863438788431925,0,0,5.2863438788431925,122.97796302478926,0.19823858340447487)"><path d="M4.24 14.15Q2.83 14.15 1.89 13.27Q0.95 12.38 0.95 10.49L0.95 10.49Q0.95 9.21 1.41 8.39Q1.86 7.57 2.60 7.20Q3.35 6.83 4.24 6.83L4.24 6.83Q5.14 6.83 5.87 7.20Q6.61 7.57 7.06 8.39Q7.52 9.21 7.52 10.49L7.52 10.49Q7.52 11.77 7.06 12.59Q6.61 13.41 5.87 13.78Q5.14 14.15 4.24 14.15L4.24 14.15ZM4.24 12.81Q4.90 12.81 5.38 12.27Q5.87 11.73 5.87 10.49L5.87 10.49Q5.87 9.25 5.38 8.71Q4.90 8.18 4.24 8.18L4.24 8.18Q3.57 8.18 3.09 8.71Q2.60 9.24 2.60 10.49L2.60 10.49Q2.60 11.73 3.09 12.27Q3.57 12.81 4.24 12.81L4.24 12.81ZM14.97 8.22Q14.29 8.22 13.82 8.53Q13.34 8.83 13.13 9.23L13.13 9.23L13.13 12.67L15.34 12.67L15.34 14L9.84 14L9.84 12.67L11.55 12.67L11.55 8.33L9.84 8.33L9.84 7L12.71 7L12.98 7.85Q13.36 7.36 13.93 7.10Q14.50 6.83 15.25 6.83L15.25 6.83Q15.92 6.83 16.45 7.10L16.45 7.10L15.99 8.44Q15.92 8.37 15.64 8.29Q15.37 8.22 14.97 8.22L14.97 8.22ZM24.32 12.57Q23.39 14.15 21.52 14.15L21.52 14.15Q20.03 14.15 19.09 13.27Q18.14 12.38 18.14 10.49L18.14 10.49Q18.14 8.58 19.08 7.71Q20.02 6.83 21.52 6.83L21.52 6.83Q22.43 6.83 23.14 7.22Q23.84 7.62 24.32 8.51L24.32 8.51L22.89 9.28Q22.44 8.18 21.48 8.18L21.48 8.18Q20.71 8.18 20.24 8.72Q19.78 9.27 19.78 10.49L19.78 10.49Q19.78 11.69 20.24 12.25Q20.71 12.81 21.48 12.81L21.48 12.81Q22.39 12.81 22.89 11.83L22.89 11.83L24.32 12.57ZM29.97 6.85Q31.23 6.85 31.84 7.57Q32.45 8.30 32.45 9.97L32.45 9.97L32.45 14L30.88 14L30.88 10.56Q30.88 9.69 30.79 9.20Q30.70 8.71 30.42 8.45Q30.14 8.19 29.58 8.19L29.58 8.19Q28.90 8.19 28.43 8.55Q27.96 8.90 27.73 9.34L27.73 9.34L27.73 14L26.17 14L26.17 2.80L27.73 2.80L27.73 7.67Q28.13 7.28 28.70 7.06Q29.27 6.85 29.97 6.85L29.97 6.85Z"></path></g><g transform="matrix(1.1111111111111112,0,0,1.1111111111111112,-1.1999982992808071,-6.666666666666671)" stroke="none" fill="#F69800"><path d="M9.9 87H99V14.1H50.4v8.1h40.5v56.699H18V54.601H9.9z"></path><path d="M34.2 14.1h-8.1V6H9.9v24.3H1.8v16.2H18V22.2h16.2v16.2h-8.1v16.201h16.2V14.1z"></path></g></svg>"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>orch dashboard</title>
<style>
  :root { color-scheme: dark; --faint: #7d8590; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding: 20px 24px 12px; border-bottom: 1px solid #21262d;
           position: sticky; top: 0; z-index: 3; background: #0d1117; }
  h1 { display: flex; align-items: center; gap: 12px; margin: 0 0 4px; font-size: 18px; }
  .logo svg { height: 30px; width: auto; display: block; }
  .sub { color: #8b949e; font-size: 13px; }
  .bar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 12px; font-weight: 600; letter-spacing: .02em;
          transition: background .15s ease, color .15s ease; }
  .pill.running    { background: #1f6feb33; color: #79c0ff; }
  .pill.complete   { background: #23863633; color: #7ee787; }
  .pill.escalated  { background: #da363333; color: #ff7b72; }
  .pill.pending      { background: #30363d66; color: #8b949e; }
  .pill.implementing { background: #1f6feb33; color: #79c0ff; }
  .pill.review       { background: #d2992233; color: #e3b341; }
  .pill.fixing       { background: #db6d2833; color: #f0883e; }
  .pill.approved     { background: #8957e533; color: #d2a8ff; }
  .pill.integrated { background: #23863633; color: #7ee787; }
  .pill.escalated-task { background: #da363333; color: #ff7b72; }
  .progress { height: 3px; background: #21262d; }
  .progress > i { display: block; height: 100%; width: 0; background: #238636;
                  transition: width .3s ease; }
  #banner { display: none; margin: 14px 24px 0; padding: 14px 16px; border-radius: 8px;
            background: #da36331a; border: 1px solid #da363366; color: #ff7b72;
            font-size: 14px; align-items: center; gap: 12px; flex-wrap: wrap; }
  #banner.on { display: flex; }
  #banner .hint { color: #e6edf3; background: #0d1117; border: 1px solid #21262d;
                  border-radius: 6px; padding: 2px 8px; font-size: 12px; }
  #banner button { margin-left: auto; background: #21262d; color: #e6edf3;
                   border: 1px solid #30363d; border-radius: 6px; padding: 4px 10px;
                   font: inherit; font-size: 12px; cursor: pointer; }
  #banner button:hover { background: #30363d; }
  main { padding: 16px 24px 8px; }
  .tablewrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #21262d;
           vertical-align: top; }
  th { color: #8b949e; font-weight: 600; font-size: 12px; text-transform: uppercase; }
  tbody tr:hover { background: #161b22; }
  td.notes { color: #8b949e; max-width: 340px; }
  .risky { color: #d29922; font-weight: 600; }
  .board-title { padding: 14px 24px 8px; color: #8b949e; font-size: 12px;
                 text-transform: uppercase; letter-spacing: .06em;
                 border-top: 1px solid #21262d; margin-top: 8px; }
  .board { display: flex; gap: 14px; padding: 0 24px 24px; overflow-x: auto;
           align-items: flex-start; }
  .col { min-width: 250px; width: 250px; background: #161b22;
         border: 1px solid #21262d; border-radius: 10px; padding: 10px; }
  .col h2 { margin: 2px 4px 10px; font-size: 12px; text-transform: uppercase;
            letter-spacing: .04em; color: #8b949e; display: flex;
            justify-content: space-between; align-items: baseline; }
  .col h2 .n { color: #e6edf3; font-size: 12px; }
  .card { background: #0d1117; border: 1px solid #21262d; border-radius: 8px;
          padding: 10px 12px; margin-bottom: 10px; transition: border-color .15s ease; }
  .card:hover { border-color: #30363d; }
  .card .t { font-weight: 600; margin-bottom: 4px; }
  .card .repo { color: #8b949e; font-size: 12px; margin-bottom: 8px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .notes { color: var(--faint); font-size: 12px; margin-top: 6px;
                 overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .foot { display: flex; justify-content: space-between; gap: 8px;
                align-items: center; flex-wrap: wrap; row-gap: 2px;
                color: #8b949e; font-size: 12px; margin-top: 6px; }
  .col .empty { color: var(--faint); font-size: 12px; padding: 8px 4px; }
  .empty-state { color: var(--faint); padding: 18px 4px; }
  .nav { display: flex; justify-content: center; gap: 8px; padding: 12px 24px 0; }
  .navlink { padding: 6px 20px; border-radius: 999px; border: 1px solid #21262d;
             color: #8b949e; text-decoration: none; font-size: 13px; font-weight: 600; }
  .navlink.active { background: #1f6feb33; color: #79c0ff; border-color: #1f6feb66; }
  a:focus-visible, button:focus-visible { outline: 2px solid #1f6feb; outline-offset: 2px; }
  footer { padding: 0 24px 24px; color: var(--faint); font-size: 12px; }
  footer.disconnected { color: #f0883e; }
  @media (max-width: 760px) {
    th:nth-child(4), td:nth-child(4),
    th:nth-child(5), td:nth-child(5) { display: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { transition: none !important; animation: none !important; }
  }
</style>
</head>
<body>
<header>
  <h1><span class="logo" aria-hidden="true"><!--LOGO--></span><span id="runid" class="sub"></span></h1>
  <div class="bar">
    <span id="state" class="pill">…</span>
    <span id="integ" class="sub"></span>
    <span class="sub">→ <span id="branch"></span></span>
  </div>
</header>
<div class="progress" aria-hidden="true"><i id="progress"></i></div>
<nav class="nav">
  <a href="#status" id="nav-status" class="navlink active" aria-current="page">Status</a>
  <a href="#cards" id="nav-cards" class="navlink">Cards</a>
</nav>
<div id="banner" role="alert">
  <span id="banner-msg"></span>
  <code class="hint" id="banner-hint">orch resume</code>
  <button id="banner-copy" type="button">copy</button>
</div>
<section id="view-status">
<main>
  <div class="tablewrap">
  <table>
    <thead><tr><th>Task</th><th>Title</th><th>State</th><th>Attempts</th><th>Review</th><th>Notes</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  </div>
  <div id="rows-empty" class="empty-state" hidden>No tasks yet.</div>
</main>
</section>
<section id="view-cards" hidden>
<div class="board-title">Board</div>
<div id="board" class="board" role="list"></div>
</section>
<footer><span id="updated">connecting…</span> · auto-refresh every 3s</footer>
<script>
function pillEl(status) {
  const s = document.createElement('span');
  s.className = 'pill ' + (status === 'escalated' ? 'escalated-task' : status);
  s.textContent = status;
  return s;
}
function statusText(t) {
  switch (t.status) {
    case 'pending': return 'Queued';
    case 'implementing': return 'Working';
    case 'fixing': return 'Changes requested' + (t.fix_attempts ? ' · fix ' + t.fix_attempts : '');
    case 'review': return 'Review pending';
    case 'approved': return 'Approved — integrating';
    case 'integrated': return 'Ready';
    case 'escalated': return 'Needs you';
    default: return t.status;
  }
}
const COLUMNS = [
  { key: 'queued',  label: 'Queued',          match: s => s === 'pending' },
  { key: 'working', label: 'Working',         match: s => s === 'implementing' || s === 'fixing' },
  { key: 'review',  label: 'In review',       match: s => s === 'review' },
  { key: 'needs',   label: 'Needs you',       match: s => s === 'escalated' },
  { key: 'ready',   label: 'Ready to merge',  match: s => s === 'approved' || s === 'integrated' },
];
function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }
function riskyMark() {
  const m = document.createElement('span');
  m.className = 'risky'; m.textContent = ' ⚑';
  m.title = 'risky (migrations/auth/payments)';
  m.setAttribute('aria-label', 'risky: migrations, auth, or payments');
  return m;
}
function updateRow(tr, t) {
  tr.replaceChildren();
  const add = (txt, cls) => {
    const td = document.createElement('td');
    if (cls) td.className = cls;
    td.textContent = txt;
    tr.appendChild(td);
    return td;
  };
  add(t.id);
  const title = add(t.title);
  if (t.risky) title.appendChild(riskyMark());
  add('').appendChild(pillEl(t.status));
  add(String(t.attempts) + (t.fix_attempts ? ' / fix:' + t.fix_attempts : ''));
  add(t.review || '—');
  add(t.notes || '—', 'notes');
}
function updateCard(card, t) {
  card.replaceChildren();
  const title = document.createElement('div'); title.className = 't';
  title.textContent = t.title || t.id;
  if (t.risky) title.appendChild(riskyMark());
  card.appendChild(title);
  const repo = document.createElement('div'); repo.className = 'repo';
  repo.textContent = t.branch || ('agent/' + t.id);
  card.appendChild(repo);
  const foot = document.createElement('div'); foot.className = 'foot';
  const left = document.createElement('span');
  left.appendChild(pillEl(t.status === 'escalated' ? 'escalated' : t.status));
  left.appendChild(document.createTextNode(' ' + statusText(t)));
  foot.appendChild(left);
  const right = document.createElement('span');
  right.textContent = t.attempts
    ? ('attempts ' + t.attempts + (t.fix_attempts ? ' / fix:' + t.fix_attempts : ''))
    : '';
  foot.appendChild(right);
  card.appendChild(foot);
  if (t.notes) {
    const n = document.createElement('div'); n.className = 'notes';
    n.textContent = truncate(t.notes, 80);
    card.appendChild(n);
  }
}
// Reconcile children of `parent` against `items`, keyed by id, REUSING nodes so
// hover and text selection survive the 3s poll (a full rebuild flickers). A node
// whose data is unchanged is not touched at all.
function syncList(parent, items, make, update) {
  const existing = new Map();
  for (const child of Array.from(parent.children)) {
    if (child.dataset.id) existing.set(child.dataset.id, child);
  }
  const seen = new Set();
  for (const item of items) {
    const id = item.id;
    seen.add(id);
    let node = existing.get(id);
    if (!node) { node = make(); node.dataset.id = id; node._sig = null; }
    const sig = JSON.stringify(item);
    if (node._sig !== sig) { update(node, item); node._sig = sig; }
    parent.appendChild(node);  // moves existing nodes; never recreates them
  }
  for (const [id, node] of existing) if (!seen.has(id)) node.remove();
}
function renderBoard(tasks) {
  const board = document.getElementById('board');
  const existing = new Map();
  for (const col of Array.from(board.children)) {
    if (col.dataset.col) existing.set(col.dataset.col, col);
  }
  for (const col of COLUMNS) {
    const items = tasks.filter(t => col.match(t.status));
    let box = existing.get(col.key);
    if (!box) {
      box = document.createElement('div'); box.className = 'col'; box.dataset.col = col.key;
      box.setAttribute('role', 'listitem');
      const h = document.createElement('h2');
      const label = document.createElement('span'); label.textContent = col.label;
      const n = document.createElement('span'); n.className = 'n';
      h.appendChild(label); h.appendChild(n); box.appendChild(h);
      const cards = document.createElement('div'); cards.className = 'cards';
      cards.setAttribute('role', 'list');
      box.appendChild(cards);
    }
    box.querySelector('h2 .n').textContent = items.length;
    const cards = box.querySelector('.cards');
    if (!items.length) {
      cards.replaceChildren();
      const e = document.createElement('div'); e.className = 'empty'; e.textContent = '—';
      cards.appendChild(e);
    } else {
      for (const e of Array.from(cards.children)) {
        if (e.classList.contains('empty')) e.remove();
      }
      syncList(cards, items, () => {
        const c = document.createElement('div');
        c.className = 'card'; c.setAttribute('role', 'listitem');
        return c;
      }, updateCard);
    }
    board.appendChild(box);
  }
  for (const [key, box] of existing) {
    if (!COLUMNS.some(c => c.key === key)) box.remove();
  }
}
async function tick() {
  let d;
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    d = await r.json();
  } catch (e) {
    const up = document.getElementById('updated');
    up.textContent = 'disconnected — retrying…';
    up.parentElement.classList.add('disconnected');
    return;
  }
  const up = document.getElementById('updated');
  up.parentElement.classList.remove('disconnected');
  const b = document.getElementById('banner');
  const msg = document.getElementById('banner-msg');
  const hint = document.getElementById('banner-hint');
  const copy = document.getElementById('banner-copy');
  if (d.error) {
    b.classList.add('on'); msg.textContent = d.error;
    hint.hidden = true; copy.hidden = true; return;
  }
  document.getElementById('runid').textContent = d.run_id;
  document.getElementById('runid').title = d.brief;
  document.title = 'orch · ' + d.run_id + ' · ' + d.run_state;
  const st = document.getElementById('state');
  st.textContent = d.run_state;
  st.className = 'pill ' + d.run_state.toLowerCase();
  document.getElementById('integ').textContent = d.integrated + ' / ' + d.total + ' integrated';
  document.getElementById('branch').textContent = d.integration_branch;
  const pct = d.total ? Math.round(100 * d.integrated / d.total) : 0;
  document.getElementById('progress').style.width = pct + '%';
  if (d.escalation) {
    b.classList.add('on');
    msg.textContent = 'ESCALATION · ' + d.escalation.kind +
      ' · task ' + (d.escalation.task_id || '—') +
      ' · ' + (d.escalation.message || '');
    const cmd = 'orch resume --run ' + d.run_id;
    hint.hidden = false; hint.textContent = cmd; copy.hidden = false;
    copy.onclick = () => { if (navigator.clipboard) navigator.clipboard.writeText(cmd); };
  } else {
    b.classList.remove('on');
  }
  const tbody = document.getElementById('rows');
  syncList(tbody, d.tasks, () => document.createElement('tr'), updateRow);
  document.getElementById('rows-empty').hidden = d.tasks.length > 0;
  renderBoard(d.tasks);
  up.textContent = 'updated ' + (d.updated_at || '—');
}
function setView(view) {
  const isCards = view === 'cards';
  document.getElementById('view-status').hidden = isCards;
  document.getElementById('view-cards').hidden = !isCards;
  document.getElementById('nav-status').classList.toggle('active', !isCards);
  document.getElementById('nav-cards').classList.toggle('active', isCards);
  document.getElementById('nav-status').setAttribute('aria-current', isCards ? 'false' : 'page');
  document.getElementById('nav-cards').setAttribute('aria-current', isCards ? 'page' : 'false');
}
window.addEventListener('hashchange', () => {
  setView(location.hash === '#cards' ? 'cards' : 'status');
});
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) tick();
});
setView(location.hash === '#cards' ? 'cards' : 'status');
tick();
setInterval(() => { if (!document.hidden) tick(); }, 3000);
</script>
</body>
</html>
"""

_PAGE = _PAGE.replace("<!--LOGO-->", _LOGO)


def build_dashboard_state(run_dir: RunDir) -> dict:
    """The dashboard payload, computed from ``state.json`` (never mutated)."""
    if not run_dir.state_path.is_file():
        return {"error": "no state.json for this run yet — has the run started?"}
    try:
        data = json.loads(run_dir.state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"error": f"state.json unreadable: {e}"}

    tasks = data.get("tasks") or {}
    task_state = data.get("task_state") or {}
    rows: list[dict] = []
    integrated = 0
    for tid, tdef in tasks.items():
        st = task_state.get(tid) or {}
        status = str(st.get("status") or "pending")
        if status in _TERMINAL_OK:
            integrated += 1
        verdicts = st.get("verdicts") or []
        review = ""
        if verdicts and isinstance(verdicts[-1], dict):
            review = str(verdicts[-1].get("verdict") or "")
        notes = str(st.get("last_error") or "")
        if not notes and st.get("violation"):
            notes = "edited disallowed path(s): " + ", ".join(st["violation"])
        rows.append(
            {
                "id": tid,
                "title": str(tdef.get("title") or ""),
                "status": status,
                "attempts": int(st.get("attempts") or 0),
                "fix_attempts": int(st.get("fix_attempts") or 0),
                "review": review,
                "risky": bool(tdef.get("risky")),
                "notes": notes,
                "branch": str(st.get("branch") or f"agent/{tid}"),
                "commits": len(st.get("commits") or []),
            }
        )

    escalation = data.get("escalation")
    if escalation:
        run_state = "ESCALATED"
    elif data.get("summary") or (rows and all(r["status"] in _TERMINAL_OK for r in rows)):
        run_state = "COMPLETE"
    else:
        run_state = "RUNNING"

    try:
        updated = datetime.fromtimestamp(run_dir.state_path.stat().st_mtime).isoformat(
            timespec="seconds"
        )
    except OSError:
        updated = ""

    return {
        "run_id": str(data.get("run_id") or run_dir.root.name),
        "brief": str(data.get("brief") or ""),
        "run_state": run_state,
        "integration_branch": str(data.get("integration_branch") or ""),
        "integrated": integrated,
        "total": len(rows),
        "escalation": escalation,
        "tasks": rows,
        "updated_at": updated,
    }


def _handler_for(run_dir: RunDir):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path in ("/", "/index.html"):
                body = _PAGE.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            elif self.path.startswith("/api/state"):
                body = json.dumps(build_dashboard_state(run_dir)).encode("utf-8")
                ctype = "application/json"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def serve(run_dir: RunDir, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve the dashboard until Ctrl-C. Read-only; safe alongside a live run."""
    httpd = ThreadingHTTPServer((host, port), _handler_for(run_dir))
    print(f"orch dashboard: http://{host}:{port}  (run {run_dir.root.name}) — Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
