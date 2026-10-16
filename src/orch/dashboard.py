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

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>orch dashboard</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #0d1117; color: #e6edf3;
         font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
  header { padding: 20px 24px 12px; border-bottom: 1px solid #21262d; }
  h1 { margin: 0 0 4px; font-size: 18px; }
  .sub { color: #8b949e; font-size: 13px; }
  .bar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px;
          font-size: 12px; font-weight: 600; letter-spacing: .02em; }
  .pill.running    { background: #1f6feb33; color: #79c0ff; }
  .pill.complete   { background: #23863633; color: #7ee787; }
  .pill.escalated  { background: #da363333; color: #ff7b72; }
  .pill.pending        { background: #30363d66; color: #8b949e; }
  .pill.implementing, .pill.review, .pill.fixing, .pill.approved
                       { background: #1f6feb33; color: #79c0ff; }
  .pill.integrated { background: #23863633; color: #7ee787; }
  .pill.escalated-task { background: #da363333; color: #ff7b72; }
  #banner { display: none; margin: 14px 24px 0; padding: 10px 14px; border-radius: 8px;
            background: #da36331a; border: 1px solid #da363366; color: #ff7b72; }
  main { padding: 16px 24px 8px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #21262d;
           vertical-align: top; }
  th { color: #8b949e; font-weight: 600; font-size: 12px; text-transform: uppercase; }
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
          padding: 10px 12px; margin-bottom: 10px; }
  .card .t { font-weight: 600; margin-bottom: 4px; }
  .card .repo { color: #8b949e; font-size: 12px; margin-bottom: 8px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .card .foot { display: flex; justify-content: space-between; gap: 8px;
                align-items: center; color: #8b949e; font-size: 12px; margin-top: 6px; }
  .col .empty { color: #484f58; font-size: 12px; padding: 8px 4px; }
  footer { padding: 0 24px 24px; color: #484f58; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>orch dashboard <span id="runid" class="sub"></span></h1>
  <div id="brief" class="sub"></div>
  <div class="bar">
    <span id="state" class="pill">…</span>
    <span id="integ" class="sub"></span>
    <span class="sub">→ <span id="branch"></span></span>
  </div>
</header>
<div id="banner"></div>
<main>
  <table>
    <thead><tr><th>Task</th><th>Title</th><th>State</th><th>Attempts</th><th>Review</th><th>Notes</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
</main>
<div class="board-title">Board</div>
<div id="board" class="board"></div>
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
function renderCard(t) {
  const card = document.createElement('div'); card.className = 'card';
  const title = document.createElement('div'); title.className = 't';
  title.textContent = t.title || t.id;
  if (t.risky) { const m = document.createElement('span');
                 m.className = 'risky'; m.textContent = ' ⚑';
                 m.title = 'risky (migrations/auth/payments)'; title.appendChild(m); }
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
    const n = document.createElement('div'); n.className = 'repo';
    n.style.marginTop = '6px'; n.textContent = truncate(t.notes, 80);
    card.appendChild(n);
  }
  return card;
}
function renderBoard(tasks) {
  const board = document.getElementById('board');
  board.replaceChildren();
  for (const col of COLUMNS) {
    const items = tasks.filter(t => col.match(t.status));
    const box = document.createElement('div'); box.className = 'col';
    const h = document.createElement('h2');
    const label = document.createElement('span'); label.textContent = col.label;
    const n = document.createElement('span'); n.className = 'n'; n.textContent = items.length;
    h.appendChild(label); h.appendChild(n); box.appendChild(h);
    if (!items.length) {
      const e = document.createElement('div'); e.className = 'empty'; e.textContent = '—';
      box.appendChild(e);
    } else {
      for (const t of items) box.appendChild(renderCard(t));
    }
    board.appendChild(box);
  }
}
async function tick() {
  let d;
  try {
    const r = await fetch('/api/state', { cache: 'no-store' });
    d = await r.json();
  } catch (e) { return; }
  if (d.error) { document.getElementById('banner').style.display = 'block';
                 document.getElementById('banner').textContent = d.error; return; }
  document.getElementById('runid').textContent = d.run_id;
  document.getElementById('brief').textContent = d.brief;
  const st = document.getElementById('state');
  st.textContent = d.run_state;
  st.className = 'pill ' + d.run_state.toLowerCase();
  document.getElementById('integ').textContent = d.integrated + ' / ' + d.total + ' integrated';
  document.getElementById('branch').textContent = d.integration_branch;
  const b = document.getElementById('banner');
  if (d.escalation) {
    b.style.display = 'block';
    b.textContent = 'ESCALATION · ' + d.escalation.kind +
      ' · task ' + (d.escalation.task_id || '—') +
      ' · ' + (d.escalation.message || '');
  } else { b.style.display = 'none'; }
  const tbody = document.getElementById('rows');
  tbody.replaceChildren();
  for (const t of d.tasks) {
    const tr = document.createElement('tr');
    const add = (txt, cls) => {
      const td = document.createElement('td');
      if (cls) td.className = cls;
      td.textContent = txt;
      tr.appendChild(td);
      return td;
    };
    add(t.id);
    const title = add(t.title);
    if (t.risky) { const m = document.createElement('span');
                   m.className = 'risky'; m.textContent = ' ⚑';
                   m.title = 'risky (migrations/auth/payments)'; title.appendChild(m); }
    add('').appendChild(pillEl(t.status));
    add(String(t.attempts) + (t.fix_attempts ? ' / fix:' + t.fix_attempts : ''));
    add(t.review || '—');
    add(t.notes || '—', 'notes');
    tbody.appendChild(tr);
  }
  renderBoard(d.tasks);
  document.getElementById('updated').textContent = 'updated ' + new Date().toLocaleTimeString();
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


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
