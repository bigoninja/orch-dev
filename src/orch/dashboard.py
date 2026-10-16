"""A tiny read-only progress dashboard: ``orch dashboard``.

Serves one local HTML page that re-fetches ``/api/state`` every few seconds. The
data comes straight from the run's ``state.json`` — the same snapshot every
pipeline node writes — so the page is always as fresh as the run itself.

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
  main { padding: 16px 24px 40px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #21262d;
           vertical-align: top; }
  th { color: #8b949e; font-weight: 600; font-size: 12px; text-transform: uppercase; }
  td.notes { color: #8b949e; max-width: 340px; }
  .risky { color: #d29922; font-weight: 600; }
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
<footer><span id="updated">connecting…</span> · auto-refresh every 3s</footer>
<script>
function pillEl(status) {
  const s = document.createElement('span');
  s.className = 'pill ' + (status === 'escalated' ? 'escalated-task' : status);
  s.textContent = status;
  return s;
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
