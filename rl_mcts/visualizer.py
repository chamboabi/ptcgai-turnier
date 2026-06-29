#!/usr/bin/env python3
import sys
import json
import tempfile
import webbrowser
import html
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/file.json")
        sys.exit(1)

    path = Path(sys.argv[1])
    raw = path.read_text()
    obj = json.loads(raw)

    if "steps" in obj:
        payload = json.dumps(obj["steps"][0][0]["visualize"])
    else:
        payload = raw

    escaped = html.escape(payload, quote=True)

    page = f"""<!DOCTYPE html>
<html>
<body>
<form id="f" method="POST" action="https://ptcgvis.heroz.jp/Visualizer/Replay/0" target="_blank">
  <input type="hidden" name="json" value="{escaped}">
</form>
<script>document.getElementById('f').submit();</script>
</body>
</html>"""

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(page)
        tmp = f.name

    webbrowser.open(f"file://{tmp}")

if __name__ == "__main__":
    main()
