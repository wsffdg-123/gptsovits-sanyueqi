# -*- coding: utf-8 -*-
"""三月七语音数据审查服务器。

读取 select/selection.csv 提供 /data.json，静态服务 tts 根目录（wav 支持 Range 拖动）。
- POST /api/edit    保存文本修正：select/edits.json + selection.csv + lists/*.list
- POST /api/remove  移出训练集：从 lists/*.list 与 selection.csv 移除，记录 select/removals.json
- POST /api/restore 撤销移出：按记录写回清单
用法: python scripts/serve_review.py [--port 6270] [--open]
页面: http://127.0.0.1:<port>/select/review.html
"""
import argparse
import csv
import json
import os
import re
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTION_CSV = os.path.join(HERE, "select", "selection.csv")
EDITS_JSON = os.path.join(HERE, "select", "edits.json")
REMOVALS_JSON = os.path.join(HERE, "select", "removals.json")
LISTS_DIR = os.path.join(HERE, "select", "lists")
LIST_FILES = ("train30.list", "train120.list", "full.list", "val.list")

LOCK = threading.Lock()
_BOOL = {"True": 1, "False": 0}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_edits() -> dict:
    if os.path.exists(EDITS_JSON):
        with open(EDITS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_removals() -> dict:
    if os.path.exists(REMOVALS_JSON):
        with open(REMOVALS_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_items(edits: dict, removals: dict) -> list:
    items = []
    with open(SELECTION_CSV, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            path = r["path"]
            text = edits.get(path, r.get("text") or "")
            items.append({
                "p": path,                            # wav 相对路径
                "t": text,                            # 清洗后文本（含人工修正）
                "r": r.get("raw_text") or "",         # 原始文本
                "d": round(fnum(r["dur"]) or 0.0, 2),
                "g": r.get("group") or "",
                "st": (r.get("sets") or "").strip(),
                "ex": r.get("excl") or "",
                "ti": r.get("tier") or "",
                "q": round(fnum(r["quality"]) if fnum(r["quality"]) is not None else 0, 2),
                "f0": round(fnum(r["f0_median"]) or 0),
                "db": round(fnum(r["speech_db"]) or 0, 1),
                "sr": int(fnum(r["sr"]) or 0),
                "mf": _BOOL.get(r.get("variant_mf"), 0),
                "bt": _BOOL.get(r.get("battle"), 0),
                "mem": _BOOL.get(r.get("memory"), 0),
                "fk": _BOOL.get(r.get("fake"), 0),
                "ed": 1 if path in edits else 0,
                "rm": 1 if path in removals else 0,
            })
    return items


def _rebuild_state():
    items = build_items(load_edits(), load_removals())
    ReviewHandler.items = items
    ReviewHandler.data_json = json.dumps(
        items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def csv_rows():
    with open(SELECTION_CSV, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(fields, rows):
    with open(SELECTION_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def update_csv_row(path: str, updates: dict):
    fields, rows = csv_rows()
    hit = False
    for r in rows:
        if r["path"] == path:
            r.update(updates)
            hit = True
    if not hit:
        raise KeyError(f"selection.csv 中找不到 {path}")
    write_csv(fields, rows)


def csv_row(path: str) -> dict:
    _, rows = csv_rows()
    for r in rows:
        if r["path"] == path:
            return r
    raise KeyError(f"selection.csv 中找不到 {path}")


def iter_list_lines(name: str):
    fp = os.path.join(LISTS_DIR, name)
    if not os.path.exists(fp):
        return None, []
    with open(fp, encoding="utf-8") as f:
        return fp, f.readlines()


def patch_lists_text(path: str, text: str):
    for name in LIST_FILES:
        fp, lines = iter_list_lines(name)
        if fp is None:
            continue
        changed = False
        for i, line in enumerate(lines):
            parts = line.rstrip("\n").split("|", 3)
            if parts[0] == path:
                parts[3] = text
                lines[i] = "|".join(parts) + "\n"
                changed = True
        if changed:
            with open(fp, "w", encoding="utf-8") as f:
                f.writelines(lines)


def remove_from_lists(path: str):
    for name in LIST_FILES:
        fp, lines = iter_list_lines(name)
        if fp is None:
            continue
        kept = [ln for ln in lines if ln.split("|", 1)[0] != path]
        if len(kept) != len(lines):
            with open(fp, "w", encoding="utf-8") as f:
                f.writelines(kept)


def add_to_lists(path: str, sets_tokens):
    text = csv_row(path).get("text") or ""
    line = f"{path}|mar7th|ZH|{text}\n"
    for name in LIST_FILES:
        set_name = name.rsplit(".", 1)[0]
        if set_name not in sets_tokens:
            continue
        fp = os.path.join(LISTS_DIR, name)
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            existing = {ln.split("|", 1)[0] for ln in f}
        if path not in existing:
            with open(fp, "a", encoding="utf-8") as f:
                f.write(line)


def _validate_path(path: str) -> str:
    full = os.path.realpath(os.path.join(HERE, path))
    if not full.startswith(os.path.realpath(HERE) + os.sep):
        raise ValueError("路径越界")
    if not os.path.isfile(full):
        raise FileNotFoundError(path)
    return full


def save_edit(path: str, text: str) -> dict:
    _validate_path(path)
    text = text.strip()
    if not text:
        raise ValueError("文本不能为空")
    if "\n" in text or "\r" in text:
        raise ValueError("文本不能包含换行（清单为行式格式）")
    with LOCK:
        edits = load_edits()
        edits[path] = text
        with open(EDITS_JSON, "w", encoding="utf-8") as f:
            json.dump(edits, f, ensure_ascii=False, indent=1)
        update_csv_row(path, {"text": text})
        patch_lists_text(path, text)
        _rebuild_state()
        return {"ok": True, "edited": len(edits)}


def remove_sample(path: str) -> dict:
    _validate_path(path)
    with LOCK:
        row = csv_row(path)
        sets_tokens = (row.get("sets") or "").split()
        removals = load_removals()
        removals[path] = {
            "sets": sets_tokens,
            "prev_excl": row.get("excl") or "",
            "reason": "人工移出",
        }
        with open(REMOVALS_JSON, "w", encoding="utf-8") as f:
            json.dump(removals, f, ensure_ascii=False, indent=1)
        remove_from_lists(path)
        update_csv_row(path, {"sets": "", "excl": "人工移出"})
        _rebuild_state()
        return {"ok": True, "removed": len(removals)}


def restore_sample(path: str) -> dict:
    _validate_path(path)
    with LOCK:
        removals = load_removals()
        if path not in removals:
            raise KeyError("该条目没有移出记录")
        rec = removals.pop(path)
        with open(REMOVALS_JSON, "w", encoding="utf-8") as f:
            json.dump(removals, f, ensure_ascii=False, indent=1)
        add_to_lists(path, rec.get("sets") or [])
        update_csv_row(path, {
            "sets": " ".join(rec.get("sets") or []),
            "excl": rec.get("prev_excl") or "",
        })
        _rebuild_state()
        return {"ok": True, "sets": " ".join(rec.get("sets") or []),
                "excl": rec.get("prev_excl") or ""}


class ReviewHandler(SimpleHTTPRequestHandler):
    data_json: bytes = b""
    items: list = []

    def log_message(self, fmt, *args):  # 静默常规访问日志
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/data.json":
            body = self.data_json
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path.lower().endswith(".wav"):
            if self._send_wav_with_range():
                return
        super().do_GET()

    def do_POST(self):
        route = self.path.split("?", 1)[0]
        if route not in ("/api/edit", "/api/remove", "/api/restore"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if route == "/api/edit":
                result = save_edit(body["path"], body["text"])
            elif route == "/api/remove":
                result = remove_sample(body["path"])
            else:
                result = restore_sample(body["path"])
            self._json_response(200, result)
        except (ValueError, FileNotFoundError, KeyError) as e:
            self._json_response(400, {"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001
            self._json_response(500, {"ok": False, "error": str(e)})

    def _json_response(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_wav_with_range(self) -> bool:
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return False
        size = os.path.getsize(path)
        rng = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip())
        if not m:
            return False  # 交给父类整文件返回
        start = int(m.group(1) or 0)
        end = int(m.group(2) or size - 1)
        end = min(end, size - 1)
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return True
        self.send_response(206)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            rest = end - start + 1
            while rest > 0:
                chunk = f.read(min(65536, rest))
                if not chunk:
                    break
                self.wfile.write(chunk)
                rest -= len(chunk)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6270)
    ap.add_argument("--open", action="store_true", help="启动后打开浏览器")
    args = ap.parse_args()

    edits = load_edits()
    removals = load_removals()
    ReviewHandler.items = build_items(edits, removals)
    ReviewHandler.data_json = json.dumps(
        ReviewHandler.items, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler = partial(ReviewHandler, directory=HERE)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/select/review.html"
    print(f"数据条目: {len(ReviewHandler.items)}"
          f"（人工修正 {len(edits)} 条，人工移出 {len(removals)} 条）")
    print(f"审查页面: {url}  (Ctrl+C 停止)")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
