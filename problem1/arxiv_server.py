from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import re
import sys
import os
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "sample_data")
PAPERS_JSON = os.path.join(DATA_DIR, "papers.json")
CORPUS_JSON = os.path.join(DATA_DIR, "corpus_analysis.json")

PAPERS = []
PAPER_BY_ID = {}

def _tokenize(text: str):
    return re.findall(r"[A-Za-z0-9]+", text.lower())

def _sentences(text: str):
    parts = re.split(r"[.!?]+", text.strip())
    return [p for p in (s.strip() for s in parts) if p]

def _compute_abstract_stats(abstract: str):
    toks = _tokenize(abstract)
    return {
        "total_words": len(toks),
        "unique_words": len(set(toks)),
        "total_sentences": len(_sentences(abstract)),
    }

def _safe_iso(dt_str: str | None):
    if not dt_str:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", dt_str):
        return dt_str
    return None

def load_data():
    global PAPERS, PAPER_BY_ID
    if os.path.exists(PAPERS_JSON):
        with open(PAPERS_JSON, "r", encoding="utf-8") as f:
            PAPERS = json.load(f)
    else:
        PAPERS = []

    for p in PAPERS:
        p.setdefault("arxiv_id", "")
        p.setdefault("title", "")
        p.setdefault("authors", [])
        p.setdefault("abstract", "")
        p.setdefault("categories", [])
        if _safe_iso(p.get("published")) is None:
            p.pop("published", None)

        p.setdefault("abstract_stats", _compute_abstract_stats(p.get("abstract", "")))

        PAPER_BY_ID[p["arxiv_id"]] = p

load_data()

CORPUS_ANALYSIS = None
if os.path.exists(CORPUS_JSON):
    with open(CORPUS_JSON, "r", encoding="utf-8") as f:
        CORPUS_ANALYSIS = json.load(f)

def build_corpus_stats():
    total_papers = len(PAPERS)
    vocab = {}
    cat_count = {}
    total_words = 0

    for p in PAPERS:
        toks = _tokenize(p.get("abstract", ""))
        total_words += len(toks)
        for t in toks:
            vocab[t] = vocab.get(t, 0) + 1

        for c in p.get("categories", []):
            cat_count[c] = cat_count.get(c, 0) + 1

    top = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))[:10]
    top_10_words = [{"word": w, "frequency": f} for w, f in top]

    return {
        "total_papers": total_papers,
        "total_words": total_words,
        "unique_words": len(vocab),
        "top_10_words": top_10_words,
        "category_distribution": cat_count,
    }


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class ArxivHandler(BaseHTTPRequestHandler):
    server_version = "ArxivServer/1.0"

    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_request_line(self, status_code: int, extra: str = ""):
        msg = f"[{now_str()}] {self.command} {self.path} - {status_code}"
        if status_code == 200:
            msg += " OK"
        elif status_code == 400:
            msg += " Bad Request"
        elif status_code == 404:
            msg += " Not Found"
        elif status_code >= 500:
            msg += " Server Error"
        if extra:
            msg += f" {extra}"
        print(msg, flush=True)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            query = parse_qs(parsed.query)

            if path == "/papers":
                return self.handle_papers()

            if path.startswith("/papers/"):
                arxiv_id = path.split("/", 2)[2] if len(path.split("/", 2)) == 3 else ""
                return self.handle_paper_detail(arxiv_id)

            if path == "/search":
                return self.handle_search(query)

            if path == "/stats":
                return self.handle_stats()

            self.send_json({"error": "endpoint not found"}, status=404)
            self.log_request_line(404)
        except Exception as e:
            self.send_json({"error": str(e)}, status=500)
            self.log_request_line(500)

    def handle_papers(self):
        results = []
        for p in PAPERS:
            results.append({
                "arxiv_id": p.get("arxiv_id", ""),
                "title": p.get("title", ""),
                "authors": p.get("authors", []),
                "categories": p.get("categories", []),
            })
        self.send_json(results, status=200)
        self.log_request_line(200, f"({len(results)} results)")

    def handle_paper_detail(self, arxiv_id: str):
        p = PAPER_BY_ID.get(arxiv_id)
        if not p:
            self.send_json({"error": "paper not found"}, status=404)
            self.log_request_line(404)
            return
        resp = {
            "arxiv_id": p.get("arxiv_id", ""),
            "title": p.get("title", ""),
            "authors": p.get("authors", []),
            "abstract": p.get("abstract", ""),
            "categories": p.get("categories", []),
            "published": p.get("published", None),
            "abstract_stats": p.get("abstract_stats", _compute_abstract_stats(p.get("abstract", ""))),
        }
        self.send_json(resp, status=200)
        self.log_request_line(200)

    def handle_search(self, query: dict):
        q_list = query.get("q", [])
        if not q_list or not q_list[0].strip():
            self.send_json({"error": "missing or empty 'q' parameter"}, status=400)
            self.log_request_line(400)
            return

        raw = q_list[0]
        terms = [t for t in _tokenize(raw) if t]
        if not terms:
            self.send_json({"error": "malformed query"}, status=400)
            self.log_request_line(400)
            return

        results = []
        for p in PAPERS:
            title = p.get("title", "")
            abstract = p.get("abstract", "")

            title_low = title.lower()
            abs_low = abstract.lower()

            def term_occurrences(text_low: str, t: str) -> int:
                return len(re.findall(rf"\b{re.escape(t)}\b", text_low))

            match_all = True
            score = 0
            in_title = False
            in_abs = False
            for t in terms:
                occ_t = term_occurrences(title_low, t)
                occ_a = term_occurrences(abs_low, t)
                if occ_t + occ_a == 0:
                    match_all = False
                    break
                score += (occ_t + occ_a)
                if occ_t > 0:
                    in_title = True
                if occ_a > 0:
                    in_abs = True

            if match_all:
                matches_in = []
                if in_title:
                    matches_in.append("title")
                if in_abs:
                    matches_in.append("abstract")
                results.append({
                    "arxiv_id": p.get("arxiv_id", ""),
                    "title": title,
                    "match_score": score,
                    "matches_in": matches_in,
                })

        results.sort(key=lambda x: (-x["match_score"], x["arxiv_id"]))

        resp = {"query": raw, "results": results}
        self.send_json(resp, status=200)
        self.log_request_line(200, f"({len(results)} results)")

    def handle_stats(self):
        if CORPUS_ANALYSIS:
            self.send_json(CORPUS_ANALYSIS, status=200)
            self.log_request_line(200)
            return
        stats = build_corpus_stats()
        self.send_json(stats, status=200)
        self.log_request_line(200)


def main():
    port = 8080
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if re.match(r"^\d+$", arg):
            port = int(arg)

    server = HTTPServer(("0.0.0.0", port), ArxivHandler)
    print(f"[{now_str()}] Server starting on port {port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print(f"[{now_str()}] Server stopped", flush=True)

if __name__ == "__main__":
    main()
