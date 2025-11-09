"""
pdf_ocr_search_cached.py

- Scans SOURCE_DIR for PDF files.
- OCRs pages in parallel (ProcessPoolExecutor).
- Caches OCR results per-PDF in ./.ocr_cache/<sha256>.json and keeps an index ./.ocr_cache/ocr_index.json
  mapping absolute path -> metadata (mtime, size, sha256, cache_filename).
- On subsequent runs, skips OCR for unchanged files.
- Prompts user for a keyword and prints matches: file -> page -> line with simple highlighting.

Edit SOURCE_DIR, POPPLER_PATH, and TESSERACT_CMD below to suit your environment.
"""

# from __future__ import annotations

import hashlib
import json
import os
import re

# import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from PIL import Image

load_dotenv(override=True)

try:
    from colorama import init as _colorama_init

    _colorama_init()  # enable on Windows
except Exception:
    # colorama not installed -> still try ANSI; Windows may not show colors
    pass

# pdf2image and pytesseract
try:
    from pdf2image import convert_from_path
except Exception:
    print("Missing pdf2image. Install: pip install pdf2image")
    raise

try:
    import pytesseract
except Exception:
    print("Missing pytesseract. Install: pip install pytesseract")
    raise

# ---------------- CONFIG ----------------

# Soft "material-ish" palette (muted, easy-on-eyes) - change here to recolor everything
COLORS = {
    "HEADER": (54, 76, 83),  # muted slate
    "INFO": (120, 144, 156),  # warm gray-blue
    "PROCESSING": (199, 168, 121),  # warm sand
    "FILE": (145, 103, 125),  # muted mauve
    "PAGE": (128, 138, 115),  # soft olive
    "LINE": (96, 125, 139),  # soft blue-gray
    "HIGHLIGHT_FG": (34, 28, 24),  # dark foreground for highlight
    "HIGHLIGHT_BG": (252, 243, 207),  # pale warm background
}

# helpers: rgb ansi + bg + bold
CSI = "\x1b["

# Directory containing PDFs (edit this)
SOURCE_DIR = os.getenv("SOURCE_DIR", None)

# Cache directory (will be created in cwd)
CACHE_DIR = Path.cwd() / ".ocr_cache"  # HC
INDEX_FILE = CACHE_DIR / "ocr_index.json"  # HC

# If Tesseract binary is not on PATH, set full path to tesseract.exe (Windows example)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", None)
# Example: TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# If Poppler location is needed (Windows), set to folder that contains pdfinfo.exe
POPPLER_PATH = os.getenv("POPPLER_PATH", None)
# Example: POPPLER_PATH = r"C:\tools\poppler-23.05.0\Library\bin"

if SOURCE_DIR:
    SOURCE_DIR = str(Path(SOURCE_DIR).resolve())
if POPPLER_PATH:
    POPPLER_PATH = str(Path(POPPLER_PATH).resolve())
if TESSERACT_CMD:
    TESSERACT_CMD = str(Path(TESSERACT_CMD).resolve())

# OCR settings
DPI = 300  # HC: Dots per inch for pdf2image conversion
OCR_LANG = None  # e.g. 'eng' or 'eng+deu'; None => default

# Multiprocessing: number of workers (None => os.cpu_count())
MAX_WORKERS = None
# ----------------------------------------

# Ensure cache dir exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# If Tesseract path provided, we still set it in parent, and workers will set it as well
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_CMD)

# ---------------- Helper functions ----------------


def _ansi_rgb(fg=None, bg=None, bold=False):
    parts = []
    if bold:
        parts.append("1")
    if fg:
        parts.append(f"38;2;{fg[0]};{fg[1]};{fg[2]}")
    if bg:
        parts.append(f"48;2;{bg[0]};{bg[1]};{bg[2]}")
    if not parts:
        return ""
    return CSI + ";".join(parts) + "m"


def colorize(text: str, fg: tuple = None, bg: tuple = None, bold: bool = False) -> str:
    """Wrap text in ANSI 24-bit color + reset. If fg/bg None -> returns text unchanged."""
    start = _ansi_rgb(fg, bg, bold)
    if not start:
        return text
    end = CSI + "0m"
    return f"{start}{text}{end}"


def file_sha256(path: Path, block_size: int = 65536) -> str:
    """Compute SHA256 of file (used for unique cache filenames)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    tmp.replace(path)


def load_index() -> Dict[str, Any]:
    if INDEX_FILE.exists():
        try:
            return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_index(index: Dict[str, Any]) -> None:
    safe_write_json(INDEX_FILE, index)


def list_pdf_files(src_dir: str) -> List[Path]:
    p = Path(src_dir)
    if not p.is_dir():
        raise SystemExit(f"SOURCE_DIR not found or not a directory: {src_dir}")
    return sorted([f.resolve() for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])


def ocr_bytes_worker(args: Tuple[bytes, int, str, str]) -> Tuple[int, str]:
    """
    Worker run in a separate process.
    args: (image_bytes, page_no, tesseract_cmd_or_empty, lang_or_empty)
    Returns (page_no, text)
    """
    image_bytes, page_no, tcmd, lang = args
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd
    try:
        img = Image.open(BytesIO(image_bytes))
        # ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang) if lang else pytesseract.image_to_string(img)
    except Exception as e:
        text = ""
    return (page_no, text)


def ocr_pdf_with_cache(pdf_path: Path, index: Dict[str, Any]) -> Dict[str, Any]:
    """
    OCR a PDF if needed. Returns data dict:
      { "path": str, "sha256": sha, "pages": [text, ...], "mtime": mtime, "size": size }
    Caching logic:
      - If pdf_path exists in index and mtime & size match, load cache file and return.
      - Else perform OCR (pages in parallel), write cache file, update index, and return.
    """
    abs_path = str(pdf_path.resolve())
    st = pdf_path.stat()
    mtime = int(st.st_mtime)
    size = st.st_size

    entry = index.get(abs_path)
    if entry and entry.get("mtime") == mtime and entry.get("size") == size:
        # load cache
        cache_file = CACHE_DIR / entry["cache_file"]
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return data
            except Exception:
                # fall through to reprocess
                pass

    # Need to process
    print(colorize(f"OCR processing: {pdf_path.name} ...", fg=COLORS["PROCESSING"], bold=True))
    sha = file_sha256(pdf_path)
    cache_filename = f"{sha}.json"
    cache_path = CACHE_DIR / cache_filename

    # Convert pages to images (PIL Images)
    try:
        images = convert_from_path(str(pdf_path), dpi=DPI, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(colorize(f"  [ERROR] convert_from_path failed for {pdf_path.name}: {e}", fg=COLORS["PROCESSING"]))
        images = []

    # Prepare bytes list for workers
    tasks = []
    for i, img in enumerate(images, start=1):
        bio = BytesIO()
        # Save as PNG to bytes
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(bio, format="PNG")
        img_bytes = bio.getvalue()
        tasks.append((img_bytes, i, str(TESSERACT_CMD) if TESSERACT_CMD else "", OCR_LANG if OCR_LANG else ""))

    pages_text = []
    if tasks:
        workers = MAX_WORKERS or os.cpu_count() or 2
        # with ProcessPoolExecutor(max_workers=workers) as ex:
        #     futures = [ex.submit(ocr_bytes_worker, t) for t in tasks]
        #     for fut in as_completed(futures):
        #         try:
        #             page_no, text = fut.result()
        #         except Exception as e:
        #             page_no, text = (None, "")
        #         pages_text.append((page_no, text))
        # # sort by page_no
        # pages_text.sort(key=lambda x: x[0])
        # pages_text = [t for (_, t) in pages_text]2
        with ProcessPoolExecutor(max_workers=workers) as ex:
            # use map to preserve order of tasks; each task is a tuple argument
            # map will raise if any worker raises — we handle inside worker so it's unlikely,
            # but we still wrap in try to be defensive.
            try:
                for page_no, text in ex.map(ocr_bytes_worker, tasks):
                    # skip invalid page numbers
                    if page_no is None:
                        continue
                    pages_text.append((page_no, text))
            except Exception as e:
                # Fallback: collect completed futures and continue
                print(f"  [WARN] Executor.map raised: {e}; attempting to collect completed futures...")
                # slower but robust: fall back to as_completed
                futures = [ex.submit(ocr_bytes_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    try:
                        page_no, text = fut.result()
                        if page_no is None:
                            continue
                        pages_text.append((page_no, text))
                    except Exception as e2:
                        print(f"    [ERROR] OCR worker failed: {e2}")

        # ensure pages are sorted by page_no and extract texts
        pages_text.sort(key=lambda x: x[0])
        pages_text = [t for (_, t) in pages_text]

    else:
        pages_text = []

    if pages_text:
        data = {
            "path": abs_path,
            "sha256": sha,
            "mtime": mtime,
            "size": size,
            "cache_file": cache_filename,
            "pages": pages_text,
        }

        # save cache file (atomic)
        safe_write_json(cache_path, data)

        # update index
        index[abs_path] = {
            "mtime": mtime,
            "size": size,
            "sha256": sha,
            "cache_file": cache_filename,
        }
        save_index(index)

    return data


def highlight_line(line: str, keyword_re: re.Pattern) -> str:
    def repl(m):
        return colorize(m.group(0), fg=COLORS["HIGHLIGHT_FG"], bg=COLORS["HIGHLIGHT_BG"], bold=True)

    return keyword_re.sub(repl, line)


def search_in_pages(pages: List[str], keyword_re: re.Pattern) -> List[Tuple[int, int, str]]:
    """
    Search keyword regex in the list of page texts.
    Returns list of (page_no, line_no, line_text) where page_no and line_no are 1-based.
    """
    results = []
    for pno, text in enumerate(pages, start=1):
        for lno, raw in enumerate(text.splitlines(), start=1):
            if keyword_re.search(raw):
                results.append((pno, lno, raw.rstrip()))
    return results


# ----------------- Main flow -----------------
def main():
    print(
        colorize("Source directory:", fg=COLORS["INFO"], bold=True),
        colorize(SOURCE_DIR or "unset", fg=COLORS["HEADER"]),
    )
    src_p = SOURCE_DIR
    pdf_list = list_pdf_files(src_p)
    if not pdf_list:
        print("No PDF files found in SOURCE_DIR. Edit the SOURCE_DIR in the script.")
        return

    # load index (already parsed cached files metadata)
    ocr_index = load_index()

    all_data = {}  # path -> data dict with pages
    for pdf in pdf_list:
        try:
            data = ocr_pdf_with_cache(pdf, ocr_index)
            all_data[str(pdf.resolve())] = data
        except Exception as e:
            print(f"[ERROR] processing {pdf}: {e}")

    # Prompt user for keyword
    kw = input("\nEnter search keyword (case-insensitive): ").strip()
    if not kw:
        print("Empty keyword. Exiting.")
        return
    keyword_re = re.compile(re.escape(kw), re.IGNORECASE)

    # Search through cached pages
    matches_any = False
    print("\n" + colorize("=== SEARCH RESULTS ===", fg=COLORS["HEADER"], bold=True))
    for pdf_path, data in all_data.items():
        pages = data.get("pages", [])
        hits = search_in_pages(pages, keyword_re)
        if hits:
            matches_any = True
            print(
                "\n"
                + colorize(f"File: {Path(pdf_path).name}", fg=COLORS["FILE"], bold=True)
                + "  "
                + colorize(f"(full: {pdf_path})", fg=COLORS["INFO"])
            )
            # group by page
            by_page = {}
            for pno, lno, txt in hits:
                by_page.setdefault(pno, []).append((lno, txt))
            for pno in sorted(by_page):
                print(colorize(f"  Page {pno}:", fg=COLORS["PAGE"], bold=True))
                for lno, txt in by_page[pno]:
                    print(colorize(f"    Line {lno}:", fg=COLORS["LINE"]) + " " + highlight_line(txt, keyword_re))
    if not matches_any:
        print(f"No matches found for '{kw}' in any PDF.")
    else:
        print("\nDone. You can open the PDF and go to the reported page & line.")
    return


if __name__ == "__main__":
    main()
