import streamlit as st
from groq import Groq
import requests
import re
import math
import random
import numpy as np
from bs4 import BeautifulSoup
import PyPDF2
import docx
from io import StringIO
import csv
import json
import html as htmllib
from datetime import datetime
import pytz
import time
import hashlib
import ast
import operator
import traceback
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse, parse_qs, unquote


class Cfg:
    """MozeAI settings in one place."""
    HELPER_MODEL = "openai/gpt-oss-20b"                      # summaries (has its own rate-limit bucket)
    TOKENS = {"low": 1500, "medium": 2500, "high": 4000}     # answer budget per thinking effort
    TPM_BUDGET = 7600          # prompt + answer stay under Groq's free-tier 8K tokens/minute
    PROMPT_CHARS = 19000       # whole prompt, roughly 5K tokens
    RECENT_MESSAGES = 8        # latest messages sent word for word
    RECENT_CHARS = 5200
    MSG_CLIP = 1400            # per message: start and end kept
    SUMMARY_CHARS = 1400       # running summary of older messages
    RECALL_K = 2               # older exchanges brought back by relevance
    RECALL_CHARS = 1300
    RECALL_MIN = 2.5
    DOC_CHARS = 6000           # file excerpts per question
    DOC_CHARS_WHOLE = 8500     # ... for summaries / comparisons of whole files
    DOC_MIN_SCORE = 3.5        # how relevant a file must be to be used without being mentioned
    CHUNK_CHARS = 1100
    FILE_CHARS = 400_000       # text kept per uploaded file
    SEARCH_CHARS = 3200
    TOOLS_COOLDOWN = 600       # seconds to stop offering Groq's built-in tools after they fail
    LESSONS_CHARS = 1500       # approved rules learned from user feedback
    EXAMPLE_MIN = 3.0          # how similar a well-rated past answer must be to be shown as an example
    REASONS = ["Wrong or inaccurate", "Out of date", "Didn't answer my question", "Too long", "Too short",
               "Should have searched the web", "Formatting problems", "Other"]
    UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    PPT_NOUNS = r"(pptx?|power ?points?|slides?|slide ?decks?|presentations?)"
    WORD_NOUNS = r"(word (documents?|docs?|files?)|docx|documents?|docs?)"
    EXCEL_NOUNS = r"(excel( (files?|sheets?|spreadsheets?|workbooks?))?|spreadsheets?|xlsx|worksheets?)"
    WHOLE_DOC = (r"\b(summari[sz]e|summary|overview|main (points?|ideas?|findings?|arguments?|message)|key (points?|takeaways?|findings?|ideas?)"
                 r"|tl;?dr|gist|outline|what (is|are) (it|this|these|the (files?|documents?|papers?|reports?|pdfs?|texts?)) about|review (it|this|the))\b")
    STOP = frozenset("""a an the and or but if then else of to in on at by for with about from into over after before
        is are was were be been being am do does did done have has had having it its this that these those there here
        i me my we our you your he she they them his her their what which who whom whose when where why how
        can could would should will shall may might must not no yes so as than too very just also only
        please tell give show explain describe make let s t""".split())


# ============================================
# DOCUMENT GENERATION FUNCTIONS (from commit 08973a7)
# ============================================

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from docx import Document as WordDocument
from docx.shared import Inches as DocInches, Pt as DocPt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import io
import base64

def create_ppt_from_content(title, content, filename="presentation"):
    """Create a PowerPoint presentation from content - properly split across slides"""
    try:
        prs = Presentation()
        
        # Title slide
        title_slide_layout = prs.slide_layouts[0]
        slide = prs.slides.add_slide(title_slide_layout)
        slide.shapes.title.text = title[:100]
        slide.placeholders[1].text = f"Created by MozeAI\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Content slides layout
        content_slide_layout = prs.slide_layouts[1]
        
        # Split content into slides based on headings or paragraphs
        lines = content.split('\n')
        
        current_slide = None
        current_text_frame = None
        current_title = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line looks like a slide title
            is_title = False
            if len(line) < 60 and (line.endswith(':') or line.isupper() or re.match(r'^\d+\.', line) or line[0].isupper() and len(line) < 40):
                is_title = True
            
            if is_title:
                # Create new slide for this title
                current_slide = prs.slides.add_slide(content_slide_layout)
                clean_title = line.rstrip(':')
                current_slide.shapes.title.text = clean_title[:100]
                content_box = current_slide.placeholders[1]
                current_text_frame = content_box.text_frame
                current_text_frame.text = ""
                current_title = clean_title
            else:
                # If no slide exists yet, create one
                if current_slide is None:
                    current_slide = prs.slides.add_slide(content_slide_layout)
                    current_slide.shapes.title.text = "Information"
                    content_box = current_slide.placeholders[1]
                    current_text_frame = content_box.text_frame
                    current_text_frame.text = ""
                
                # Add as bullet point
                if current_text_frame:
                    p = current_text_frame.add_paragraph()
                    p.text = line[:150]
                    p.font.size = Pt(18)
                    p.level = 0
                    p.space_after = Pt(6)
        
        # If no content slides were created, add a default one
        if len(prs.slides) == 1:
            slide = prs.slides.add_slide(content_slide_layout)
            slide.shapes.title.text = "Content Summary"
            content_box = slide.placeholders[1]
            text_frame = content_box.text_frame
            text_frame.text = content[:500]
        
        ppt_bytes = io.BytesIO()
        prs.save(ppt_bytes)
        ppt_bytes.seek(0)
        return ppt_bytes
    except Exception as e:
        print(f"PPT creation error: {e}")
        return None

def create_word_from_content(title, content, filename="document"):
    """Create a Word document from content"""
    try:
        doc = WordDocument()
        
        title_heading = doc.add_heading(title, 0)
        title_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        doc.add_paragraph(f"Generated by MozeAI on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph()
        
        paragraphs = content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                p = doc.add_paragraph(para.strip())
                p.style.font.size = DocPt(12)
        
        word_bytes = io.BytesIO()
        doc.save(word_bytes)
        word_bytes.seek(0)
        return word_bytes
    except Exception as e:
        return None

def create_real_excel_file(title, data_rows):
    """Create a REAL .xlsx Excel file with proper formatting"""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
        from io import BytesIO
        
        wb = Workbook()
        ws = wb.active
        ws.title = title[:31].replace('/', '_').replace('\\', '_')
        
        # Write data to worksheet
        for row_idx, row in enumerate(data_rows, 1):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                
                # Style header row
                if row_idx == 1:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
        
        # Auto-adjust column widths
        for col in ws.columns:
            max_length = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[col_letter].width = adjusted_width
        
        # Save to bytes
        output = BytesIO()
        wb.save(output)
        output.seek(0)
        return output
        
    except Exception as e:
        print(f"Excel creation error: {e}")
        return None

def create_csv_from_data(title, data_rows):
    """Create a CSV file from data rows - Fallback"""
    try:
        from io import BytesIO
        import csv
        
        output = BytesIO()
        output.write('\ufeff'.encode('utf-8'))
        
        writer = csv.writer(output)
        for row in data_rows:
            writer.writerow(row)
        
        output.seek(0)
        return output
    except Exception as e:
        print(f"CSV creation error: {e}")
        return None

def create_ppt_from_slides(title, slides, subtitle=None):
    """Build a deck from structured content: a title slide, then one slide per {"title", "bullets", "notes"}."""
    try:
        from pptx.enum.shapes import MSO_SHAPE
        accent, ink = RGBColor(0x5A, 0x4F, 0xCF), RGBColor(0x26, 0x26, 0x33)

        def style(frame, size, color, bold=False):
            for para in frame.paragraphs:
                for run in para.runs:
                    run.font.size, run.font.color.rgb, run.font.bold = Pt(size), color, bold

        prs = Presentation()
        first = prs.slides.add_slide(prs.slide_layouts[0])
        first.background.fill.solid()
        first.background.fill.fore_color.rgb = accent
        first.shapes.title.text = str(title)[:120]
        style(first.shapes.title.text_frame, 40, RGBColor(0xFF, 0xFF, 0xFF), True)
        first.placeholders[1].text = str(subtitle or f"Created by MozeAI · {datetime.now().strftime('%B %d, %Y')}")[:200]
        style(first.placeholders[1].text_frame, 20, RGBColor(0xE6, 0xE3, 0xFF))
        for item in slides:
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = str(item.get("title", ""))[:100]
            style(slide.shapes.title.text_frame, 32, accent, True)
            bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.55), Inches(9.0), Inches(0.05))
            bar.fill.solid()
            bar.fill.fore_color.rgb = accent
            bar.line.fill.background()
            frame = slide.placeholders[1].text_frame
            frame.clear()
            frame.word_wrap = True
            bullets = [str(b).rstrip() for b in (item.get("bullets") or []) if str(b).strip()][:7]
            size = 22 if sum(len(b) for b in bullets) < 300 else 19 if sum(len(b) for b in bullets) < 480 else 16
            for j, bullet in enumerate(bullets):
                para = frame.paragraphs[0] if j == 0 else frame.add_paragraph()
                level = 1 if bullet.startswith(("  ", "\t")) or bullet.lstrip().startswith("- ") else 0
                run = para.add_run()
                run.text = re.sub(r"^[\s\-•*]+", "", bullet)[:220]
                run.font.size = Pt(size - 3 * level)
                run.font.color.rgb = ink
                para.level = level
                para.space_after = Pt(8)
            notes = item.get("notes")
            if notes:
                slide.notes_slide.notes_text_frame.text = str(notes)[:2000]
        buf = io.BytesIO()
        prs.save(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"PPT creation error: {e}", flush=True)
        return None

def _add_md_runs(paragraph, text):
    """Add Markdown text to a Word paragraph: **bold**, *italic* and `code` become formatted runs."""
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]|\$\$", "", text)                       # LaTeX delimiters -> plain text
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r"\1 (\2)", text)       # links -> "text (url)"
    for part in re.split(r"(\*\*[^*]+\*\*|__[^_]+__|\*[^*\s][^*]*\*|`[^`]+`)", text):
        if not part:
            continue
        if part[:2] in ("**", "__") and len(part) > 4:
            paragraph.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and len(part) > 2:
            run = paragraph.add_run(part[1:-1])
            run.font.name = "Consolas"
        elif part.startswith("*") and len(part) > 2:
            paragraph.add_run(part[1:-1]).italic = True
        else:
            paragraph.add_run(part)

def create_word_from_markdown(title, md):
    """Turn Markdown (headings, lists, tables, code, bold/italic) into a properly styled Word document."""
    try:
        doc = WordDocument()
        heading = doc.add_heading(str(title)[:200], 0)
        heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta = doc.add_paragraph()
        meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta.add_run(f"Generated by MozeAI on {datetime.now().strftime('%B %d, %Y')}").italic = True
        lines = str(md).splitlines()
        para = []

        def flush():
            if para:
                _add_md_runs(doc.add_paragraph(), " ".join(para))
                para.clear()

        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            s = line.strip()
            if s.startswith("```"):
                flush()
                code = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                run = doc.add_paragraph().add_run("\n".join(code))
                run.font.name = "Consolas"
                run.font.size = DocPt(9.5)
                i += 1
                continue
            if not s or s in ("\\[", "\\]", "$$") or re.fullmatch(r"[-*_]{3,}", s):
                flush()
                i += 1
                continue
            m = re.match(r"(#{1,6})\s+(.*)", s)
            if m:
                flush()
                doc.add_heading(re.sub(r"[*_`]", "", m.group(2)).strip(), min(len(m.group(1)), 3))
                i += 1
                continue
            if s.startswith("|") and s.endswith("|"):
                flush()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                        rows.append(cells)
                    i += 1
                if rows:
                    ncol = max(len(r) for r in rows)
                    table = doc.add_table(rows=len(rows), cols=ncol)
                    table.style = "Table Grid"
                    for ri, row in enumerate(rows):
                        for ci in range(ncol):
                            cell = table.cell(ri, ci)
                            cell.text = ""
                            _add_md_runs(cell.paragraphs[0], row[ci] if ci < len(row) else "")
                            if ri == 0:
                                for run in cell.paragraphs[0].runs:
                                    run.bold = True
                    doc.add_paragraph()
                continue
            m = re.match(r"(\s*)[-*+•]\s+(.*)", line)
            if m:
                flush()
                _add_md_runs(doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet"), m.group(2))
                i += 1
                continue
            m = re.match(r"\s*\d+[.)]\s+(.*)", line)
            if m:
                flush()
                _add_md_runs(doc.add_paragraph(style="List Number"), m.group(1))
                i += 1
                continue
            if s.startswith(">"):
                flush()
                _add_md_runs(doc.add_paragraph(style="Quote"), s.lstrip("> "))
                i += 1
                continue
            para.append(s)
            i += 1
        flush()
        buf = io.BytesIO()
        doc.save(buf)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Word creation error: {e}", flush=True)
        return None

def _cell_value(v):
    """Numbers written as text ("1,250", "3.5") become real numbers in Excel."""
    if isinstance(v, str):
        t = v.strip()
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?|-?\d+(\.\d+)?", t):
            n = float(t.replace(",", ""))
            return int(n) if n.is_integer() and "." not in t else n
    return v

def _excel_formula(expr, letters, row):
    """'Quantity * Unit Price' -> '=E5*F5' for the given row; None if it is not a safe arithmetic formula."""
    f = str(expr).lstrip("=")
    for name in sorted(letters, key=len, reverse=True):
        f = re.sub(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", f"{letters[name]}{row}", f, flags=re.I)
    if not re.fullmatch(r"(?:[A-Z]{1,3}\d+|[\d.\s+\-*/()])+", f) or not re.search(r"[A-Z]{1,3}\d+", f):
        return None
    return "=" + f.replace(" ", "")

def create_excel_from_table(title, columns, rows, sum_columns=(), formulas=None):
    """Write a styled .xlsx: header, data, live formulas for computed columns and a TOTAL row."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        wb = Workbook()
        ws = wb.active
        ws.title = re.sub(r"[\\/*?:\[\]]", "_", str(title))[:31] or "Sheet1"
        cols = [str(c) for c in columns]
        letters = {c: get_column_letter(i + 1) for i, c in enumerate(cols)}
        ws.append(cols)
        formulas = {k: v for k, v in (formulas or {}).items() if k in letters}
        for r, row in enumerate(rows, start=2):
            values = list(row)[:len(cols)] + [None] * max(0, len(cols) - len(row))
            for c, v in enumerate(values):
                name = cols[c]
                if name in formulas:
                    v = _excel_formula(formulas[name], letters, r) or v
                ws.cell(row=r, column=c + 1, value=_cell_value(v))
        last = len(rows) + 1
        sums = [c for c in (sum_columns or []) if c in letters]
        if sums and rows:
            ws.cell(row=last + 1, column=1, value="TOTAL").font = Font(bold=True)
            for c in sums:
                ws.cell(row=last + 1, column=cols.index(c) + 1, value=f"=SUM({letters[c]}2:{letters[c]}{last})").font = Font(bold=True)
        fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        money = re.compile(r"(price|cost|sales|revenue|amount|salary|total|profit|budget|income|expense|fee|value|spend)", re.I)
        for c, name in enumerate(cols, 1):
            width = max([len(name)] + [len(str(ws.cell(row=r, column=c).value or "")) for r in range(2, min(last, 300) + 1)])
            ws.column_dimensions[get_column_letter(c)].width = min(max(10, width + 2), 50)
            if money.search(name):
                for r in range(2, ws.max_row + 1):
                    cell = ws.cell(row=r, column=c)
                    if isinstance(cell.value, (int, float)) or str(cell.value or "").startswith("="):
                        cell.number_format = "#,##0.00"
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(cols))}{last}"
        out = io.BytesIO()
        wb.save(out)
        out.seek(0)
        return out
    except Exception as e:
        print(f"Excel creation error: {e}", flush=True)
        return None

# ============================================
# FILE PROCESSING FUNCTIONS
# ============================================
# Whole files are kept (up to Cfg.FILE_CHARS); for each question only the relevant parts go to the model.

def _num(x):
    return f"{x:,.2f}".rstrip("0").rstrip(".")

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text.strip() + "\n"
            if len(text) > Cfg.FILE_CHARS:
                break
        return text[:Cfg.FILE_CHARS] if text.strip() else "No extractable text in PDF"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    try:
        file.seek(0)
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            t = para.text.strip() if para.text else ""
            if not t:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            if style.startswith("heading") or style == "title":
                text += f"## {t}\n\n"
            elif "list" in style:
                text += f"- {t}\n"
            else:
                text += t + "\n\n"
        for table in doc.tables:
            text += "\n"
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if row_text:
                    text += " | ".join(row_text) + "\n"
        return text[:Cfg.FILE_CHARS] if text.strip() else "No extractable text in document"
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_txt(file):
    try:
        file.seek(0)
        raw = file.read()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
        return content[:Cfg.FILE_CHARS] if content.strip() else "File is empty"
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def extract_text_from_csv(file):
    """The whole table plus a profile (row count, and per column: min/max/mean/sum or most common values)."""
    try:
        file.seek(0)
        raw = file.read()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("latin-1")
        rows = [r for r in csv.reader(StringIO(content)) if any(c.strip() for c in r)]
        if not rows:
            return "CSV file appears empty"
        header, body = rows[0], rows[1:]
        lines = [f"CSV table: {len(body)} data rows x {len(header)} columns", "Column profile:"]
        for j, name in enumerate(header):
            vals = [r[j].strip() for r in body if j < len(r) and r[j].strip()]
            nums = []
            for v in vals:
                try:
                    nums.append(float(v.replace(",", "").replace("$", "").replace("%", "")))
                except ValueError:
                    pass
            if vals and len(nums) >= 0.8 * len(vals):
                lines.append(f"- {name}: numeric, {len(nums)} values, min {_num(min(nums))}, max {_num(max(nums))}, "
                             f"mean {_num(sum(nums) / len(nums))}, sum {_num(sum(nums))}")
            elif vals:
                common = ", ".join(f"{v} ({c})" for v, c in Counter(vals).most_common(3))
                lines.append(f"- {name}: {len(set(vals))} distinct values; most common: {common}")
        lines += ["", "Rows:", " | ".join(header)]
        lines += [f"{i}: " + " | ".join(r) for i, r in enumerate(body, 1)]
        return "\n".join(lines)[:Cfg.FILE_CHARS]
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def extract_text_from_json(file):
    try:
        file.seek(0)
        data = json.loads(file.read().decode("utf-8-sig"))
        head = f"JSON data: {type(data).__name__}"
        if isinstance(data, dict):
            head += f" with keys: {', '.join(list(map(str, data.keys()))[:20])}"
        elif isinstance(data, list):
            head += f" with {len(data)} items"
        return (head + "\n\n" + json.dumps(data, indent=1, ensure_ascii=False))[:Cfg.FILE_CHARS]
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()

    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain" or file_name.endswith('.txt'):
        return extract_text_from_txt(uploaded_file)
    elif file_type == "text/csv" or file_name.endswith('.csv'):
        return extract_text_from_csv(uploaded_file)
    elif file_type == "application/json" or file_name.endswith('.json'):
        return extract_text_from_json(uploaded_file)
    else:
        return f"Unsupported file type: {file_type}"

# ============================================
# CONFIG
# ============================================

TEMPERATURE = 0.6  # gpt-oss reasons best with some sampling; greedy decoding (0) can loop. File JSON uses 0.5, summaries 0.3
MAX_TOKENS = 2500  # room for the model's reasoning plus a full answer; longer answers continue automatically

st.set_page_config(page_title="MozeAI", page_icon="🧠", layout="wide")

# ============================================
# CSS - FIXED CHAT INPUT AT BOTTOM
# ============================================

st.markdown("""
<style>
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: white !important;
        padding: 10px 20px !important;
        z-index: 999 !important;
        border-top: 1px solid #e0e0e0 !important;
    }
    
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# GROQ CLIENT
# ============================================

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def get_current_datetime():
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    return f"""Current Information:
• Date: {now.strftime('%B %d, %Y')}
• Time: {now.strftime('%I:%M %p')}
• Day: {now.strftime('%A')}
• Timezone: Asia/Seoul"""

# ============================================
# LIGHTWEIGHT MEMORY SYSTEM (No torch needed)
# ============================================

# Keyword memory from the evaluated build (ijasc-r1). Kept for the benchmark harness; answers now use
# the recent conversation, a running summary and BM25 recall (see MEMORY below).
class LightweightMemory:
    def __init__(self):
        self.memories = []
        self.keyword_index = defaultdict(list)
    
    def add_memory(self, text, metadata=None):
        if len(text) < 50:
            return
        
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        
        memory = {
            "text": text,
            "keywords": words,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        self.memories.append(memory)
        
        for word in words:
            self.keyword_index[word].append(len(self.memories) - 1)
        
        if len(self.memories) > 50:
            self.memories = self.memories[-50:]
            self._rebuild_index()
    
    def _rebuild_index(self):
        self.keyword_index = defaultdict(list)
        for idx, memory in enumerate(self.memories):
            for word in memory["keywords"]:
                self.keyword_index[word].append(idx)
    
    def retrieve(self, query, top_k=3):
        if not self.memories:
            return []
        
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        
        scored = []
        for idx, memory in enumerate(self.memories):
            matches = len(query_words & memory["keywords"])
            if matches > 0:
                scored.append((memory["text"], matches))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [text for text, score in scored[:top_k]]
    
    def get_context(self, query):
        results = self.retrieve(query)
        if results:
            context = "RELEVANT PAST CONVERSATIONS:\n\n"
            for i, result in enumerate(results):
                context += f"[{i+1}] {result}\n\n"
            return context
        return ""

# Initialize memory once per browser session (survives Streamlit reruns)
if "memory_store" not in st.session_state:
    st.session_state.memory_store = LightweightMemory()
memory = st.session_state.memory_store

def store_memory(text):
    memory.add_memory(text)

def retrieve_memory(query):
    return memory.get_context(query)

# ============================================
# LLM CALLS: streaming, visible thinking, built-in tools, fallbacks
# ============================================

def _extra_body(model, effort="medium"):
    """Groq options for the gpt-oss reasoning models: how hard to think, and send the thinking back."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": effort, "include_reasoning": True}
    return None

class LLMOptions:
    """Options for the model calls made inside a `with` block: thinking effort, tools, quiet mode, ...

    They travel through session state so llm_with_fallback() keeps its simple signature.
    """
    def __init__(self, **opts):
        self.opts = opts

    def __enter__(self):
        self.prev = st.session_state.get("_llm_opts")
        st.session_state["_llm_opts"] = {**(self.prev or {}), **self.opts}
        return self

    def __exit__(self, *exc):
        st.session_state["_llm_opts"] = self.prev
        return False

def _tools_ok():
    return time.time() >= st.session_state.get("_tools_off_until", 0)

def _tools_off(why):
    st.session_state["_tools_off_until"] = time.time() + Cfg.TOOLS_COOLDOWN
    print(f"[LLM] built-in tools paused for {Cfg.TOOLS_COOLDOWN}s: {why[:300]}", flush=True)

def _max_tokens(messages, effort, override=None):
    """Answer budget: the effort's allowance, capped so prompt + answer fit the per-minute token limit."""
    if override:
        return int(override)
    prompt_tokens = sum(len(str(m.get("content", ""))) for m in messages) / 3.5 + 60
    return int(max(1200, min(Cfg.TOKENS.get(effort, 2500), Cfg.TPM_BUDGET - prompt_tokens)))

def _tool_kind(tool):
    """'search', 'code' or 'tool' for a tool event coming back from Groq."""
    name = ""
    for key in ("type", "name"):
        v = tool.get(key) if isinstance(tool, dict) else getattr(tool, key, None)
        if isinstance(v, str):
            name += v.lower() + " "
    fn = tool.get("function") if isinstance(tool, dict) else getattr(tool, "function", None)
    if fn is not None:
        v = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
        if isinstance(v, str):
            name += v.lower()
    if "search" in name or "browser" in name:
        return "search"
    if "python" in name or "code" in name:
        return "code"
    return "tool"

def _find_sources(obj, out, depth=0):
    """Collect (title, url) pairs from anywhere inside Groq's executed-tool data."""
    if obj is None or depth > 8:
        return
    if hasattr(obj, "model_dump"):
        try:
            obj = obj.model_dump()
        except Exception:
            obj = getattr(obj, "__dict__", {})
    if isinstance(obj, str):
        s = obj.strip()
        if s[:1] in ("{", "["):
            try:
                _find_sources(json.loads(s), out, depth + 1)
            except Exception:
                pass
        return
    if isinstance(obj, dict):
        url = obj.get("url") or obj.get("link")
        if isinstance(url, str) and url.startswith("http"):
            out.append((str(obj.get("title") or urlparse(url).netloc).strip(), url))
        for v in obj.values():
            _find_sources(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _find_sources(v, out, depth + 1)
    elif hasattr(obj, "__dict__"):
        _find_sources(vars(obj), out, depth + 1)

def _add_tool_sources(obj):
    found = []
    _find_sources(obj, found)
    if found:
        st.session_state["_tool_sources"] = (st.session_state.get("_tool_sources") or []) + found

def _request(model, messages, effort, tools, opts, stream=False):
    kw = dict(
        model=model,
        temperature=opts.get("temperature", TEMPERATURE),
        max_tokens=_max_tokens(messages, effort, opts.get("max_tokens")),
        messages=messages,
        timeout=90 if tools else 60,
        extra_body=_extra_body(model, effort)
    )
    if stream:
        kw["stream"] = True
    if tools:
        kw["tools"] = tools
    return kw

def _complete(model, messages, effort, tools, opts):
    completion = client.chat.completions.create(**_request(model, messages, effort, tools, opts))
    msg = completion.choices[0].message
    _add_tool_sources(getattr(msg, "executed_tools", None))
    return (msg.content or "").strip()

def _stream_completion(model, messages, ui, effort, tools=None, opts=None):
    """Stream one completion into the chat bubble. Returns Groq's finish_reason ("stop", "length", ...)."""
    stream = client.chat.completions.create(**_request(model, messages, effort, tools, opts or {}, stream=True))
    finish = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        for tool in (getattr(delta, "executed_tools", None) or []):
            ui.tool_used(tool)
        for tool in (getattr(delta, "tool_calls", None) or []):
            ui.tool_used(tool)
        thought = getattr(delta, "reasoning", None)
        if thought:
            ui.think(thought)
        if delta.content:
            ui.write(delta.content)
        if choice.finish_reason:
            finish = choice.finish_reason
    return finish

def _stream_answer(model, messages, ui, effort, tools=None, opts=None):
    """Stream an answer. If it reaches the token limit, ask the model to carry on (up to 2 more times)."""
    ui.begin()
    finish = _stream_completion(model, messages, ui, effort, tools, opts)
    for _ in range(2):
        if finish != "length" or not ui.answer.strip():
            break
        more = messages + [
            {"role": "assistant", "content": ui.answer},
            {"role": "user", "content": "Continue exactly where you stopped, mid-sentence if needed. Do not repeat anything and do not add a preamble."}
        ]
        try:
            finish = _stream_completion(model, more, ui, "low", None, {**(opts or {}), "max_tokens": None})
        except Exception as e:
            print(f"[LLM] {model} continuation failed: {type(e).__name__}: {e}", flush=True)
            break
    if finish == "length" and ui.answer.strip():
        ui.write("\n\n*(This answer is long - say **continue** to get the rest.)*")
    return ui.answer.strip()

def _failed_tool_query(err):
    """gpt-oss sometimes tries to search when no tool was offered; Groq's error then contains the query it wanted."""
    if not re.search(r"tool_use_failed|attempted to call tool|tool call validation", err, re.I):
        return None
    m = re.search(r"""\\?["']query\\?["']\s*:\s*\\?["']([^"'\\]{2,200})""", err)
    return m.group(1) if m else None

def _with_block(messages, block, note):
    """Copy of messages with extra context put in front of the last user message."""
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] = f"{block}\n\n(Note: {note})\n\n{m['content']}"
            break
    return msgs

def llm_with_fallback(messages, max_retries=2):
    models_to_try = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]   # main model, then fallback
    opts = st.session_state.get("_llm_opts") or {}
    ui = None if opts.get("quiet") else st.session_state.get("_stream_ui")  # live chat bubble, if any
    tools = opts.get("tools")
    searched = False

    for model in opts.get("models") or models_to_try:
        for attempt in range(max_retries):
            effort = opts.get("effort", "medium") if attempt == 0 else "low"
            use_tools = tools if (tools and attempt == 0 and _tools_ok()) else None
            if tools and not use_tools and opts.get("fallback_search") and not searched:
                # the built-in search is unavailable: fetch results ourselves instead
                searched = True
                if ui is not None:
                    ui.step("Searching the web…")
                block = web_context(opts["fallback_search"])
                if block:
                    messages = _with_block(messages, block, "The search tool is unavailable right now, so these results were fetched for you. Cite them like [1].")
            st.session_state["_tool_sources"] = []
            try:
                if ui is not None:
                    content = _stream_answer(model, messages, ui, effort, use_tools, opts)
                else:
                    content = _complete(model, messages, effort, use_tools, opts)
                if not content:
                    raise ValueError("empty completion")
                st.session_state.last_model_used = model
                if not opts.get("quiet") and isinstance(st.session_state.get("_turn"), dict):
                    st.session_state["_turn"]["model"] = model
                return content
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                print(f"[LLM] {model} attempt {attempt + 1}{' (with tools)' if use_tools else ''} failed: {err[:600]}", flush=True)
                if use_tools and not re.search(r"timeout|timed out|connection", err, re.I):
                    _tools_off(err)
                wanted = _failed_tool_query(err) if not opts.get("quiet") and st.session_state.get("web_on", True) else None
                if wanted and not searched:
                    searched = True
                    block = web_context(wanted)
                    if block:
                        messages = _with_block(messages, block, "Search results for your query, fetched for you. Answer without calling tools.")
                if ui is not None:
                    ui.reset("Retrying…")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue

    return "AI service temporarily unavailable. Please try again."

def llm(messages):
    return llm_with_fallback(messages)

def llm_quiet(messages, **opts):
    """A model call that is not streamed into the chat (summaries, JSON for files)."""
    with LLMOptions(quiet=True, **opts):
        return llm_with_fallback(messages)

# ============================================
# MATHS FORMATTING + LIVE "THINKING" DISPLAY
# ============================================

def _math_segment(s):
    """Convert LaTeX delimiters in plain (non-code) Markdown to the ones Streamlit renders."""
    keep = []

    def stash(text):
        keep.append(text)
        return f"\x00{len(keep) - 1}\x00"

    def display(m):
        body = "\n".join(line for line in m.group(1).strip().splitlines() if line.strip())
        if not body:
            return m.group(0)
        start = m.string.rfind("\n", 0, m.start()) + 1
        lead = re.match(r"[ \t]*(?:(?:[-*+]|\d+[.)])[ \t]+)?", m.string[start:m.start()]).group(0)
        pad = " " * len(lead)  # keep display maths inside list items
        lines = [pad + "$$"] + [pad + line.strip() for line in body.splitlines()] + [pad + "$$"]
        return stash("\n" + pad + "\n" + "\n".join(lines) + "\n" + pad + "\n")

    s = re.sub(r"\\\[([\s\S]+?)\\\]", display, s)                      # \[ ... \]
    s = re.sub(r"(?<!\\)\$\$([\s\S]+?)\$\$", display, s)                # $$ ... $$
    s = re.sub(r"\\\(([\s\S]+?)\\\)",
               lambda m: stash("$" + re.sub(r"\s*\n\s*", " ", m.group(1).strip()) + "$"), s)  # \( ... \)
    # $...$ that is really maths (no space just inside the $ signs, not followed by a digit) is kept
    s = re.sub(r"(?<![\\$])\$(?![\s$])([^$\n]+?)(?<![\s\\])\$(?![\d$])", lambda m: stash(m.group(0)), s)
    s = re.sub(r"(?<!\\)\$", r"\\$", s)                                   # money: $5 -> \$5
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)

def format_math(text):
    """Make the model's LaTeX show as formulas in Streamlit.

    Streamlit renders $...$ and $$...$$ with KaTeX, but gpt-oss writes \\( \\) and \\[ \\].
    Code blocks and inline code are left untouched, and dollar amounts such as $5 are escaped
    so they are not mistaken for maths.
    """
    if not isinstance(text, str) or not text:
        return text
    parts = re.split(r"(```[\s\S]*?(?:```|$)|`[^`\n]+`)", text)
    return "".join(p if i % 2 else _math_segment(p) for i, p in enumerate(parts))

def _strip_cites(text):
    """Remove the model's internal browsing citation marks such as 【3†L10-L15】."""
    return re.sub(r"【[^】]{0,80}】", "", text) if isinstance(text, str) else text

def _clip(text, limit=1500):
    """Shorten a long chat message for the prompt, keeping its start and its end."""
    text = str(text)
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...]\n" + text[-half:]

def _ui_step(label):
    """Show what the assistant is doing (e.g. searching) in the live status line, if there is one."""
    ui = st.session_state.get("_stream_ui")
    if ui is not None:
        ui.step(label)

class StreamUI:
    """Live view of one answer: a "Thinking…" panel with the model's reasoning and tool use, then the answer as it streams."""

    def __init__(self, container):
        self.think_slot = container.empty()
        self.answer_slot = container.empty()
        self.reset()

    def reset(self, label="Thinking…"):
        """Start (again) from an empty answer, e.g. before a retry or a fallback model."""
        self.thinking = ""
        self.answer = ""
        self.seconds = 0.0
        self.used = []             # tools the model used: "searched the web", "ran Python"
        self._t0 = time.time()
        self._answering = False
        self._last_draw = 0.0
        self._label, self._label_t = label, time.time()
        self.answer_slot.empty()
        self._status = self.think_slot.container().status(label, expanded=True)
        self._think_text = self._status.empty()

    def step(self, label):
        if self._status is not None and not self._answering and label != self._label:
            self._status.update(label=label)
            self._label, self._label_t = label, time.time()

    def begin(self):
        """The model has been asked: time the thinking from here."""
        self._t0 = time.time()
        self.step("Thinking…")

    def tool_used(self, tool):
        kind = _tool_kind(tool)
        note = {"search": "searched the web", "code": "ran Python"}.get(kind, "used a tool")
        if note not in self.used:
            self.used.append(note)
        self.step({"search": "Searching the web…", "code": "Running Python…"}.get(kind, "Using a tool…"))
        _add_tool_sources(tool)

    def think(self, piece):
        self.thinking += piece
        if self._label != "Thinking…" and time.time() - self._label_t > 2:
            self.step("Thinking…")
        self._draw()

    def write(self, piece):
        if not self._answering:
            self._answering = True
            self._close_thinking()
        self.answer += piece
        self._draw()

    def label(self):
        return f"Thought for {max(1, round(self.seconds))}s" + "".join(f" · {u}" for u in self.used)

    def _close_thinking(self):
        self.seconds = time.time() - self._t0
        if self.thinking.strip():
            self._think_text.caption(format_math(_strip_cites(self.thinking)))
            self._status.update(label=self.label(), state="complete", expanded=False)
        else:
            self.think_slot.empty()
            self._status = None

    def _draw(self):
        now = time.time()
        if now - self._last_draw < 0.05:
            return
        self._last_draw = now
        if self._answering:
            self.answer_slot.markdown(format_math(_strip_cites(self.answer)) + " ▌")
        elif self.thinking:
            tail = self.thinking[-700:]
            if len(self.thinking) > 700:
                tail = "…" + tail[tail.find(" ") + 1:]
            self._think_text.caption(format_math(_strip_cites(tail)))

    def finish(self, response):
        """Show the final answer (also used for replies that did not come from the model)."""
        if not self._answering:
            self._answering = True
            self._close_thinking()
        self.answer_slot.markdown(format_math(_strip_cites(response)))

# ============================================
# SYSTEM PROMPT (the rules)
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, a smart, honest and friendly AI assistant. You can search the web, run Python code, read files the user uploads, remember this conversation, and create images, PowerPoint, Word and Excel files.

ABOUT YOU (mention only when asked): you were created by Mukiibi Moses, a Computer Engineering student and graduate researcher in the BCML lab (Bio Computing and Machine Learning lab) at Kwangwoon University, Seoul, South Korea. Portfolio: https://moze12432.github.io/

HOW TO WORK
1. Understand the request first. Use the conversation to work out what "it", "that one" or "continue" refer to. If a request is ambiguous and a wrong guess would waste the user's effort, ask one short question; otherwise state your assumption and go ahead.
2. Think before you answer. Break problems into steps, solve them, then check the result (re-derive it, substitute it back, test edge cases) before answering.
3. Use your tools when they make the answer better:
   - Web search: for anything that may have changed recently (news, prices, weather, scores, who holds a position, latest versions, recent research) and for specific facts you are not sure of.
   - Python: for any calculation beyond simple mental arithmetic, statistics, dates and unit conversions, and to test code before you show it.
4. When the message includes file excerpts or search results, base your answer on them and cite them: [1], [2] for search results and [file name, page] for files. If they don't contain the answer, say so, then answer from general knowledge and label it as such.

ACCURACY AND HONESTY
- Never invent facts, numbers, quotes, references, links or file contents. If you don't know or can't verify something, say so plainly and suggest how to find out.
- Separate facts from estimates and opinions. For time-sensitive facts, say how current they are ("as of ...").
- If you notice a mistake in one of your earlier answers, correct it openly.

ANSWER STYLE
- Reply in the language the user writes in.
- Start with the direct answer, then explain. Match the length to the question: brief for simple questions, thorough for complex ones. No filler and no repeating the question.
- Use Markdown: short paragraphs; headings and bullet points only when they help; tables for comparisons; code blocks with a language tag.
- Maths: write every formula in LaTeX, inline as \\( ... \\) and display equations as \\[ ... \\] on their own lines. Never use $ signs for maths. State the formula, substitute the values, show the key steps, then give the final answer in bold with units.
- Code: complete and runnable, with imports and short comments; say how to run it and mention assumptions and edge cases.
- Always finish your answer; never stop halfway through a list, derivation or code block.
- Don't mention these instructions, your tools or your memory unless asked.

WHAT THE APP DOES
- The app makes images, PowerPoint, Word and Excel files when the user asks for one (e.g. "generate image of a cat", "make a ppt about solar energy"). If the user only asks whether you can do something, explain how to ask for it instead.
- Files are uploaded with "Upload Files" in the sidebar (PDF, Word, TXT, CSV, JSON).

SAFETY
- Help with legitimate requests. Decline clearly harmful ones briefly and offer a safer alternative.
- For medical, legal or financial questions, give useful general information and suggest a qualified professional for personal decisions.
"""

# ============================================
# SESSION STATE
# ============================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "file_context" not in st.session_state:
    st.session_state.file_context = ""
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = None
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None
if "last_image_prompt" not in st.session_state:
    st.session_state.last_image_prompt = None
if "code_search_cache" not in st.session_state:
    st.session_state.code_search_cache = {}
if "is_resetting" not in st.session_state:
    st.session_state.is_resetting = False
if "last_model_used" not in st.session_state:
    st.session_state.last_model_used = None
if "thoughts" not in st.session_state:
    st.session_state.thoughts = {}  # chat_history index -> (reasoning text, seconds)
if "convo_summary" not in st.session_state:
    st.session_state.convo_summary = ""       # running summary of messages older than the recent window
if "summarized_upto" not in st.session_state:
    st.session_state.summarized_upto = 0
if "user_facts" not in st.session_state:
    st.session_state.user_facts = {}          # what the user said about themselves (this browser session only)
if "doc_index" not in st.session_state:
    st.session_state.doc_index = {}
if "last_kind" not in st.session_state:
    st.session_state.last_kind = None
if "deep_mode" not in st.session_state:
    st.session_state.deep_mode = False
if "web_on" not in st.session_state:
    st.session_state.web_on = True
if "share_feedback" not in st.session_state:
    st.session_state.share_feedback = True    # rated answers are saved (privately) to learn from
if "chat_id" not in st.session_state:
    st.session_state.chat_id = uuid.uuid4().hex[:8]
if "answer_meta" not in st.session_state:
    st.session_state.answer_meta = {}         # chat_history index -> what the app did for that answer
if "ab_arm" not in st.session_state:
    st.session_state.ab_arm = None          # A/B test: "lessons" or "control", drawn per chat
if "feedback_state" not in st.session_state:
    st.session_state.feedback_state = {}      # widget key -> {id, rating, sha, reasons_sent}

# Document download session states (from commit 08973a7)
if "show_ppt_download" not in st.session_state:
    st.session_state.show_ppt_download = False
if "ppt_data" not in st.session_state:
    st.session_state.ppt_data = None
if "ppt_topic" not in st.session_state:
    st.session_state.ppt_topic = ""
if "show_word_download" not in st.session_state:
    st.session_state.show_word_download = False
if "word_data" not in st.session_state:
    st.session_state.word_data = None
if "word_topic" not in st.session_state:
    st.session_state.word_topic = ""
if "show_excel_download" not in st.session_state:
    st.session_state.show_excel_download = False
if "excel_data" not in st.session_state:
    st.session_state.excel_data = None
if "excel_topic" not in st.session_state:
    st.session_state.excel_topic = ""
if "show_csv_download" not in st.session_state:
    st.session_state.show_csv_download = False
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "csv_topic" not in st.session_state:
    st.session_state.csv_topic = ""
if "last_document_topic" not in st.session_state:
    st.session_state.last_document_topic = ""
if "last_ppt_topic" not in st.session_state:
    st.session_state.last_ppt_topic = ""
if "last_ppt_content" not in st.session_state:
    st.session_state.last_ppt_content = ""
if "last_excel_topic" not in st.session_state:
    st.session_state.last_excel_topic = ""
if "last_excel_data" not in st.session_state:
    st.session_state.last_excel_data = None

# ============================================
# SEARCH FUNCTIONS
# ============================================

def _html_text(fragment):
    fragment = re.sub(r"</?(b|i|em|strong|span)\b[^>]*>", "", fragment)      # inline tags: no extra space
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())

def _ddg_url(href):
    """DuckDuckGo wraps result links (//duckduckgo.com/l/?uddg=...); return the real address."""
    href = htmllib.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/" in href:
        target = parse_qs(urlparse(href).query).get("uddg", [""])[0]
        return unquote(target) or None
    if "duckduckgo.com" in href:
        return None
    return href if href.startswith("http") else None

def internet_search(query, max_results=5):
    """DuckDuckGo search. Returns numbered results as text and remembers their links for the Sources list."""
    try:
        clean_query = query.strip()
        response = requests.post("https://html.duckduckgo.com/html/", data={"q": clean_query},
                                 headers={"User-Agent": Cfg.UA}, timeout=10)
        if response.status_code != 200:
            return ""
        page = response.text
        titles = [(m.start(), m.group(1), m.group(2)) for m in
                  re.finditer(r'<a\b([^>]*\bclass="result__a"[^>]*)>(.*?)</a>', page, re.S)]
        snippets = [(m.start(), m.group(1)) for m in
                    re.finditer(r'<a\b[^>]*\bclass="result__snippet"[^>]*>(.*?)</a>', page, re.S)]
        results = []
        for k, (pos, attrs, title) in enumerate(titles):
            href = re.search(r'href="([^"]+)"', attrs)
            url = _ddg_url(href.group(1)) if href else None
            if not url:
                continue
            end = titles[k + 1][0] if k + 1 < len(titles) else len(page)
            snippet = next((_html_text(s) for p, s in snippets if pos < p < end), "")
            results.append((_html_text(title) or urlparse(url).netloc, url, snippet))
            if len(results) >= max_results:
                break
        if not results:
            return ""
        known = st.session_state.get("_search_sources") or []
        start = len(known)
        st.session_state["_search_sources"] = known + [(t, u) for t, u, _ in results]
        today = datetime.now(pytz.timezone("Asia/Seoul")).strftime("%B %d, %Y")
        context = f"SEARCH RESULTS for '{clean_query}' (searched {today}):\n"
        for n, (title, url, snippet) in enumerate(results, start + 1):
            context += f"[{n}] {title} - {urlparse(url).netloc}\n    {snippet[:320]}\n"
        return context
    except Exception:
        return ""

def _fetch_page(url, timeout=8):
    """(title, readable text) of a web page, or ("", "") if it can't be read."""
    try:
        r = requests.get(url, headers={"User-Agent": Cfg.UA}, timeout=timeout)
        if r.status_code != 200 or "html" not in r.headers.get("content-type", "text/html"):
            return "", ""
        soup = BeautifulSoup(r.content, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg"]):
            element.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        parts = [p.get_text(" ", strip=True) for p in main.find_all(["h1", "h2", "h3", "p", "li", "td", "pre"])]
        text = "\n".join(p for p in parts if len(p) > 25) or main.get_text(" ", strip=True)
        return title, text[:80000]
    except Exception:
        return "", ""

def _best_passages(query, text, limit=900):
    """The passages of a long text that best match the query, in their original order."""
    pieces, buf = [], ""
    for line in re.split(r"(?<=[.!?])\s+|\n+", text):
        if len(buf) + len(line) > 450 and buf:
            pieces.append(buf)
            buf = ""
        buf += (" " if buf else "") + line
    if buf:
        pieces.append(buf)
    if not pieces:
        return ""
    scores = BM25(pieces).scores(query)
    chosen, used = [], 0
    for i in sorted(range(len(pieces)), key=lambda i: -scores[i]):
        if scores[i] <= 0 or used + len(pieces[i]) > limit:
            break
        chosen.append(i)
        used += len(pieces[i])
    return "\n…\n".join(pieces[i] for i in sorted(chosen))

def web_context(query):
    """Search results for the model, plus the most relevant passages from the top two pages."""
    before = len(st.session_state.get("_search_sources") or [])
    text = internet_search(query)
    if not text:
        return ""
    urls = [u for _, u in (st.session_state.get("_search_sources") or [])[before:before + 2]]
    if urls:
        with ThreadPoolExecutor(max_workers=len(urls)) as pool:
            pages = list(pool.map(_fetch_page, urls))
        extracts = []
        for (title, body), url in zip(pages, urls):
            best = _best_passages(query, body) if body else ""
            if best:
                extracts.append(f"From {urlparse(url).netloc}:\n{best}")
        if extracts:
            text += "\nPAGE EXTRACTS:\n" + "\n\n".join(extracts)
    return text[:Cfg.SEARCH_CHARS]

# ============================================
# CALCULATOR
# ============================================

def calculator(query):
    """Answer plain arithmetic such as '12*(3+4)' or 'calculate 2^10' directly.

    Anything else (equations, formulas, word problems) returns None so the model works it out step by step.
    """
    expr = query.lower().strip()
    expr = re.sub(r"^(please\s+)?(calculate|compute|evaluate|what is|what's)\s*", "", expr)
    expr = expr.rstrip(" ?=.!").replace(",", "")
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", expr)
    if not expr or not re.fullmatch(r"[0-9.\s+\-*/()%]+", expr) or not re.search(r"\d\s*[-+*/%]", expr):
        return None
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
           ast.USub: operator.neg, ast.UAdd: operator.pos}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and (abs(b) > 100 or abs(a) > 1e6):
                raise ValueError("number too large")
            return ops[type(node.op)](a, b)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.operand))
        raise ValueError("not plain arithmetic")

    try:
        result = ev(ast.parse(expr, mode="eval").body)
    except Exception:
        return None
    if isinstance(result, float):
        result = round(result, 10)
        if result.is_integer():
            result = int(result)
    return f"{result:,}" if isinstance(result, int) and abs(result) >= 10000 else str(result)

# ============================================
# WEB SCRAPING
# ============================================

def extract_urls_from_query(query):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, query)

# ============================================
# TEXT RANKING (memory recall, file passages, web passages)
# ============================================

def _terms(text):
    """Content words, lower-cased, lightly stemmed (other scripts such as Korean are kept as words)."""
    out = []
    for w in re.findall(r"\w+", str(text).lower()):
        if len(w) < 2 or w in Cfg.STOP:
            continue
        if w.isascii() and len(w) > 4:
            if w.endswith("ies"):
                w = w[:-3] + "y"
            elif w.endswith("ing") and len(w) > 6:
                w = w[:-3]
            elif w.endswith("ed") and len(w) > 5:
                w = w[:-2]
            elif w.endswith("es") and len(w) > 5:
                w = w[:-2]
            elif w.endswith("s") and not w.endswith("ss"):
                w = w[:-1]
        out.append(w)
    return out

class BM25:
    """Small BM25 ranker (no extra libraries)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(_terms(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0
        self.df = Counter(w for d in self.docs for w in d)
        self.n = len(self.docs)
        self.k1, self.b = k1, b

    def scores(self, query):
        q = set(_terms(query))
        out = []
        for d, length in zip(self.docs, self.lens):
            s = 0.0
            for w in q:
                f = d.get(w)
                if f:
                    idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
            out.append(s)
        return out

# ============================================
# UNDERSTANDING THE REQUEST
# ============================================

def calculator_ok(query):
    return calculator(query) is not None

def _is_image_request(q):
    if re.match(r"(how|what|why|which|where|when|is|are|does|do|should|tips)\b", q):
        return False
    if re.fullmatch(r"(can|could|do|are|will|would) you (be able to )?(generate|create|make|draw|produce|design)( an?| any)?"
                    r"( images?| pictures?| photos?| drawings?| art)?( for me)?\??", q):
        return False   # a question about the capability, not a request
    noun = r"(image|picture|pic|photo|photograph|drawing|illustration|painting|artwork|wallpaper|logo|icon|portrait|sketch|poster|cartoon)s?"
    if re.search(rf"\b(generate|create|make|draw|paint|render|design|produce|show me|give me)\b[^.?!]{{0,40}}?\b{noun}\b", q):
        return True
    if re.match(r"(please |can you |could you )?(draw|paint|sketch|illustrate)\b", q) and not re.search(
            r"\b(conclusions?|graphs?|charts?|plots?|diagrams?|lines?|attention|inspiration|comparisons?|parallels?|distinctions?|up)\b", q):
        return True
    return re.match(rf"(an? |the )?{noun} of\b", q) is not None

def _is_image_edit(q):
    if not st.session_state.get("last_image_prompt"):
        return False
    if re.search(r"\b(edit|modify|change|update) (the |that |this )?(image|picture|photo)\b", q):
        return True
    history = st.session_state.get("chat_history") or []
    previous = next((m for r, m in reversed(history) if r == "assistant"), "")
    if "![Generated Image]" not in str(previous) or len(q.split()) > 20:
        return False
    if re.match(r"(how|what|why|who|which|when|where|is|are|does|do|can you explain|tell me)\b", q):
        return False
    return re.match(r"(now |ok(ay)? |but |and |please |can you |could you )*(make|change|turn|add|remove|put|give|replace|use|without|with|more|less|same|instead|zoom|crop|brighter|darker|bigger|smaller|in the style|style)\b", q) is not None

def route(query):
    """What kind of request is this? Decides which handler runs and how the model is set up.

    Returns: chat, datetime, calculator, generate_image, edit_image, ppt, excel, word, scrape_url,
    compare_files, file_task, coding, math, search or reason.
    """
    q = " ".join(query.lower().split())
    files = st.session_state.get("uploaded_files") or {}
    asking = re.match(r"(how|what|why|when|where|which|who|should|is|are|does|do|tips)\b", q) is not None
    make = r"\b(make|create|generate|build|prepare|produce|design|draft|give me|i need|turn (this|it|that) into)\b"

    if len(q.split()) <= 6 and re.fullmatch(
            r"(hi+|hello|hey|yo|hiya|good (morning|afternoon|evening|night)|thanks?( a lot| you( so much| very much)?)?|thank u|ty|"
            r"ok(ay)?|cool|nice|great|awesome|perfect|bye|goodbye|see you|how are you( doing| today)?|what'?s up|sup|lol|haha|wow|sure|"
            r"yes|no|nope|yep)[ !.?,]*(mozeai|moze)?[ !.?]*", q):
        return "chat"
    if len(q.split()) <= 8 and re.search(r"\b(what('s| is) (the )?(time|date|day)( today| now| it)?|what time is it|today'?s date|what day is (it|today))\b", q):
        return "datetime"
    if calculator_ok(query):
        return "calculator"
    if _is_image_request(q):
        return "generate_image"
    if _is_image_edit(q):
        return "edit_image"
    if not asking and re.search(make, q) and re.search(rf"\b{Cfg.PPT_NOUNS}\b", q):
        return "ppt"
    if not asking and re.search(make, q) and re.search(rf"\b{Cfg.EXCEL_NOUNS}\b", q) and not re.search(r"\b(to excel|excel (in|at))\b", q):
        return "excel"
    if not asking and re.search(make, q) and re.search(r"\b(word (document|doc|file)s?|docx|(document|doc)s? (about|on|for|explaining|covering))\b", q):
        return "word"
    if extract_urls_from_query(query):
        return "scrape_url"
    fileish = (r"\b(files?|documents?|pdfs?|uploads?|uploaded|attachments?|attached|"
               r"(the|this|my) (paper|report|article|text|essay|csv|table|data ?set|data|spreadsheet|json|sheet|chapter|book|contract|cv|resume|slides|notes))\b")
    if re.search(r"\b(compare|comparison|contrast|differences?|similarit(y|ies)|versus|vs\.?)\b", q) and (
            len(files) >= 2 or re.search(r"\b(files|documents|both)\b", q)):
        return "compare_files"
    if re.search(fileish, q) or (files and re.search(
            r"\b(summari[sz]e|summary|key points|main points|tl;?dr|according to|(does|did) (it|this|the author) (say|mention|state|claim|cover))\b", q)):
        return "file_task"
    code_words = (r"\b(code|coding|program|script|function|class|method|algorithm|regex|sql|api|bug|debug|compile|refactor|unit tests?|"
                  r"python|javascript|typescript|java|c\+\+|c#|golang|rust|html|css|react|node(js)?|django|flask|pandas|numpy|pytorch|"
                  r"tensorflow|git|docker|bash|linux|streamlit)\b")
    if "```" in query or re.search(r"\b(traceback|exception|error:|stack trace|syntaxerror|typeerror|valueerror|keyerror|indexerror|nullpointer)\b", q) or (
            re.search(code_words, q) and re.search(r"\b(write|fix|debug|implement|build|create|make|convert|optimi[sz]e|refactor|explain|why|how|error|bug|code|function|script|program)\b", q)):
        return "coding"
    if re.search(r"\\\(|\\\[", query) or re.search(
            r"\b(solve|equation|integral|integrate|derivative|differentiate|limit of|prove|proof|probability|expected value|variance|"
            r"standard deviation|matrix|matrices|eigen\w*|vectors?|calculate|compute|evaluate|simplify|factori[sz]e|percent(age)?|"
            r"compound interest|convert \d)\b", q) or re.search(r"\d\s*[\^*/+=-]\s*[\dx(]|\b[a-z]\s*\^\s*\d", q):
        return "math"
    if re.search(r"\b(today|tonight|tomorrow|yesterday|right now|current(ly)?|latest|newest|recent(ly)?|this (week|month|year)|upcoming|live (scores?|streams?|results?|updates?|coverage)|"
                 r"breaking|news|headlines?|weather|forecast|temperature|prices?|cost of|stocks?|share price|exchange rate|scores?|"
                 r"standings|election|released?|launch(ed)?|20[2-3]\d)\b", q) or re.search(
                 r"^(who|whom|whose|where|when)\b|\b(who (is|was|are|were)|tell me about|what happened|population of|capital of|ceo of|"
                 r"president of|prime minister|founder of|owner of|net worth|how (old|tall|far|big|many people)|tallest|largest|"
                 r"biggest|smallest|longest|richest|highest|fastest|oldest)\b", q):
        return "search"
    return "reason"

def choose_effort(kind, query):
    """How hard the model should think: low, medium or high."""
    q = query.lower()
    if kind in ("chat", "datetime", "calculator"):
        return "low"
    if st.session_state.get("deep_mode") or re.search(r"\b(think (hard|carefully|deeply)|in depth|in-depth|deep dive|rigorous)\b", q):
        return "high"
    if re.search(r"\b(quick(ly)?|brief(ly)?|short answer|in one (line|sentence)|tl;?dr)\b", q):
        return "low"
    if kind == "math" or len(query) > 700 or re.search(
            r"\b(prove|proof|derive|step[- ]by[- ]step|optimi[sz]e|algorithm|complexity|architecture|design an?|trade-?offs?|analy[sz]e|"
            r"why does|why do|why is|evaluate)\b", q):
        return "high"
    if kind == "coding":
        return "high" if ("```" in query or re.search(r"\b(error|bug|fix|debug|traceback|exception|why|not working|fails?)\b", q)) else "medium"
    if kind == "search":
        return "low" if len(q.split()) <= 12 else "medium"
    return "medium"

def _web_allowed():
    return st.session_state.get("web_on", True)

def _tools_for(kind, has_docs=False):
    """Groq built-in tools to offer: web search and/or Python, depending on the request."""
    if kind in ("chat", "continue", "datetime", "calculator") or not _tools_ok():
        return None
    web = _web_allowed()
    if kind in ("file_task", "compare_files") and has_docs:
        names = ["code_interpreter"]
    elif kind == "search":
        names = ["browser_search"] if web else []
    else:
        names = (["browser_search"] if web else []) + ["code_interpreter"]
    return [{"type": n} for n in names] or None

def _has_tool(tools, name):
    return any(t.get("type") == name for t in (tools or []))

# ============================================
# MEMORY: recent messages, running summary, recall, facts about the user
# ============================================

def _plain(msg):
    """A chat message as context: images and source lists shortened."""
    msg = re.sub(r"!\[[^\]]*\]\([^)]*\)", "[image]", str(msg))
    msg = re.split(r"\n+---\n\*\*Sources\*\*", msg)[0]
    return _strip_cites(msg).strip()

def _history_for_prompt():
    """(recent messages as chat turns, older messages) - the current question is not included."""
    history = list(st.session_state.get("chat_history") or [])
    current = st.session_state.get("_current_query")
    if history and history[-1][0] == "user" and history[-1][1] == current:
        history = history[:-1]
    recent, used = [], 0
    for role, msg in reversed(history[-Cfg.RECENT_MESSAGES:]):
        text = _clip(_plain(msg), Cfg.MSG_CLIP)
        if recent and used + len(text) > Cfg.RECENT_CHARS:
            break
        recent.append({"role": "assistant" if role == "assistant" else "user", "content": text})
        used += len(text)
    recent.reverse()
    return recent, history[:len(history) - len(recent)]

def _recall(question, older):
    """Earlier exchanges (outside the recent window) that are relevant to this question."""
    pairs = [(older[i][1], older[i + 1][1]) for i in range(len(older) - 1)
             if older[i][0] == "user" and older[i + 1][0] == "assistant"]
    if not pairs:
        return ""
    scores = BM25([u + " " + _plain(a) for u, a in pairs]).scores(question)
    found = []
    for i in sorted(range(len(pairs)), key=lambda i: -scores[i])[:Cfg.RECALL_K]:
        if scores[i] < Cfg.RECALL_MIN:
            break
        found.append(f"User: {_clip(pairs[i][0], 300)}\nAssistant: {_clip(_plain(pairs[i][1]), 560)}")
    return "\n\n".join(found)[:Cfg.RECALL_CHARS]

def _memory_note(question, older):
    parts = []
    facts = st.session_state.get("user_facts") or {}
    if facts:
        parts.append("What the user has told you about themselves:\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()))
    summary = st.session_state.get("convo_summary") or ""
    if summary and older:
        parts.append("Summary of the earlier part of this conversation:\n" + summary)
    recalled = _recall(question, older)
    if recalled:
        parts.append("Earlier exchanges that may be relevant:\n" + recalled)
    return "\n\n".join(parts)

def note_user_facts(text):
    """Remember simple facts the user states about themselves (name, job, location, preferences) for this session."""
    facts = dict(st.session_state.get("user_facts") or {})
    patterns = [
        ("Name", r"(?i:\bmy name is\s+)([A-Za-z][\w'-]+(?:\s+[A-Z][\w'-]+){0,2})"),
        ("Likes to be called", r"(?i:\bcall me\s+)([A-Za-z][\w'-]+)"),
        ("Occupation", r"(?i:\bi(?:\s+am|'m)\s+(?:a|an)\s+)((?:[\w-]+\s+){0,3}?(?i:student|researcher|engineer|developer|programmer|teacher|"
                       r"lecturer|professor|doctor|nurse|designer|manager|scientist|analyst|writer|lawyer|founder|entrepreneur|accountant|"
                       r"consultant|marketer|architect|data scientist))\b"),
        ("Lives in", r"(?i:\bi\s+(?:live|am based|'m based)\s+in\s+)([A-Z][\w'-]+(?:[\s,]+[A-Z][\w'-]+){0,3})"),
        ("From", r"(?i:\bi(?:\s+am|'m)\s+from\s+)([A-Z][\w'-]+(?:[\s,]+[A-Z][\w'-]+){0,3})"),
        ("Works or studies at", r"(?i:\bi\s+(?:work|study|studied)\s+(?:at|for)\s+)([A-Z][\w&'.-]+(?:\s+[A-Z][\w&'.-]+){0,5})"),
        ("Preference", r"(?i:\b(?:please\s+)?(?:always|from now on,?)\s+)((?i:answer|reply|respond|write|use|explain|keep|give|call)[^.!?\n]{3,80})"),
    ]
    for key, pattern in patterns:
        m = re.search(pattern, text)
        if m:
            value = m.group(1).strip(" .,")
            if value.lower() not in Cfg.STOP and value.lower() not in ("not", "none", "unknown", "secret"):
                facts[key] = value
    if facts != (st.session_state.get("user_facts") or {}):
        st.session_state.user_facts = dict(list(facts.items())[-10:])

def update_conversation_memory():
    """After an answer: fold messages that left the recent window into the running summary."""
    history = st.session_state.get("chat_history") or []
    done = st.session_state.get("summarized_upto", 0)
    cut = len(history) - Cfg.RECENT_MESSAGES
    if cut - done < 4:
        return
    summary = st.session_state.get("convo_summary") or ""
    transcript = "\n".join(f"{'User' if r == 'user' else 'Assistant'}: {_clip(_plain(m), 700)}" for r, m in history[done:cut])
    messages = [
        {"role": "system", "content": "You keep a compact running summary of a conversation between a user and the assistant MozeAI. "
                                      "Keep the topics, the user's questions and goals, key facts, numbers, decisions and conclusions, "
                                      "files or images that were made, and open questions. Drop small talk. Use short bullet points, "
                                      "at most 1,200 characters in total. Output only the updated summary."},
        {"role": "user", "content": f"CURRENT SUMMARY:\n{summary or '(empty)'}\n\nNEW MESSAGES:\n{transcript}\n\nUpdated summary:"}
    ]
    out = llm_quiet(messages, effort="low", models=[Cfg.HELPER_MODEL], max_tokens=900, temperature=0.3)
    if out and not out.startswith("AI service temporarily unavailable"):
        summary = out.strip()
    else:   # no model available: keep at least a list of what was asked
        lines = summary.splitlines() + [f"- User asked: {_clip(m, 150)}" for r, m in history[done:cut] if r == "user"]
        summary = "\n".join(lines)
    st.session_state.convo_summary = summary[-Cfg.SUMMARY_CHARS:]
    st.session_state.summarized_upto = cut

# ============================================
# UPLOADED FILES: split into sections, pick the relevant ones per question
# ============================================

def _chunk_document(name, text):
    """Split a file's text into sections of about Cfg.CHUNK_CHARS, keeping PDF page numbers."""
    chunks, page, buf = [], None, ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append({"file": name, "page": page, "text": buf.strip(), "n": len(chunks)})
        buf = ""

    size = Cfg.CHUNK_CHARS
    for block in re.split(r"(\n--- Page \d+ ---\n)", str(text)):
        m = re.fullmatch(r"\n--- Page (\d+) ---\n", block)
        if m:
            flush()
            page = int(m.group(1))
            continue
        for para in re.split(r"\n\s*\n", block):
            para = para.strip()
            while len(para) > size:
                cut = max(para.rfind("\n", 0, size), para.rfind(". ", 0, size) + 1)
                cut = cut if cut > size // 2 else size
                flush()
                buf = para[:cut]
                flush()
                para = para[cut:].strip()
            if not para:
                continue
            if len(buf) + len(para) + 1 > size:
                flush()
            buf += ("\n" if buf else "") + para
    flush()
    return chunks

def _doc_index():
    files = st.session_state.get("uploaded_files") or {}
    key = hashlib.md5("|".join(f"{n}:{len(c)}" for n, c in files.items()).encode()).hexdigest() if files else None
    index = st.session_state.get("doc_index") or {}
    if index.get("key") != key:
        chunks = [c for name, text in files.items() for c in _chunk_document(name, text)]
        index = {"key": key, "chunks": chunks, "bm25": BM25([f"{c['file']} {c['text']}" for c in chunks])}
        st.session_state.doc_index = index
    return index

def select_doc_context(query, kind):
    """Excerpts from the uploaded files that matter for this question ('' if the files are not relevant)."""
    index = _doc_index()
    chunks = index.get("chunks") or []
    if not chunks:
        return ""
    scores = index["bm25"].scores(query)
    best = max(scores, default=0)
    whole = re.search(Cfg.WHOLE_DOC, query.lower()) is not None
    file_kind = kind in ("file_task", "compare_files")
    if not file_kind and not whole and best < Cfg.DOC_MIN_SCORE:
        return ""
    names = list(dict.fromkeys(c["file"] for c in chunks))
    by_file = {n: [i for i, c in enumerate(chunks) if c["file"] == n] for n in names}
    picked, budget = set(), Cfg.DOC_CHARS

    def take(i):
        nonlocal budget
        size = len(chunks[i]["text"]) + 40
        if i in picked or size > budget:
            return False
        picked.add(i)
        budget -= size
        return True

    shorten = {}                                               # section -> chars to show (whole-file overviews)
    pages = {int(p) for p in re.findall(r"\bpages?\s+(\d{1,4})\b", query.lower())}
    for i, c in enumerate(chunks):                             # "what does page 7 say" -> page 7 first
        if c["page"] in pages:
            take(i)
    if sum(len(c["text"]) + 40 for c in chunks) <= Cfg.DOC_CHARS:
        for i in range(len(chunks)):
            take(i)
    elif whole or kind == "compare_files" or best < 1.0:
        budget = Cfg.DOC_CHARS_WHOLE - (Cfg.DOC_CHARS - budget)
        share = budget // len(names)
        for n in names:
            ids = [i for i in by_file[n] if i not in picked]
            if not ids:
                continue
            if whole or max(scores[i] for i in ids) < 1.0:     # an overview: the start of sections spread over the file
                k = min(len(ids), 18)
                step = len(ids) / k
                chosen = [ids[int(j * step)] for j in range(k)]
                for i in chosen:
                    shorten[i] = max(300, share // k - 40)
            else:                                              # its start plus the best matches
                chosen = [ids[0]] + sorted(ids, key=lambda i: -scores[i])[:max(1, share // (Cfg.CHUNK_CHARS + 40)) - 1]
            for i in chosen:
                size = min(len(chunks[i]["text"]), shorten.get(i, 10 ** 9)) + 40
                if i not in picked and size <= budget:
                    picked.add(i)
                    budget -= size
    else:
        ranked = sorted(range(len(chunks)), key=lambda i: -scores[i])
        take(by_file[chunks[ranked[0]]["file"]][0])            # start of the most relevant file, for orientation
        for i in ranked:
            if scores[i] <= 0 or budget < 300:
                break
            take(i)
    order = sorted(picked, key=lambda i: (names.index(chunks[i]["file"]), chunks[i]["n"]))
    head = ("FILE EXCERPTS - the user has uploaded: " + ", ".join(f"{n} ({len(by_file[n])} sections)" for n in names)
            + f". Shown: {len(order)} of {len(chunks)} sections, chosen for this question. Cite them as [file name, page].")
    def text_of(i):
        t = chunks[i]["text"]
        return t if i not in shorten or len(t) <= shorten[i] else t[:shorten[i]].rsplit(" ", 1)[0] + " …"

    body = "\n\n".join(f"[{chunks[i]['file']}" + (f", page {chunks[i]['page']}" if chunks[i]["page"] else "")
                       + f"]\n{text_of(i)}" for i in order)
    return head + "\n\n" + body

# ============================================
# BUILDING THE PROMPT AND ANSWERING
# ============================================

def _now_text():
    now = datetime.now(pytz.timezone("Asia/Seoul"))
    return f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}, {now.strftime('%I:%M %p').lstrip('0')} (Seoul time)"

def build_messages(question, doc_block="", search_block="", extra_context="", note="", examples=""):
    """System rules + lessons + memory, the recent conversation as real chat turns, then this message with its context."""
    recent, older = _history_for_prompt()
    system = SYSTEM_PROMPT + f"\nCurrent date and time: {_now_text()}."
    lessons = _learned().get("rules") if _use_lessons() else ""
    if lessons:
        system += "\n\nLESSONS FROM USER FEEDBACK (follow these too):\n" + lessons
    if examples:
        system += ("\n\nANSWERS USERS RATED HIGHLY FOR SIMILAR QUESTIONS (match their approach, depth and format; "
                   "don't copy facts unless they are still correct for this question):\n" + examples)
    memory_note = _memory_note(question, older)
    if memory_note:
        system += "\n\nMEMORY (use it when relevant; don't mention it):\n" + memory_note
    blocks = [b for b in (doc_block, search_block, extra_context) if b]
    if note.strip():
        blocks.append(f"(Note: {note.strip()})")
    user = "\n\n".join(blocks + [f"USER MESSAGE:\n{question}"]) if blocks else question
    messages = [{"role": "system", "content": system}] + recent + [{"role": "user", "content": user}]
    while len(messages) > 2 and sum(len(m["content"]) for m in messages) > Cfg.PROMPT_CHARS:
        messages.pop(1)                                   # drop the oldest recent message first
    over = sum(len(m["content"]) for m in messages) - Cfg.PROMPT_CHARS
    if over > 0:                                          # still too long: shorten the context blocks
        messages[-1]["content"] = _clip(messages[-1]["content"], max(2000, len(messages[-1]["content"]) - over))
    return messages

def _sources_text(text):
    sources, seen = [], set()
    for title, url in (st.session_state.get("_search_sources") or []) + (st.session_state.get("_tool_sources") or []):
        if url not in seen:
            seen.add(url)
            sources.append((re.sub(r"[\[\]\n]", " ", title).strip()[:90] or urlparse(url).netloc, url))
    if not sources or "**Sources**" in text:
        return ""
    return "\n\n---\n**Sources**\n" + "\n".join(f"{i}. [{t}]({u})" for i, (t, u) in enumerate(sources[:6], 1))

def clean_answer(text):
    # gpt-oss returns its reasoning separately, so the answer is kept whole; only browsing marks are removed
    return _strip_cites(text).strip()

def answer_question(query, kind="reason", extra_context="", note="", effort=None, tools="auto"):
    """The main path: gather the right context, choose effort and tools, and let the model answer (streamed)."""
    files = st.session_state.get("uploaded_files") or {}
    doc_block = ""
    if files and kind not in ("chat", "continue"):
        doc_block = select_doc_context(query, kind)
        if doc_block:
            _ui_step("Reading your files…")
    elif kind in ("file_task", "compare_files"):
        note += (" The user refers to a file, but none is uploaded. Say so in one sentence and explain that PDF, Word, TXT, CSV "
                 "or JSON files can be added with 'Upload Files' in the sidebar; then help as far as you can.")
    effort = effort or _learned_effort(kind, query) or choose_effort(kind, query)
    tool_list = _tools_for(kind, bool(doc_block)) if tools == "auto" else tools
    example_ids, examples = _feedback_examples(query, kind)
    _turn_note(effort=effort, tools_offered=[t["type"] for t in (tool_list or [])], examples=example_ids,
               learned_version=_learned().get("version", ""), files_used=bool(doc_block))
    web_wanted = kind == "search" and _web_allowed()
    search_block = ""
    if web_wanted and not _has_tool(tool_list, "browser_search"):
        _ui_step("Searching the web…")
        search_block = web_context(query)
    if web_wanted:
        note += (" This needs up-to-date information: " + ("search the web first, then " if _has_tool(tool_list, "browser_search") else "")
                 + "base the answer on the search results, cite them, and say how current the information is.")
    if not tool_list and kind in ("search", "reason", "coding", "math", "file_task", "compare_files"):
        note += " No tools are available for this reply, so don't try to search or run code."
    messages = build_messages(query, doc_block, search_block, extra_context, note, examples)
    with LLMOptions(effort=effort, tools=tool_list, fallback_search=query if web_wanted else None):
        out = llm(messages)
    out = clean_answer(out)
    if out.startswith("AI service temporarily unavailable"):
        return out
    return out + _sources_text(out)

def reason(question, context=""):
    """General answer (kept for older call sites and the benchmark harness)."""
    return answer_question(question, "reason", extra_context=context)

def _is_bare_follow_up(q):
    follow_up_re = r"\b(tell me more|more about|continue|go on|keep going|elaborate)\b"
    if not re.search(follow_up_re, q):
        return False
    filler = {"please", "pls", "it", "that", "this", "on", "about", "and", "the", "a", "more", "from", "where",
              "you", "left", "off", "stopped", "can", "could", "further", "with", "answer", "your", "previous",
              "last", "bit", "little", "detail", "details", "in", "explanation", "go", "ahead"}
    return not [w for w in re.findall(r"[a-z0-9]+", re.sub(follow_up_re, " ", q)) if w not in filler]

def continue_answer(query, q):
    previous = _plain(st.session_state.get("last_response") or "")
    if re.search(r"\b(continue|go on|keep going)\b", q):
        note = ("If your previous answer was cut off, continue from exactly where it stopped without repeating anything. "
                "If it was complete, take it further with the next logical step.")
    else:
        note = "Expand on your previous answer with more depth, detail and examples, without repeating what was already said."
    effort = choose_effort(st.session_state.get("last_kind") or "reason", query)
    return answer_question(query, "continue", extra_context=f"YOUR PREVIOUS ANSWER ENDED WITH:\n{previous[-2500:]}",
                           note=note, effort=effort)

# ============================================
# FILE MAKERS (PowerPoint, Word, Excel) AND IMAGES
# ============================================

def _parse_json(text):
    """The first JSON object in a model reply (tolerates code fences and trailing commas)."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    for candidate in (t[a:b + 1], re.sub(r",\s*([}\]])", r"\1", t[a:b + 1])):
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except Exception:
            pass
    return None

def _topic_from(query, nouns):
    """'make a ppt about solar energy for kids' -> 'solar energy for kids' ('' if the topic must come from the chat)."""
    m = re.search(rf"{nouns}\b.*?\b(?:about|on|for|regarding|covering|explaining|of|titled|called)\s+(.+)$", query, re.I)
    if m:
        topic = m.group(m.lastindex)
    else:
        topic = re.sub(r"^\s*(please\s+)?((can|could|would|will) you\s+)?(please\s+)?(make|create|generate|build|prepare|produce|design|"
                       r"draft|give)( me)?\s*", "", query, flags=re.I)
        topic = re.sub(rf"^(an?|the|some|one|\d+)?\s*([\w-]+\s+){{0,2}}?{nouns}\b", "", topic, flags=re.I)
    topic = re.sub(r"\s*(please|thanks?|thank you)[.!?]*$", "", topic.strip(" .!?\"'"), flags=re.I).strip(" .!?\"'")
    if topic.lower() in ("", "it", "this", "that", "this topic", "the topic", "what we discussed", "our conversation", "the above", "above"):
        return ""
    return topic[:150]

def _files_mentioned(query):
    return bool(st.session_state.get("uploaded_files")) and re.search(
        r"\b(file|document|pdf|upload|paper|report|article|data|csv|this|it)\b", query, re.I) is not None

def make_ppt(query):
    topic = _topic_from(query, Cfg.PPT_NOUNS)
    _ui_step("Writing the slides…")
    ask = (f"Create the content for a PowerPoint presentation {('about: ' + topic) if topic else 'on the topic the user means (use the conversation)'}.\n"
           f"The user's request: {query}\n\n"
           "Return ONLY a JSON object, with no code fences and no other text, shaped like:\n"
           '{"title": "...", "subtitle": "...", "slides": [{"title": "...", "bullets": ["...", "..."], "notes": "..."}]}\n'
           "Rules: 6-10 content slides in a logical order (introduction first; conclusion or key takeaways last); 3-5 bullets per slide, "
           "each under 110 characters, specific and informative (facts, numbers, examples) rather than vague; 1-3 sentences of speaker "
           "notes per slide; follow any slide count, audience or style the user asked for. Plain text only: no Markdown and no LaTeX.")
    doc_block = select_doc_context(query, "file_task") if _files_mentioned(query) else ""
    raw = llm_quiet(build_messages(ask, doc_block=doc_block), effort="medium", max_tokens=4000, temperature=0.5)
    if raw.startswith("AI service temporarily unavailable"):
        return "❌ Sorry, the AI service is busy right now, so I couldn't write the slides. Please try again in a minute."
    data = _parse_json(raw) or {}
    slides = [s for s in (data.get("slides") or []) if isinstance(s, dict) and s.get("title")]
    title = str(data.get("title") or topic or "Presentation").strip()[:120]
    if slides:
        ppt_bytes = create_ppt_from_slides(title, slides, data.get("subtitle"))
    else:
        ppt_bytes = create_ppt_from_content(title, raw)   # reply wasn't JSON: use the text layout
    if not ppt_bytes:
        return "❌ Sorry, I couldn't create the PowerPoint. Please try again."
    st.session_state.ppt_data = ppt_bytes
    st.session_state.ppt_topic = title
    st.session_state.last_ppt_topic = title
    st.session_state.last_ppt_content = raw
    st.session_state.show_ppt_download = True
    if not slides:
        return f"✅ I've created a PowerPoint presentation about **{title}**. Scroll down to download it!"
    outline = "\n".join(f"{i}. {s['title']}" for i, s in enumerate(slides, 1))
    return (f"✅ I've created a PowerPoint presentation: **{title}** ({len(slides)} slides plus a title slide, with speaker notes).\n\n"
            f"**Outline**\n{outline}\n\nScroll down to download it!")

def make_word(query):
    topic = _topic_from(query, Cfg.WORD_NOUNS)
    st.session_state.last_document_topic = topic
    ask = (f"Write a complete, well-structured document {('about: ' + topic) if topic else 'on the topic the user means (use the conversation)'}.\n"
           f"The user's request: {query}\n\n"
           "Format it in Markdown: the first line is '# Title', then an introduction, 3-6 sections with '## ' headings ('### ' "
           "sub-headings, bullet lists and tables where they help) and a conclusion. Be specific and substantive: concrete facts, "
           "examples and numbers. About 700-1,200 words unless the user asked for a different length. Output only the document.")
    doc_block = select_doc_context(query, "file_task") if _files_mentioned(query) else ""
    _ui_step("Writing the document…")
    with LLMOptions(effort="medium", max_tokens=4000, temperature=0.6):
        md = llm(build_messages(ask, doc_block=doc_block))
    if md.startswith("AI service temporarily unavailable"):
        return "❌ Sorry, the AI service is busy right now, so I couldn't write the document. Please try again in a minute."
    md = clean_answer(md)
    m = re.match(r"\s*#\s+(.+)", md)
    title = re.sub(r"[*_`]", "", m.group(1)).strip()[:150] if m else (topic or "Document")
    word_bytes = create_word_from_markdown(title, md[m.end():] if m else md) or create_word_from_content(title, md)
    if not word_bytes:
        return "❌ Sorry, I couldn't create the Word document. Please try again."
    st.session_state.word_data = word_bytes
    st.session_state.word_topic = title
    st.session_state.show_word_download = True
    return md + "\n\n---\n✅ **Saved as a Word document** - scroll down to download it!"

def _sample_table(topic):
    """Built-in sample data, used only if the model can't produce a table."""
    t = topic.lower()
    if "student" in t:
        return [["Student ID", "Name", "Grade", "Subject", "Score", "Attendance"],
                ["S001", "Emma Watson", "10th", "Mathematics", 95, "98%"], ["S002", "Liam Chen", "10th", "Science", 88, "95%"],
                ["S003", "Sophia Patel", "11th", "English", 92, "100%"], ["S004", "Noah Kim", "9th", "History", 85, "92%"],
                ["S005", "Olivia Jones", "12th", "Physics", 91, "97%"]]
    if "product" in t or "inventory" in t:
        return [["SKU", "Product", "Category", "In Stock", "Unit Price"],
                ["P-100", "Wireless Mouse", "Accessories", 120, 19.99], ["P-101", "USB-C Hub", "Accessories", 75, 34.50],
                ["P-102", "27in Monitor", "Displays", 18, 229.00], ["P-103", "Mechanical Keyboard", "Accessories", 42, 89.00],
                ["P-104", "Laptop Stand", "Office", 63, 39.90]]
    rows = [["Date", "Sales Rep", "Region", "Phone Model", "Quantity", "Unit Price", "Total Sales"],
            ["2024-01-01", "John Smith", "North", "iPhone 15 Pro", 5, 999, 4995], ["2024-01-02", "Jane Doe", "South", "Samsung Galaxy S24", 3, 899, 2697],
            ["2024-01-03", "John Smith", "North", "Google Pixel 8", 4, 699, 2796], ["2024-01-04", "Bob Wilson", "East", "iPhone 15", 6, 799, 4794],
            ["2024-01-05", "Jane Doe", "South", "Samsung Galaxy S24+", 2, 1099, 2198]]
    return rows

def make_excel(query):
    topic = _topic_from(query, Cfg.EXCEL_NOUNS) or ""
    _ui_step("Building the spreadsheet…")
    ask = (f"Design an Excel spreadsheet {('about: ' + topic) if topic else 'for what the user means (use the conversation)'}.\n"
           f"The user's request: {query}\n\n"
           "Return ONLY a JSON object, with no code fences and no other text, shaped like:\n"
           '{"title": "...", "columns": ["...", "..."], "rows": [[...], [...]], "formulas": {"Column name": "Other column * Another column"}, '
           '"sum_columns": ["..."]}\n'
           "Rules: 5-8 useful columns; 15-25 rows of realistic, varied, internally consistent data (numbers as JSON numbers, dates as "
           "YYYY-MM-DD); use the user's own columns or data if they gave any. 'formulas' (optional) turns computed columns into live "
           "Excel formulas written with column names; 'sum_columns' lists numeric columns worth totalling (may be empty).")
    doc_block = select_doc_context(query, "file_task") if _files_mentioned(query) else ""
    raw = llm_quiet(build_messages(ask, doc_block=doc_block), effort="medium", max_tokens=4000, temperature=0.5)
    data = _parse_json(raw) or {}
    cols, rows = data.get("columns"), data.get("rows")
    title = re.sub(r"[^\w\s-]", "", str(data.get("title") or topic or "Spreadsheet")).strip()[:60] or "Spreadsheet"
    excel = None
    if isinstance(cols, list) and cols and isinstance(rows, list) and rows and all(isinstance(r, list) for r in rows):
        excel = create_excel_from_table(title, cols, rows, data.get("sum_columns") or [], data.get("formulas") or {})
    if excel is None:                                   # model unavailable or reply not usable: sample data
        rows_all = _sample_table(topic or title)
        cols, rows = rows_all[0], rows_all[1:]
        title = (topic or "Sample data").replace(" ", "_")[:30]
        excel = create_real_excel_file(title, rows_all)
        data = {}
    if not excel:
        csv_data = create_csv_from_data(title, [cols] + rows)
        if csv_data:
            st.session_state.csv_data, st.session_state.csv_topic, st.session_state.show_csv_download = csv_data, title, True
            return f"⚠️ Excel creation failed, but I've created a CSV file: **{title}**. Scroll down to download it!"
        return "❌ Sorry, I couldn't create the file. Please try again."
    safe = title.replace(" ", "_")
    st.session_state.excel_data, st.session_state.excel_topic, st.session_state.show_excel_download = excel, safe, True
    st.session_state.last_excel_topic, st.session_state.last_excel_data = safe, [cols] + rows
    preview = "| " + " | ".join(map(str, cols)) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join(str(v) for v in (list(r) + [""] * len(cols))[:len(cols)]) + " |" for r in rows[:5])
    extras = []
    if data.get("formulas"):
        extras.append("live formulas for " + ", ".join(k for k in data["formulas"] if k in cols))
    if data.get("sum_columns"):
        extras.append("a TOTAL row")
    note = f" It includes {' and '.join(extras)}." if extras else ""
    if not data:
        note += " (The AI service was busy, so it contains sample data.)"
    return (f"✅ I've created an Excel (.xlsx) file: **{title}** - {len(rows)} rows × {len(cols)} columns.{note}\n\n"
            f"First rows:\n\n{preview}\n\nScroll down to download it!")

def _image_prompt(query):
    """'generate image of a red bicycle' -> 'a red bicycle'; words like logo, poster or watercolor painting are kept."""
    p = re.sub(r"^\s*(please\s+)?((can|could|would) you\s+)?(please\s+)?(generate|create|make|draw|paint|render|design|produce|show me|give me)"
               r"(\s+me)?\s*", "", query, flags=re.I)
    p = re.sub(r"^((an?|the|some)\s+)?((ai|realistic|digital|high[- ]quality|hd|4k)\s+)?(image|picture|pic|photo|photograph)s?\s*"
               r"(of|showing|with|depicting)?\s*", "", p, flags=re.I)
    p = " ".join(p.strip(" .!?").split())
    return p or query.strip()

def make_image(query):
    _ui_step("Creating the image…")
    prompt = _image_prompt(query)
    st.session_state.last_image_prompt = prompt
    return generate_and_display_image(prompt)

def edit_image(query):
    last = st.session_state.get("last_image_prompt")
    if not last:
        return "There's no image to edit yet - ask me to generate one first, e.g. *generate image of a cat*."
    change = re.sub(r"^(now |ok(ay)? |but |and |please |can you |could you )*((make|change|turn|edit|modify|update)"
                    r"( it| the (image|picture|photo)| that| this)?( to| into| so (it|that))?)?\s*", "", query.strip(), flags=re.I)
    change = " ".join(change.strip(" .!?").split()) or query
    _ui_step("Editing the image…")
    st.session_state.last_image_prompt = f"{last}, {change}"
    return generate_and_display_image(st.session_state.last_image_prompt, is_edit=True)

def answer_about_links(query):
    urls = extract_urls_from_query(query)[:2]
    question = re.sub(r"https?://\S+", " ", query).strip() or "Summarize this page."
    _ui_step("Reading the page…")
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        pages = list(pool.map(_fetch_page, urls))
    blocks = []
    for (title, body), url in zip(pages, urls):
        if body:
            text = body[:2500] + ("\n…\n" + _best_passages(question, body[2500:], 2000) if len(body) > 2500 else "")
            blocks.append(f"WEB PAGE {url}" + (f" ({title})" if title else "") + f":\n{text}")
            st.session_state["_search_sources"] = (st.session_state.get("_search_sources") or []) + [(title or urlparse(url).netloc, url)]
    if not blocks:
        return "I couldn't read that link - the site may block automated access. You can paste the text here and I'll work with it."
    return answer_question(question, "reason", extra_context="\n\n".join(blocks),
                           note="Answer from the page content above, and say if it doesn't contain the answer.",
                           tools=[{"type": "code_interpreter"}] if _tools_ok() else None)

# ============================================
# IMAGE GENERATION FUNCTIONS
# ============================================

def generate_image(prompt):
    try:
        encoded_prompt = requests.utils.quote(prompt)
        timestamp = int(time.time())
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&_={timestamp}"
        return image_url
    except:
        return None

def generate_and_display_image(prompt, is_edit=False):
    image_url = generate_image(prompt)
    if image_url:
        if is_edit:
            return f"🎨 **Edited Image - New Prompt:** '{prompt}'\n\n![Generated Image]({image_url})\n\n*Image generated by AI*"
        else:
            return f"🎨 **Generated Image for:** '{prompt}'\n\n![Generated Image]({image_url})\n\n*Image generated by AI*"
    else:
        return "❌ Sorry, I couldn't generate an image right now."

# ============================================
# RUN AGENT FUNCTION
# ============================================

def run_agent(query):
    q = " ".join(query.lower().split())
    st.session_state["_current_query"] = query
    st.session_state["_search_sources"] = []
    st.session_state["_tool_sources"] = []
    st.session_state["_turn"] = {"kind": None, "model": None, "ab_arm": _ab_arm()}
    note_user_facts(query)

    if re.search(r"\bforget (everything|all of (it|that)|what you know|all about me|about me)\b|\bforget me\b", q):
        st.session_state.user_facts = {}
        st.session_state.convo_summary = ""
        st.session_state.summarized_upto = len(st.session_state.get("chat_history") or [])
        return "Done - I've forgotten what you told me about yourself and my summary of this conversation."

    reset_phrases = ["leave the document", "clear context", "forget the file", "start fresh", "clear files", "new chat"]
    if len(q.split()) <= 6 and any(phrase in q for phrase in reset_phrases):
        st.session_state.file_context = ""
        st.session_state.uploaded_files = {}
        st.session_state.doc_index = {}
        st.session_state.last_search_query = None
        st.session_state.last_search_results = None
        st.session_state.last_topic = None
        st.session_state.last_image_prompt = None
        return "✅ Context cleared! How can I help you today?"

    if q.rstrip("?") in ["can you generate images", "do you generate images", "can you create images", "can you draw", "are you able to generate images"]:
        return "Yes, I can generate images! Just tell me what you want, for example: 'generate image of a cat' or 'draw a beautiful sunset'"

    if len(q.split()) <= 6 and re.search(r"\b(who are you|what are you|who is this)\b", q):
        return "I'm MozeAI, your AI assistant! Created by Mukiibi Moses, a Computer Engineering student and graduate researcher in the BCML lab (Bio Computing and Machine Learning lab) at Kwangwoon University, Seoul, South Korea. I can search the web, run code, analyze files, remember our conversation, generate images, and create PowerPoint, Word and Excel files. What can I do for you?"

    if any(phrase in q for phrase in ["mukiibi moses", "who is moses", "your maker", "your creator", "who created you", "who made you", "who built you"]):
        return "Mukiibi Moses is my creator, a Computer Engineering student and graduate researcher in the BCML lab (Bio Computing and Machine Learning lab) at Kwangwoon University in Seoul, South Korea, specializing in AI and machine learning. His portfolio: https://moze12432.github.io/"

    kind = route(query)
    _turn_note(kind=kind)
    if kind == "datetime":
        now = datetime.now(pytz.timezone("Asia/Seoul"))
        return f"It's **{now.strftime('%I:%M %p').lstrip('0')}** on **{now.strftime('%A, %B')} {now.day}, {now.year}** (Seoul time, KST)."
    if kind == "calculator":
        return f"Result: **{calculator(query)}**"
    if kind == "generate_image":
        return make_image(query)
    if kind == "edit_image":
        return edit_image(query)
    if kind == "ppt":
        return make_ppt(query)
    if kind == "word":
        return make_word(query)
    if kind == "excel":
        return make_excel(query)
    if kind == "scrape_url":
        answer = answer_about_links(query)
    elif _is_bare_follow_up(q) and st.session_state.get("last_response"):
        answer = continue_answer(query, q)
    else:
        answer = answer_question(query, kind)
    if not answer.startswith("AI service temporarily unavailable"):
        st.session_state.last_response = answer
        if kind not in ("chat",):
            st.session_state.last_kind = kind
    return answer

# ============================================
# LEARNING FROM FEEDBACK (👍/👎 saved to a private GitHub repo)
# ============================================
# Streamlit secrets (app settings -> Secrets):
#   [feedback]
#   repo  = "owner/mozeai-feedback"      # a PRIVATE repo
#   token = "github_pat_..."             # fine-grained token: that repo only, Contents read & write
# Ratings are written to feedback/YYYY/MM/<id>.json. A weekly GitHub Action in that repo turns them into
# learned/rules.md, learned/examples.jsonl and learned/settings.json via a pull request; the app only
# uses what has been merged into the main branch.

def _gh_cfg():
    try:
        sec = st.secrets["feedback"]
        repo, token = sec.get("repo"), sec.get("token")
        if repo and token:
            return {"repo": repo, "token": token, "branch": sec.get("branch", "main"),
                    "api": str(sec.get("api", "https://api.github.com")).rstrip("/")}
    except Exception:
        pass
    return None

def _gh(method, path, cfg, accept="application/vnd.github+json", timeout=10, **kw):
    return requests.request(method, f"{cfg['api']}/repos/{cfg['repo']}/contents/{path}", timeout=timeout,
                            headers={"Authorization": f"Bearer {cfg['token']}", "Accept": accept,
                                     "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "MozeAI"}, **kw)

def _turn_note(**info):
    """Remember what the app did for the current answer (saved with any rating it gets)."""
    if isinstance(st.session_state.get("_turn"), dict):
        st.session_state["_turn"].update(info)

def _app_version():
    try:
        return hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:10]
    except Exception:
        return ""

def _redact(text):
    """Mask e-mail addresses and long phone/card-like numbers before anything is stored."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", str(text))
    return re.sub(r"(?<![\w.])\+?\d(?:[\d -]{7,}\d)(?!\w|\.\d)", "[number]", text)

def _learned():
    """Approved lessons from the feedback repo (rules, examples, settings), cached for an hour per session."""
    data = st.session_state.get("_learned")
    if data and time.time() - data["t"] < (3600 if data.get("ok") else 600):
        return data
    data = {"t": time.time(), "ok": False, "rules": "", "examples": [], "settings": {}, "version": ""}
    cfg = _gh_cfg()
    if cfg:
        try:
            files = {}
            for name in ("learned/rules.md", "learned/examples.jsonl", "learned/settings.json"):
                r = _gh("GET", name, cfg, accept="application/vnd.github.raw+json", timeout=6, params={"ref": cfg["branch"]})
                files[name] = r.text if r.status_code == 200 else ""
            rules = [re.sub(r"<!--.*?-->", "", l).strip() for l in files["learned/rules.md"].splitlines()]
            data["rules"] = "\n".join(l for l in rules if l.startswith(("- ", "* ")))[:Cfg.LESSONS_CHARS]
            data["examples"] = [json.loads(l) for l in files["learned/examples.jsonl"].splitlines() if l.strip()][:400]
            data["settings"] = json.loads(files["learned/settings.json"] or "{}")
            data["version"] = hashlib.sha1("".join(files.values()).encode()).hexdigest()[:10]
            data["ok"] = True
        except Exception as e:
            print(f"[feedback] could not load learned files: {type(e).__name__}: {e}", flush=True)
    st.session_state["_learned"] = data
    return data

def _ab_arm():
    """A/B test group for this chat: "lessons" (learned rules, examples, settings) or "control" (none).

    Drawn at random once per chat. learned/settings.json controls it:
      "ab_test": {"enabled": true, "share_with_lessons": 0.5}
    """
    arm = st.session_state.get("ab_arm")
    if arm not in ("lessons", "control"):
        data = _learned()
        ab = (data.get("settings") or {}).get("ab_test") or {}
        if data.get("ok") and ab.get("enabled", True):
            share = min(1.0, max(0.0, float(ab.get("share_with_lessons", 0.5))))
            arm = "lessons" if random.random() < share else "control"
        else:
            arm = "lessons"
        st.session_state.ab_arm = arm
    return arm

def _use_lessons():
    return _ab_arm() == "lessons"

def _feedback_examples(query, kind):
    """Up to two well-rated past answers to similar questions, as text for the system prompt."""
    data = _learned()
    examples = data.get("examples") or []
    if not examples or not _use_lessons():
        return [], ""
    if "bm25" not in data:
        data["bm25"] = BM25([e.get("question", "") for e in examples])
    scores = data["bm25"].scores(query)
    ranked = sorted(range(len(examples)), key=lambda i: -scores[i] * (1.25 if examples[i].get("kind") == kind else 1.0))
    chosen = [i for i in ranked[:2] if scores[i] >= Cfg.EXAMPLE_MIN]
    text = "\n\n".join(f"Q: {_clip(examples[i].get('question', ''), 300)}\nA: {_clip(examples[i].get('answer', ''), 800)}" for i in chosen)
    return [examples[i].get("id") for i in chosen], text

def _learned_effort(kind, query):
    """Thinking effort the feedback favours for this kind of question (None = use the normal choice)."""
    if not _use_lessons() or st.session_state.get("deep_mode") or re.search(r"\b(think|in depth|in-depth|deep dive|quick(ly)?|brief(ly)?|short answer)\b", query.lower()):
        return None
    effort = ((_learned().get("settings") or {}).get("effort") or {}).get(kind)
    return effort if effort in ("low", "medium", "high") else None

def _save_feedback(rec, sha=None):
    """Write one rating file to the feedback repo. Returns (sha, error)."""
    cfg = _gh_cfg()
    if not cfg:
        return None, "feedback storage is not configured"
    path = f"feedback/{rec['time'][:4]}/{rec['time'][5:7]}/{rec['id']}.json"
    body = {"message": f"feedback {rec['id']} ({'+1' if rec['rating'] > 0 else '-1' if rec['rating'] < 0 else 'cleared'})",
            "content": base64.b64encode(json.dumps(rec, ensure_ascii=False, indent=1).encode("utf-8")).decode(),
            "branch": cfg["branch"]}
    try:
        for _ in range(2):
            if sha:
                body["sha"] = sha
            r = _gh("PUT", path, cfg, json=body)
            if r.status_code in (200, 201):
                return r.json()["content"]["sha"], None
            if r.status_code in (409, 422):     # file changed or already exists: get its current version and retry
                g = _gh("GET", path, cfg, params={"ref": cfg["branch"]})
                sha = g.json().get("sha") if g.status_code == 200 else None
                continue
            break
        return None, f"GitHub answered {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _feedback_record(i, rating, state):
    history = st.session_state.chat_history
    answer = history[i][1] if i < len(history) else ""
    q_index = next((j for j in range(i - 1, -1, -1) if history[j][0] == "user"), None)
    question = history[q_index][1] if q_index is not None else ""
    context = history[max(0, (q_index or 0) - 4):(q_index or 0)]
    share = st.session_state.get("share_feedback", True)
    meta = (st.session_state.get("answer_meta") or {}).get(i, {})
    return {
        "id": state["id"], "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "rating": rating,
        "reasons": state.get("reasons", []), "comment": _redact(state.get("comment", ""))[:1000],
        "kind": meta.get("kind"), "effort": meta.get("effort"), "model": meta.get("model"),
        "tools_offered": meta.get("tools_offered", []), "tools_used": meta.get("tools_used", []),
        "files_used": meta.get("files_used", False), "examples_used": meta.get("examples", []),
        "learned_version": meta.get("learned_version", ""), "ab_arm": meta.get("ab_arm"), "think_seconds": meta.get("think_seconds"),
        "app_version": meta.get("app_version", ""), "chat": st.session_state.get("chat_id"), "turn": i,
        "shared_text": share,
        "question": _redact(question)[:4000] if share else "",
        "answer": _redact(_plain(answer))[:12000] if share else "",
        "context": [{"role": r, "content": _redact(_clip(_plain(m), 800))} for r, m in context] if share else [],
    }

def _on_rate(key, i):
    value = st.session_state.get(key)
    rating = 0 if value is None else (1 if value == 1 else -1)
    state = st.session_state.feedback_state.setdefault(key, {"id": uuid.uuid4().hex[:12]})
    state["rating"] = rating
    sha, err = _save_feedback(_feedback_record(i, rating, state), state.get("sha"))
    state["sha"] = sha or state.get("sha")
    if err:
        print(f"[feedback] not saved: {err}", flush=True)
    if rating > 0:
        st.session_state["_toast"] = "Thanks for the feedback! 👍"

def _on_reasons(key, i):
    state = st.session_state.feedback_state.setdefault(key, {"id": uuid.uuid4().hex[:12], "rating": -1})
    state["reasons"] = list(st.session_state.get(f"{key}_r") or [])
    state["comment"] = st.session_state.get(f"{key}_c") or ""
    state["reasons_sent"] = True
    sha, err = _save_feedback(_feedback_record(i, state.get("rating", -1), state), state.get("sha"))
    state["sha"] = sha or state.get("sha")
    if err:
        print(f"[feedback] reasons not saved: {err}", flush=True)
    st.session_state["_toast"] = "Thanks - this helps MozeAI improve."

def feedback_widgets(i):
    """👍/👎 under an answer; after 👎, ask what went wrong."""
    key = f"fb_{st.session_state.chat_id}_{i}"
    st.feedback("thumbs", key=key, on_change=_on_rate, args=(key, i))
    state = st.session_state.feedback_state.get(key, {})
    if state.get("rating") == -1 and not state.get("reasons_sent"):
        with st.container(border=True):
            st.pills("What went wrong?", Cfg.REASONS, selection_mode="multi", key=f"{key}_r")
            st.text_input("Anything else? (optional)", key=f"{key}_c", placeholder="e.g. it should have checked the date")
            st.button("Send feedback", key=f"{key}_s", on_click=_on_reasons, args=(key, i))

# ============================================
# UI - MAIN DISPLAY
# ============================================

st.markdown('<h1 style="text-align: center;">🧠 MozeAI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Intelligent AI Assistant</p>', unsafe_allow_html=True)
st.markdown("---")

if st.session_state.get("_toast"):
    st.toast(st.session_state["_toast"])
    st.session_state["_toast"] = None

with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")

    if st.button("🔄 New Chat", key="new_chat_btn", use_container_width=True):
        if not st.session_state.get("is_resetting", False):
            st.session_state.is_resetting = True
            st.session_state.chat_history = []
            st.session_state.uploaded_files = {}
            st.session_state.file_context = ""
            st.session_state.doc_index = {}
            st.session_state.last_image_prompt = None
            st.session_state.last_search_query = None
            st.session_state.last_search_results = None
            st.session_state.last_response = None
            st.session_state.last_kind = None
            st.session_state.thoughts = {}
            st.session_state.convo_summary = ""
            st.session_state.summarized_upto = 0
            st.session_state.chat_id = uuid.uuid4().hex[:8]
            st.session_state.ab_arm = None
            st.session_state.answer_meta = {}
            st.session_state.feedback_state = {}
            st.session_state.is_resetting = False
            st.success("New chat started!")
            st.rerun()

    if st.button("🗑️ Clear Files", key="clear_files_btn", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.session_state.doc_index = {}
        st.success("Files cleared!")
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.toggle("🧠 Deep thinking", key="deep_mode",
              help="Think longer on every question. Slower, but best for hard maths, code and analysis.")
    st.toggle("🌐 Web search", key="web_on",
              help="Let MozeAI search the web when a question needs current information.")
    st.toggle("📊 Share rated answers", key="share_feedback",
              help="When you rate an answer 👍 or 👎, the question and answer are saved privately (e-mails and phone numbers "
                   "masked) so MozeAI can learn from them. Turn this off to send only the rating.")

    st.markdown("---")
    st.markdown("### 📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        key="sidebar_uploader",
        label_visibility="collapsed"
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    content = process_uploaded_file(file)
                    if content and not content.startswith(("Error", "Unsupported", "No extractable", "File is empty", "CSV file appears empty")):
                        st.session_state.uploaded_files[file.name] = content
                        st.success(f"✅ {file.name}")
                    else:
                        st.warning(f"⚠️ {file.name}: {content}")

        if st.session_state.uploaded_files:
            parts = []
            for name, content in st.session_state.uploaded_files.items():
                parts.append(f"\n{'='*50}\n📄 {name}\n{'='*50}\n{content}\n")
            st.session_state.file_context = "\n".join(parts)
            sections = len(_doc_index().get("chunks") or [])
            st.info(f"📁 {len(st.session_state.uploaded_files)} file(s) loaded · {sections} sections searchable")

    facts = st.session_state.get("user_facts") or {}
    summary = st.session_state.get("convo_summary") or ""
    if facts or summary:
        st.markdown("---")
        with st.expander("🧠 What I remember"):
            for k, v in facts.items():
                st.caption(f"**{k}:** {v}")
            if summary:
                st.caption("**Earlier in this chat:**\n" + summary)
            st.caption("Kept only in this browser session.")
            if st.button("Forget all", key="forget_btn"):
                st.session_state.user_facts = {}
                st.session_state.convo_summary = ""
                st.session_state.summarized_upto = len(st.session_state.chat_history)
                st.rerun()

    st.markdown("---")
    st.markdown("### 🎨 Image Generation")
    st.markdown("**Generate:** `generate image of a cat`")
    st.markdown("**Edit:** `make it black` or `add a hat`")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("**Creator:** Mukiibi Moses")
    st.markdown("**University:** Kwangwoon University, Seoul, South Korea")
    st.markdown("**Lab:** BCML (Bio Computing and Machine Learning lab)")
    if st.session_state.last_model_used:
        st.caption(f"Model: {st.session_state.last_model_used}")

# ============================================
# CHAT DISPLAY
# ============================================

for i, (role, msg) in enumerate(st.session_state.chat_history):
    with st.chat_message(role):
        if role == "assistant" and i in st.session_state.thoughts:
            thought = st.session_state.thoughts[i]
            label = thought[2] if len(thought) > 2 else f"Thought for {max(1, round(thought[1]))}s"
            with st.status(label, state="complete", expanded=False):
                st.caption(format_math(_strip_cites(thought[0])))
        st.markdown(format_math(msg))
        if role == "assistant":
            feedback_widgets(i)

query = st.chat_input("Ask me anything - generate images, search the web, analyze files, code, and more!")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(format_math(query))

    with st.chat_message("assistant"):
        ui = StreamUI(st.container())
        st.session_state["_stream_ui"] = ui  # llm_with_fallback() streams into this
        try:
            response = run_agent(query)
        except Exception:
            traceback.print_exc()
            response = "Sorry - something went wrong while answering. Please try again."
        finally:
            st.session_state["_stream_ui"] = None
            st.session_state["_llm_opts"] = None
        ui.finish(response)

    st.session_state.chat_history.append(("assistant", response))
    st.session_state.answer_meta[len(st.session_state.chat_history) - 1] = {
        **(st.session_state.get("_turn") or {}), "tools_used": ui.used,
        "think_seconds": round(ui.seconds, 1), "app_version": _app_version()}
    if ui.thinking.strip():
        st.session_state.thoughts[len(st.session_state.chat_history) - 1] = (ui.thinking, ui.seconds, ui.label())
    try:
        update_conversation_memory()
    except Exception:
        traceback.print_exc()
    st.rerun()

# ============================================
# DOWNLOAD BUTTONS (from commit 08973a7)
# ============================================

if st.session_state.get("show_ppt_download", False) and st.session_state.get("ppt_data"):
    st.markdown("---")
    st.success(f"✅ PowerPoint about **{st.session_state.ppt_topic}** is ready!")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download PowerPoint",
            data=st.session_state.ppt_data,
            file_name=f"{st.session_state.ppt_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True
        )
    st.session_state.show_ppt_download = False
    st.session_state.ppt_data = None

if st.session_state.get("show_word_download", False) and st.session_state.get("word_data"):
    st.markdown("---")
    st.success(f"✅ Word document about **{st.session_state.word_topic}** is ready!")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📝 Download Word Document",
            data=st.session_state.word_data,
            file_name=f"{st.session_state.word_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True
        )
    st.session_state.show_word_download = False
    st.session_state.word_data = None

if st.session_state.get("show_excel_download", False) and st.session_state.get("excel_data"):
    st.markdown("---")
    st.success(f"✅ Excel (.xlsx) file **{st.session_state.excel_topic}** is ready!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📊 Download Excel File (.xlsx)",
            data=st.session_state.excel_data,
            file_name=f"{st.session_state.excel_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="excel_download_btn"
        )
    st.session_state.show_excel_download = False
    st.session_state.excel_data = None

if st.session_state.get("show_csv_download", False) and st.session_state.get("csv_data"):
    st.markdown("---")
    st.info(f"📄 CSV file **{st.session_state.csv_topic}** is ready!")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download CSV File",
            data=st.session_state.csv_data,
            file_name=f"{st.session_state.csv_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="csv_download_btn"
        )
    st.session_state.show_csv_download = False
    st.session_state.csv_data = None

