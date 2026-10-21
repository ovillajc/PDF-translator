import re
import pymupdf as fitz
from typing import List, Dict
from src.config import MONO_FONT_KEYWORDS, MATH_FONT_KEYWORDS, SANS_FONT_KEYWORDS, SERIF_FONT_KEYWORDS

def normalize_block_text(raw_text: str) -> str:
    """Une palabras cortadas por guion al final de línea y limpia saltos de línea internos."""
    # Unir palabras divididas por guion suave al final de línea (ej. "ran-\ndomness" -> "randomness")
    text = re.sub(r"([A-Za-z])-\s*\n\s*([a-z])", r"\1\2", raw_text.strip())
    # Reemplazar saltos de línea simples por espacio
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_mono_span(span: dict) -> bool:
    font = span.get("font", "").lower()
    flags = span.get("flags", 0)
    if flags & 8:  # PDF FixedPitch flag
        return True
    return any(k in font for k in MONO_FONT_KEYWORDS)

def is_math_span(span: dict) -> bool:
    font = span.get("font", "").lower()
    return any(k in font for k in MATH_FONT_KEYWORDS)

def is_code_or_math_dict_block(b: dict) -> bool:
    """Detecta bloques de código fuente (monospace) o fórmulas matemáticas densas."""
    total_chars = 0
    mono_chars = 0
    math_chars = 0
    for line in b.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "").strip()
            if not t:
                continue
            n = len(t)
            total_chars += n
            if is_mono_span(span):
                mono_chars += n
            if is_math_span(span):
                math_chars += n
    if total_chars == 0:
        return True
    if mono_chars / total_chars >= 0.55:
        return True
    if math_chars / total_chars >= 0.50:
        return True
    return False

def is_translatable_text(clean_text: str, page_num: int, in_references: bool) -> bool:
    """Filtra numeración, código, listas de autores y entradas bibliográficas."""
    if len(clean_text) <= 2:
        return False
    alpha_count = sum(c.isalpha() for c in clean_text)
    if alpha_count / len(clean_text) < 0.45:
        return False

    # Evitar traducir citas bibliográficas en la sección de Referencias/Bibliografía
    if in_references:
        if re.search(r"\b(19|20)\d{2}[a-z]?\b", clean_text) and (
            "," in clean_text
            or "arXiv" in clean_text
            or "Proceedings" in clean_text
            or "pp." in clean_text
            or "et al." in clean_text
            or "Journal" in clean_text
            or "Conference" in clean_text
        ):
            return False

    # Evitar traducir bloque de nombres de autores en la portada
    if page_num == 0 and len(clean_text) < 320:
        if (
            ("†" in clean_text or "∗" in clean_text or "*" in clean_text)
            and sum(c.isdigit() for c in clean_text) >= 4
            and "University" not in clean_text
            and "Laboratory" not in clean_text
            and "Institute" not in clean_text
        ):
            return False

    return True

def extract_dominant_style(
    dict_blocks: List[dict], clip_rect: fitz.Rect, label: str, page_width: float
) -> dict:
    """
    Extrae el estilo tipográfico ponderando por cantidad de caracteres.
    """
    size_weights: Dict[float, int] = {}
    color_weights: Dict[int, int] = {}
    bold_chars = 0
    italic_chars = 0
    serif_chars = 0
    sans_chars = 0
    total_chars = 0
    line_count = 0

    for b in dict_blocks:
        for line in b.get("lines", []):
            l_rect = fitz.Rect(line["bbox"])
            if not l_rect.intersects(clip_rect):
                continue
            inter = l_rect & clip_rect
            if inter.height < l_rect.height * 0.4:
                continue
            line_count += 1
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if not t:
                    continue
                n = len(t)
                sz = round(span.get("size", 10.0), 1)
                col = span.get("color", 0)
                font = span.get("font", "").lower()
                flags = span.get("flags", 0)

                if not (flags & 1) and not is_math_span(span):
                    size_weights[sz] = size_weights.get(sz, 0) + n
                else:
                    size_weights[sz] = size_weights.get(sz, 0) + max(1, n // 4)

                color_weights[col] = color_weights.get(col, 0) + n
                total_chars += n

                if (flags & 16) or any(w in font for w in ("bold", "black", "demi", "medi", "heavy")):
                    bold_chars += n
                if (flags & 2) or any(w in font for w in ("ital", "obli", "slant")):
                    italic_chars += n

                if any(k in font for k in SANS_FONT_KEYWORDS):
                    sans_chars += n
                elif (flags & 4) or any(k in font for k in SERIF_FONT_KEYWORDS):
                    serif_chars += n

    dom_size = max(size_weights.items(), key=lambda x: (x[1], x[0]))[0] if size_weights else 10.0
    dom_color = max(color_weights.items(), key=lambda x: x[1])[0] if color_weights else 0
    hex_color = f"#{dom_color:06x}"

    is_bold = (bold_chars / max(1, total_chars)) > 0.65 or label in ("section-header", "title")
    is_italic = (italic_chars / max(1, total_chars)) > 0.55
    font_family = "sans-serif" if sans_chars > serif_chars else "serif"

    align = "justify"
    if label in ("section-header", "title"):
        center_x = (clip_rect.x0 + clip_rect.x1) / 2.0
        if (
            abs(center_x - page_width / 2.0) < 25.0
            or (
                clip_rect.width < 120
                and (
                    abs(center_x - page_width * 0.28) < 25.0
                    or abs(center_x - page_width * 0.72) < 25.0
                )
            )
        ):
            align = "center"
        else:
            align = "left"
    elif label in ("list-item", "footnote") or line_count <= 2:
        align = "left"

    return {
        "size": dom_size,
        "color": hex_color,
        "bold": is_bold,
        "italic": is_italic,
        "family": font_family,
        "align": align,
        "lines": max(1, line_count),
    }
