import re
from typing import Optional, List, Dict, Tuple
import pymupdf as fitz
from tqdm import tqdm
import os

from src.config import *
from src.pdf.utils import *

def extract_page_blocks(page: fitz.Page, page_num: int, in_references: bool) -> Tuple[List[dict], bool]:
    """
    Combina el detector neuronal de layout (pymupdf.layout) con los bloques de texto
    estructurados para proteger figuras, tablas, ecuaciones y código fuente.
    """
    page.get_layout()
    layout_boxes = page.layout_information or []
    dict_blocks = [b for b in page.get_text("dict").get("blocks", []) if b.get("type") == 0]

    # Fallback en caso de páginas donde layout_information esté vacío pero haya texto
    if not layout_boxes and dict_blocks:
        layout_boxes = [
            [b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3], "text"]
            for b in dict_blocks
        ]

    all_box_rects = [fitz.Rect(b[:4]) for b in layout_boxes]
    items: List[dict] = []

    for idx_b, box in enumerate(layout_boxes):
        x0, y0, x1, y1, label = box[0], box[1], box[2], box[3], box[4]
        if label not in TRANSLATABLE_CLASSES:
            continue

        rect = fitz.Rect(x0, y0, x1, y1)

        # Filtrar bloques de código o fórmulas que pymupdf.layout haya agrupado como texto/lista
        overlapping_dicts = [
            b
            for b in dict_blocks
            if (fitz.Rect(b["bbox"]) & rect).get_area()
            > 0.25 * max(1.0, fitz.Rect(b["bbox"]).get_area())
        ]
        if overlapping_dicts:
            non_code_dicts = [b for b in overlapping_dicts if not is_code_or_math_dict_block(b)]
            if not non_code_dicts:
                continue
            if len(non_code_dicts) < len(overlapping_dicts):
                u = fitz.Rect(non_code_dicts[0]["bbox"])
                for b in non_code_dicts[1:]:
                    u |= fitz.Rect(b["bbox"])
                rect = u & rect

        raw_txt = page.get_text("text", clip=rect, sort=True).strip()
        clean_txt = normalize_block_text(raw_txt)

        if label == "section-header":
            header_key = clean_txt.lower().strip("0123456789. ")
            if header_key in ("references", "bibliography", "bibliografía", "referencias"):
                in_references = True
            else:
                in_references = False

        if not is_translatable_text(clean_txt, page_num, in_references):
            continue

        style = extract_dominant_style(dict_blocks, rect, label, page.rect.width)

        # Calcular caja de inserción (insert_rect) expandiendo encabezados hacia el espacio vacío
        # de su columna para evitar que palabras más largas en español reduzcan la fuente.
        insert_rect = fitz.Rect(rect)
        if label in ("section-header", "title") or style["lines"] == 1:
            if style["align"] == "left":
                col_right = (
                    page.rect.width * 0.485
                    if (rect.x0 < page.rect.width * 0.48 and rect.x1 < page.rect.width * 0.52)
                    else page.rect.width - 48.0
                )
                max_x1 = max(rect.x1, col_right)
                for other_i, other_r in enumerate(all_box_rects):
                    if (
                        other_i != idx_b
                        and other_r.y1 > rect.y0 + 1
                        and other_r.y0 < rect.y1 - 1
                        and other_r.x0 >= rect.x1 - 2
                    ):
                        max_x1 = min(max_x1, other_r.x0 - 4.0)
                insert_rect.x1 = max(rect.x1, max_x1)
            elif style["align"] == "center":
                cx = (rect.x0 + rect.x1) / 2.0
                half_w = max(rect.width / 2.0, 70.0)
                insert_rect.x0 = cx - half_w
                insert_rect.x1 = cx + half_w
            insert_rect.y0 -= 1.5
            insert_rect.y1 += 2.5
        else:
            max_y1 = rect.y1 + 5.0
            for other_i, other_r in enumerate(all_box_rects):
                if (
                    other_i != idx_b
                    and other_r.x1 > rect.x0 + 5
                    and other_r.x0 < rect.x1 - 5
                    and other_r.y0 >= rect.y1 - 2
                ):
                    max_y1 = min(max_y1, other_r.y0 - 1.5)
            insert_rect.y0 -= 1.0
            insert_rect.y1 = max(rect.y1 + 1.0, max_y1)

        items.append(
            {
                "redact_rect": rect,
                "insert_rect": insert_rect,
                "label": label,
                "text": clean_txt,
                "style": style,
            }
        )

    return items, in_references

def parse_page_range(page_range_str: Optional[str], total_pages: int) -> List[int]:
    """Convierte una cadena como '1-15' o '1,3,5-10' en índices de página 0-based."""
    if not page_range_str:
        return list(range(total_pages))
    selected = set()
    for part in page_range_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s, e = part.split("-", 1)
            start = max(1, int(s.strip()))
            end = min(total_pages, int(e.strip()))
            for p in range(start - 1, end):
                selected.add(p)
        else:
            p = int(part) - 1
            if 0 <= p < total_pages:
                selected.add(p)
    return sorted(selected)

def translate_pdf_preserve_layout(
    input_path: str,
    output_path: str,
    translator,
    page_range_str: Optional[str] = None,
    checkpoint_every: int = 25,
):
    print(f"\n[2/3] Analizando estructura neuronal y layout de: {os.path.basename(input_path)}")
    doc = fitz.open(input_path)
    total_pages = len(doc)
    target_pages = parse_page_range(page_range_str, total_pages)

    print(f"      Páginas a procesar: {len(target_pages)} de {total_pages}")

    # Procesar en ventanas de páginas (permite guardar checkpoints en libros de 500+ hojas
    # y maximizar la eficiencia del batching en GPU entre páginas consecutivas)
    in_references = False

    try:
        for chunk_start in range(0, len(target_pages), checkpoint_every):
            chunk_pages = target_pages[chunk_start : chunk_start + checkpoint_every]
            chunk_label = f"Págs {chunk_pages[0] + 1}-{chunk_pages[-1] + 1}"

            page_items_map: Dict[int, List[dict]] = {}
            chunk_texts: List[str] = []

            for pno in tqdm(chunk_pages, desc=f"Analizando layout ({chunk_label})", unit="pág"):
                page = doc[pno]
                items, in_references = extract_page_blocks(page, pno, in_references)
                page_items_map[pno] = items
                for it in items:
                    chunk_texts.append(it["text"])

            # Traducir todos los bloques del grupo de páginas en lotes paralelos de GPU
            translations = translator.translate_texts_batch(
                chunk_texts, progress_desc=f"Traduciendo ({chunk_label})"
            )

            # Aplicar redacción limpia (sin cajas blancas) e inyectar texto con estilo fiel
            for pno in chunk_pages:
                items = page_items_map.get(pno, [])
                if not items:
                    continue
                page = doc[pno]

                for it in items:
                    # fill=False elimina solo los glifos de texto sin pintar rectángulos blancos
                    # encima de diagramas, fondos o líneas vectoriales.
                    page.add_redact_annot(it["redact_rect"], fill=False)

                page.apply_redactions(
                    images=fitz.PDF_REDACT_IMAGE_NONE,
                    graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                )

                for it in items:
                    st = it["style"]
                    translated_text = translations.get(it["text"], it["text"])
                    safe_text = (
                        translated_text.replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                    )
                    # Convertir énfasis Markdown (**negrita**, *cursiva*) a etiquetas HTML reales
                    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)
                    safe_text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe_text)
                    weight = "bold" if st["bold"] else "normal"
                    fstyle = "italic" if st["italic"] else "normal"

                    css_style = (
                        f"* {{ font-family: {st['family']}; font-size: {st['size']}pt; "
                        f"color: {st['color']}; font-weight: {weight}; font-style: {fstyle}; "
                        f"line-height: 1.08; text-align: {st['align']}; margin: 0; padding: 0; }}"
                    )

                    page.insert_htmlbox(
                        it["insert_rect"],
                        safe_text,
                        css=css_style,
                        scale_low=0.62,
                    )

            # Si el libro es largo y hay más bloques pendientes, guardar checkpoint parcial
            if len(target_pages) > checkpoint_every and (chunk_start + checkpoint_every) < len(target_pages):
                print(f"      [Checkpoint] Guardando avance hasta página {chunk_pages[-1] + 1}...")
                doc.save(output_path, garbage=2, deflate=True)

    except KeyboardInterrupt:
        print("\n[!] Interrupción detectada (Ctrl+C). Guardando progreso actual en el PDF...")
        print("    (Todas las traducciones realizadas quedaron guardadas en la caché SQLite para reanudar al instante).")

    print(f"\n[3/3] Compactando fuentes y guardando PDF final en: {output_path}")
    doc.subset_fonts()
    doc.save(output_path, garbage=3, deflate=True)
    doc.close()
    translator.cache.close()
    print("¡Proceso completado con éxito!")
