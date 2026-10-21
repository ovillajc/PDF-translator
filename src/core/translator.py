import argparse
import hashlib
import os
import re
import sqlite3
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import pymupdf.layout  # Activa el analizador neuronal ONNX (BoxRFDGNN) en PyMuPDF
import pymupdf as fitz
from mlx_lm import batch_generate, generate, load
from tqdm import tqdm

from src.config import *
from src.utils.diagnostics import check_macos_performance_state
from src.core.cache import TranslationCache
from src.pdf.utils import normalize_block_text

# ==========================================
# MOTOR DE TRADUCCIÓN MLX POR LOTES (BATCH)
# ==========================================
class LocalMLXTranslator:
    def __init__(self, model_id: str, src_lang: str = "en", tgt_lang: str = "es"):
        self.model_id = model_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.low_power, self.avail_gb = check_macos_performance_state()
        self.cache = TranslationCache()

        print(f"\n[1/3] Cargando modelo especializado '{model_id}' en memoria unificada (MLX)...")
        t0 = time.time()
        self.model, self.tokenizer = load(model_id)

        model_lower = model_id.lower()
        self.is_translategemma = "translategemma" in model_lower
        self.is_hunyuan_mt = "hunyuan-mt" in model_lower

        # Fix crítico para TranslateGemma en mlx-community: registrar <end_of_turn> como EOS
        for tok_name in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<|eos|>"):
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids(tok_name)
                if isinstance(tok_id, int) and tok_id > 0 and tok_id != self.tokenizer.unk_token_id:
                    self.tokenizer.add_eos_token(tok_name)
            except Exception:
                pass

        print(f"      Modelo listo en GPU en {time.time() - t0:.1f}s (EOS IDs: {self.tokenizer.eos_token_ids}).")

    def _build_prompt_ids(self, text: str) -> List[int]:
        if self.is_translategemma:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": self.src_lang,
                            "target_lang_code": self.tgt_lang,
                            "text": text,
                        }
                    ],
                }
            ]
        elif self.is_hunyuan_mt:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Translate the following segment into Spanish, without additional explanation.\n\n"
                        f"{text}"
                    ),
                }
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_GENERIC},
                {"role": "user", "content": text},
            ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

    @staticmethod
    def _clean_output(source: str, output: str) -> str:
        cleaned = output
        for tag in ("<end_of_turn>", "<|im_end|>", "<|eos|>", "<eos>"):
            cleaned = cleaned.replace(tag, "")
        cleaned = cleaned.strip()

        # Si el original iniciaba con numeración de sección ("1. ", "3.2. ") o letra ("A. ", "B. ")
        # y el modelo la omitió, restaurarla automáticamente:
        m_num = re.match(r"^((?:\d+(?:\.\d+)*|[A-Z])\.\s+)", source)
        if m_num:
            prefix = m_num.group(1)
            if not cleaned.startswith(prefix.strip()):
                cleaned = f"{prefix}{cleaned}"

        # Si el original iniciaba con "Figure X." o "Table X." y el modelo lo omitió:
        m_fig = re.match(r"^(Figure|Fig\.|Table)\s+(\d+[.:])\s+", source, flags=re.IGNORECASE)
        if m_fig:
            kind = "Tabla" if m_fig.group(1).lower().startswith("tab") else "Figura"
            num_part = m_fig.group(2)
            if not re.match(r"^(Figura|Fig\.|Tabla|Figure|Table)\s+\d+", cleaned, flags=re.IGNORECASE):
                cleaned = f"{kind} {num_part} {cleaned}"

        # Si el original iniciaba con viñeta "• " y el modelo la omitió, restaurarla:
        if source.startswith("•") and not cleaned.startswith("•"):
            cleaned = f"• {cleaned.lstrip('-* ')}"

        return cleaned

    @staticmethod
    def _try_fast_header_translation(text: str) -> Optional[str]:
        """Resuelve encabezados estándar de forma instantánea sin usar la GPU."""
        clean = text.strip()
        if len(clean) > 75:
            return None
        m = re.match(r"^((?:[A-Z]|\d+(?:\.\d+)*)\.?\s+)?([A-Za-z][A-Za-z\s\-&]+?)\.?$", clean)
        if not m:
            return None
        prefix = m.group(1) or ""
        core = re.sub(r"\s+", " ", m.group(2).strip()).lower()
        if core in STANDARD_HEADERS_ES:
            translated_core = STANDARD_HEADERS_ES[core]
            if m.group(2).isupper():
                translated_core = translated_core.upper()
            return f"{prefix}{translated_core}"
        return None

    def translate_texts_batch(self, texts: List[str], progress_desc: str = "Traduciendo lotes GPU") -> Dict[str, str]:
        """
        Traduce una lista de textos deduplicados usando:
        1. Diccionario instantáneo para encabezados comunes
        2. Caché persistente SQLite
        3. batch_generate en GPU con lotes ordenados por longitud y tamaño adaptativo a la RAM
        """
        unique_texts = list(dict.fromkeys(t for t in texts if t and t.strip()))
        results: Dict[str, str] = {}

        # 1. Resolución instantánea de encabezados estándar
        remaining: List[str] = []
        for t in unique_texts:
            fast_h = self._try_fast_header_translation(t)
            if fast_h is not None:
                results[t] = fast_h
            else:
                remaining.append(t)

        # 2. Consulta en caché SQLite
        if remaining:
            cached = self.cache.get_many(self.model_id, self.src_lang, self.tgt_lang, remaining)
            for k, v in cached.items():
                results[k] = self._clean_output(k, v)
            remaining = [t for t in remaining if t not in cached]

        if not remaining:
            return results

        # 3. Ordenar por longitud para minimizar padding en GPU y agrupar en lotes adaptativos
        remaining_sorted = sorted(remaining, key=len)

        # Ajustar tamaño de lote según la RAM libre del sistema (16GB M4 Air)
        ram_factor = 1 if self.avail_gb < 6.0 else 2

        batches: List[List[str]] = []
        idx = 0
        while idx < len(remaining_sorted):
            clen = len(remaining_sorted[idx])
            if clen < 160:
                bsize = 6 * ram_factor
            elif clen < 550:
                bsize = 4 * ram_factor
            elif clen < 1100:
                bsize = 3 if ram_factor == 1 else 4
            else:
                bsize = 2 if ram_factor == 1 else 3
            batches.append(remaining_sorted[idx : idx + bsize])
            idx += bsize

        pbar = tqdm(total=len(remaining_sorted), desc=progress_desc, unit="blq")
        for chunk in batches:
            prompts = []
            max_toks = []
            for t in chunk:
                p_ids = self._build_prompt_ids(t)
                prompts.append(p_ids)
                # Estimar tokens del texto fuente descontando plantilla
                src_toks = max(6, len(p_ids) - (68 if self.is_translategemma else 40))
                max_toks.append(max(24, min(int(src_toks * 1.48) + 20, 900)))

            batch_res = batch_generate(
                self.model,
                self.tokenizer,
                prompts=prompts,
                max_tokens=max_toks,
                prefill_batch_size=len(prompts),
                completion_batch_size=len(prompts),
                verbose=False,
            )
            mx.clear_cache()

            new_pairs: List[Tuple[str, str]] = []
            for src_t, raw_out in zip(chunk, batch_res.texts):
                clean_out = self._clean_output(src_t, raw_out)
                if not clean_out:
                    clean_out = src_t
                results[src_t] = clean_out
                new_pairs.append((src_t, clean_out))

            self.cache.put_many(self.model_id, self.src_lang, self.tgt_lang, new_pairs)
            pbar.update(len(chunk))

        pbar.close()
        return results

    def translate_block(self, text: str) -> str:
        """Método compatible para traducir un bloque individual."""
        clean_text = normalize_block_text(text)
        if not clean_text or len(clean_text) < 3 or clean_text.isdigit():
            return text
        res = self.translate_texts_batch([clean_text])
        return res.get(clean_text, text)
