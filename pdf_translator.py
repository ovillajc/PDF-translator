import argparse
import os
import sys

from src.config import MODEL_PRESETS
from src.ui.selector import select_files_and_model
from src.core.translator import LocalMLXTranslator
from src.pdf.engine import translate_pdf_preserve_layout

def parse_args():
    parser = argparse.ArgumentParser(
        description="Traductor de Libros PDF Local en Apple Silicon (MLX + PyMuPDF Neural Layout)"
    )
    parser.add_argument("input", nargs="?", help="Ruta del archivo PDF de entrada")
    parser.add_argument("output", nargs="?", help="Ruta del archivo PDF de salida (opcional)")
    parser.add_argument(
        "--preset",
        choices=["balanced", "quality", "fast"],
        default=None,
        help=(
            "Perfil de modelo: 'balanced' = Google TranslateGemma-4B (default, ~2.1GB), "
            "'quality' = Tencent Hunyuan-MT-7B (~4.0GB), 'fast' = Qwen2.5-1.5B (~1.0GB)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="ID personalizado de modelo en Hugging Face (sobreescribe --preset)",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Rango de páginas a traducir, ej: '1-15' o '1,3,5-10' (por defecto: todas)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    explicit_cli_model = args.model if args.model else (MODEL_PRESETS[args.preset] if args.preset else None)

    print("=== TRADUCTOR DE LIBROS PDF LOCAL (MLX Batch + PyMuPDF Neural Layout) ===")
    ruta_entrada, ruta_salida, chosen_model = select_files_and_model(
        args.input, args.output, explicit_cli_model
    )
    print(f"Origen : {ruta_entrada}")
    print(f"Destino: {ruta_salida}")
    print(f"Modelo : {chosen_model}")

    traductor = LocalMLXTranslator(chosen_model)
    translate_pdf_preserve_layout(
        ruta_entrada,
        ruta_salida,
        traductor,
        page_range_str=args.pages,
    )
