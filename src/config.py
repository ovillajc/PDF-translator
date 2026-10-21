import os

# Cargar variables de entorno desde archivo .env si existe en la raíz del proyecto
def _load_env_file():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root_dir, ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val

_load_env_file()

# ==========================================
# MODELOS CONFIGURADOS (VÍA .ENV O PRESETS)
# ==========================================
MODEL_PRESETS = {
    # Especializado de Google para traducción (55 idiomas). ~2.1 GB en 4-bit.
    "balanced": os.getenv("MODEL_BALANCED", "mlx-community/translategemma-4b-it-4bit"),
    # Ganador #1 mundial de traducción WMT25 (Tencent). ~4.0 GB en 4-bit.
    "quality": os.getenv("MODEL_QUALITY", "mlx-community/Hunyuan-MT-7B-4bit"),
    # Ultra-rápido (~160+ tokens/seg en M4). ~1.0 GB en 4-bit.
    "fast": os.getenv("MODEL_FAST", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
}

DEFAULT_PRESET = os.getenv("DEFAULT_PRESET", "balanced")
DEFAULT_MODEL_ID = MODEL_PRESETS.get(DEFAULT_PRESET, MODEL_PRESETS["balanced"])

SYSTEM_PROMPT_GENERIC = (
    "You are a professional English to Spanish book translator. "
    "Translate the user's text into natural, accurate Spanish, preserving technical precision and tone. "
    "Output ONLY the Spanish translation without introductions, notes, or explanations."
)

# Diccionario instantáneo para encabezados estándar de libros y papers (evita llamadas al GPU
# y garantiza traducción técnica consistente p. ej. "Training" -> "Entrenamiento")
STANDARD_HEADERS_ES = {
    "abstract": "Resumen",
    "summary": "Resumen",
    "introduction": "Introducción",
    "background": "Antecedentes",
    "related work": "Trabajo relacionado",
    "preliminaries": "Preliminares",
    "method": "Método",
    "methods": "Métodos",
    "methodology": "Metodología",
    "model": "Modelo",
    "model architecture": "Arquitectura del modelo",
    "architecture": "Arquitectura",
    "training": "Entrenamiento",
    "implementation": "Implementación",
    "implementation details": "Detalles de implementación",
    "experiments": "Experimentos",
    "experimental setup": "Configuración experimental",
    "evaluation": "Evaluación",
    "results": "Resultados",
    "main results": "Resultados principales",
    "analysis": "Análisis",
    "analysis and discussions": "Análisis y discusión",
    "discussion": "Discusión",
    "ablation study": "Estudio de ablación",
    "limitations": "Limitaciones",
    "conclusion": "Conclusión",
    "conclusions": "Conclusiones",
    "impact statement": "Declaración de impacto",
    "acknowledgments": "Agradecimientos",
    "acknowledgements": "Agradecimientos",
    "references": "Referencias",
    "bibliography": "Bibliografía",
    "appendix": "Apéndice",
    "preface": "Prefacio",
    "foreword": "Prólogo",
    "table of contents": "Contenido",
    "contents": "Contenido",
    "index": "Índice",
    "overview": "Descripción general",
    "key takeaways": "Puntos clave",
    "takeaways": "Puntos clave",
    "exercises": "Ejercicios",
    "further reading": "Lecturas adicionales",
    "notes": "Notas",
    "note": "Nota",
    "warning": "Advertencia",
    "tip": "Consejo",
    "this outputs": "Esto produce",
    "the output is": "La salida es",
    "the code prints": "El código imprime",
    "here is a teaser": "Aquí hay un adelanto",
    "here’s a teaser": "Aquí hay un adelanto",
}

# Clases del analizador neuronal de layout (pymupdf.layout)
TRANSLATABLE_CLASSES = {"text", "section-header", "caption", "list-item", "title", "footnote"}
PROTECTED_CLASSES = {"picture", "formula", "table", "page-header", "page-footer"}

MONO_FONT_KEYWORDS = (
    "mono", "courier", "consolas", "inconsolata", "anonymous", "menlo",
    "monaco", "fira", "jetbrains", "sourcecode", "cmtt", "lmtt", "txtt",
    "typewriter", "pcr", "lucida", "fixed",
)
MATH_FONT_KEYWORDS = (
    "cmmi", "cmsy", "cmex", "msam", "msbm", "math", "symbol", "eusm", "eurm",
)
SANS_FONT_KEYWORDS = (
    "sans", "arial", "helvetica", "opensans", "franklin", "calibri",
    "avenir", "futura", "gill", "humanist", "verdana", "tahoma",
    "trebuchet", "inter", "roboto", "lato", "montserrat",
)
SERIF_FONT_KEYWORDS = (
    "times", "cmr", "lmr", "baskerville", "libertine", "garamond",
    "georgia", "palatino", "minion", "roman", "serif", "nimbusrom",
    "cambria", "century", "charter", "utopia", "book",
)
