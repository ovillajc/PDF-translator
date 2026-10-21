# PDF Translator MLX

Traductor local de libros y papers científicos en formato PDF manteniendo intacto el diseño visual original (maquetación, tipografía, fórmulas matemáticas, tablas, diagramas y columnas). Utiliza inferencia acelerada por hardware en la GPU mediante el framework **Apple MLX** y análisis neuronal de documentos con **PyMuPDF Layout**.

---

> [!WARNING]
> ### Compatibilidad del Sistema: Solo macOS con Apple Silicon
> Este proyecto está desarrollado específicamente para aprovechar la arquitectura de **Memoria Unificada (UMA)** y la aceleración **Metal GPU** de Apple mediante las librerías [`mlx`](https://github.com/ml-explore/mlx) y [`mlx-lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm).
>
> * **Compatible con:** macOS en Macs con chips **Apple Silicon (M1, M2, M3, M4 o superiores)**.
> * **No compatible con:** Windows o Linux (las librerías de MLX son exclusivas del ecosistema de Apple).

---

## Objetivo del Proyecto

Traducir documentos PDF extensos (como libros técnicos de 500+ páginas o papers de arXiv de múltiples columnas) conservando la calidad editorial y el diseño visual original, de forma:

1. **100% Local y Privada:** Tus documentos nunca salen de tu Mac hacia servidores o APIs de terceros.
2. **Sin Coste:** Cero cobros por tokens de APIs externas.
3. **Respetuosa con el Formato:** No reemplaza imágenes por texto plano ni desarma las columnas del libro.

---

## Características Principales

* **Analizador Neuronal de Layout (`pymupdf.layout` / ONNX):**
  Identifica y clasifica regiones de la página (títulos, encabezados, párrafos y listas) y **protege automáticamente**:
  * **Gráficos, figuras y diagramas vectoriales:** No dibuja parches blancos sobre esquemas visuales.
  * **Fórmulas matemáticas densas:** Detecta simbología LaTeX/Math y ecuaciones numeradas para mantenerlas intactas.
  * **Bloques de código:** Identifica tipografías monoespaciadas (`Courier`, `Consolas`, `Menlo`, etc.) para no traducir código fuente.
  * **Tablas y citas bibliográficas:** Evita desalinear columnas numéricas o deformar referencias bibliográficas.

* **Reconstrucción Tipográfica Inteligente:**
  * **Redacción limpia (`fill=False`):** Elimina exclusivamente los glifos del texto en inglés sin pintar cajas opacas.
  * **Estilo tipográfico ponderado:** Detecta si la fuente es *Serif* o *Sans-serif*, su color exacto y si es negrita o cursiva, ignorando subíndices/superíndices al promediar tamaños.
  * **Des-guionado automático:** Une palabras divididas por guiones al final de línea (`ran-\ndomness` → `randomness`) antes de traducir.
  * **Expansión de encabezados:** Amplía dinámicamente el ancho de inserción de títulos para que palabras más largas en español no reduzcan la fuente innecesariamente.

* **Motor GPU en Paralelo (Batching Adaptativo):**
  Procesa múltiples bloques de texto simultáneamente en la GPU Metal mediante `batch_generate()`, ajustando automáticamente el tamaño del lote según la memoria RAM libre de tu Mac.

* **Caché Persistente SQLite y Checkpoints:**
  * Guarda cada bloque traducido en una base de datos local (`~/.cache/pdf_traductor_mlx/translations_cache.sqlite3`).
  * Si cancelas con `Ctrl+C` o la batería se agota en un libro de 500 páginas, al reanudar no gastarás tiempo retraduciendo lo que ya se procesó.
  * Guarda copias parciales de seguridad del PDF cada 25 páginas.

* **Selector Nativo de macOS:**
  Incluye ventanas de diálogo nativas de Finder (vía AppleScript) para elegir el PDF de origen y la ruta de guardado sin requerir Tkinter ni configurar rutas largas manualmente.

---

## Estructura del Proyecto

El código está organizado bajo una arquitectura limpia y modular que separa claramente las responsabilidades:

```text
pdf-traductor/
├── .env.example                 # Plantilla con nombres de modelos y presets a elegir
├── .env                         # Variables de entorno locales activas (ignorado en git)
├── public/                      # Archivos de prueba y demostración
│   ├── 2509.25020.pdf           # Paper científico de ejemplo en inglés
│   └── 2509.25020_ES.pdf        # Resultado completo traducido al español
├── src/                         # Código fuente modularizado
│   ├── config.py                # Carga de .env, presets de modelos, diccionarios y constantes
│   ├── core/                    # Motor de inferencia y almacenamiento
│   │   ├── cache.py             # Base de datos SQLite para reanudación instantánea
│   │   └── translator.py        # Motor de traducción MLX por lotes (Batch GPU)
│   ├── pdf/                     # Procesamiento del PDF y reconstrucción visual
│   │   ├── engine.py            # Pipeline de análisis neuronal, redacción e inyección HTML
│   │   └── utils.py             # Extracción de tipografía, filtros y sanitización
│   ├── ui/                      # Selectores gráficos y consola
│   │   └── selector.py          # Selector nativo macOS (AppleScript), Tkinter y CLI
│   └── utils/                   # Diagnósticos del sistema
│       └── diagnostics.py       # Detección de bajo consumo y monitoreo de memoria RAM
├── requirements.txt             # Dependencias del proyecto
├── pdf_translator.py         # Punto de entrada principal y configuración CLI
└── README.md                    # Documentación del proyecto
```

---

## Modelos Soportados (Presets)

El script incluye perfiles optimizados para Macs con 16 GB de RAM. Estos modelos y su perfil predeterminado se configuran cómodamente desde el archivo `.env`:

| Perfil (`--preset`) | Variable en `.env` | Modelo Hugging Face por Defecto | VRAM / RAM | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| **`balanced`** *(Por defecto)* | `MODEL_BALANCED` | `mlx-community/translategemma-4b-it-4bit` | ~2.1 GB | **Google TranslateGemma 4B:** Entrenado específicamente para traducción (55 idiomas). Excelente consistencia técnica y cero alucinaciones. |
| **`quality`** | `MODEL_QUALITY` | `mlx-community/Hunyuan-MT-7B-4bit` | ~4.0 GB | **Tencent Hunyuan-MT 7B:** Ganador mundial WMT25. Calidad literaria y técnica máxima (recomendado con >7 GB de RAM libre). |
| **`fast`** | `MODEL_FAST` | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~1.0 GB | **Qwen2.5 1.5B:** Modo borrador de alta velocidad (~160+ tokens/seg en Apple Silicon M4). |

---

## Requisitos Previos

* **Mac con chip Apple Silicon** (M1 / M2 / M3 / M4 / M5 / M6 / Pro / Max / Ultra).
* **macOS** 13.0 (Ventura) o posterior.
* **Python** 3.10 o superior (recomendado 3.11 o 3.12).
* **Memoria RAM:** Mínimo 8 GB (16 GB recomendado para el modelo Hunyuan-MT 7B).

---

## Instalación y Puesta en Marcha

Sigue estos pasos para clonar, configurar y levantar el proyecto en tu entorno local:

### 1. Clonar o acceder al repositorio
Abre la Terminal de macOS y sitúate en el directorio del proyecto:
```zsh
cd /ruta/a/pdf-traductor
```

### 2. Crear y activar el entorno virtual
Se recomienda aislar las dependencias en un entorno virtual propio:
```zsh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar las dependencias
Instala los paquetes necesarios, incluyendo `mlx`, `mlx-lm` y `pymupdf` con soporte para modelos neuronales de maquetación:
```zsh
pip install -r requirements.txt
```

### 4. Configurar las variables de entorno (.env)
El proyecto utiliza un archivo `.env` para gestionar los nombres de los modelos de Hugging Face y el perfil predeterminado sin necesidad de modificar el código fuente.

Copia la plantilla de ejemplo para crear tu `.env` local:
```zsh
cp .env.example .env
```

Contenido editable del archivo `.env`:
```bash
# Perfil equilibrado (Google TranslateGemma-4B - Recomendado por defecto, ~2.1 GB)
MODEL_BALANCED=mlx-community/translategemma-4b-it-4bit

# Perfil de maxima calidad (Tencent Hunyuan-MT-7B, ~4.0 GB)
MODEL_QUALITY=mlx-community/Hunyuan-MT-7B-4bit

# Perfil rapido para borradores (Qwen2.5-1.5B-Instruct, ~1.0 GB)
MODEL_FAST=mlx-community/Qwen2.5-1.5B-Instruct-4bit

# Preset por defecto al ejecutar sin argumentos (balanced | quality | fast)
DEFAULT_PRESET=balanced
```

Si deseas utilizar otros modelos cuantizados de la comunidad MLX en Hugging Face, simplemente cambia los valores en tu archivo `.env`.

### 5. Verificar la instalación
Comprueba que el CLI y el entorno se hayan levantado correctamente consultando la ayuda:
```zsh
python pdf_translator.py --help
```

---

## Archivos de Ejemplo para Prueba Rápida

La carpeta `public/` contiene un documento científico real listo para probar el pipeline:

* **[public/2509.25020.pdf](public/2509.25020.pdf):** Paper original de arXiv de 14 páginas en formato a doble columna con diagramas, tablas y fórmulas matemáticas (*"Deep Thinking by Markov Chain of Continuous Thoughts"*).
* **[public/2509.25020_ES.pdf](public/2509.25020_ES.pdf):** Resultado completo traducido al español que sirve como referencia de calidad visual.

Para realizar una prueba rápida sobre este archivo:

```zsh
# Probar traduciendo las primeras 2 páginas del ejemplo
python pdf_translator.py public/2509.25020.pdf public/prueba_ES.pdf --pages 1-2

# Traducir el documento de ejemplo completo
python pdf_translator.py public/2509.25020.pdf public/2509.25020_ES.pdf
```

---

## Guía de Uso

Asegúrate de tener activo tu entorno virtual (`source .venv/bin/activate`) antes de ejecutar los comandos.

### 1. Modo Interactivo (Ventana nativa de macOS)
Si no especificas archivos, el script abrirá el selector nativo de Finder para elegir el archivo de origen y la ubicación de guardado:
```zsh
python pdf_translator.py
```

### 2. Modo Directo por Línea de Comandos (CLI)
Pasa las rutas del PDF original y de salida como argumentos:
```zsh
python pdf_translator.py "/ruta/al/libro.pdf" "/ruta/al/libro_traducido.pdf"
```
*(Si omites el archivo de salida, creará automáticamente uno con el sufijo `_ES.pdf` en la misma carpeta).*

### 3. Traducir un Rango Específico de Páginas
Ideal para verificar la calidad en unas pocas hojas antes de procesar un libro completo:
```zsh
# Traducir solo las primeras 10 páginas
python pdf_translator.py "/ruta/al/libro.pdf" --pages 1-10

# Traducir páginas específicas separadas por comas
python pdf_translator.py "/ruta/al/libro.pdf" --pages 1,3,5-8
```

### 4. Cambiar el Perfil del Modelo
```zsh
# Usar el modelo de máxima calidad (Hunyuan-MT-7B)
python pdf_translator.py "/ruta/al/libro.pdf" --preset quality

# Usar el modelo ultra-rápido (Qwen2.5-1.5B)
python pdf_translator.py "/ruta/al/libro.pdf" --preset fast
```

### 5. Utilizar un Modelo Personalizado de Hugging Face
Puedes cargar cualquier modelo cuantizado para MLX disponible en Hugging Face:
```zsh
python pdf_translator.py "/ruta/al/libro.pdf" --model "mlx-community/Llama-3.2-3B-Instruct-4bit"
```

---

## Consejos de Rendimiento para Mac M-Series

1. **Modo de Bajo Consumo:**  
   Si estás en un MacBook (Air o Pro) con batería, asegúrate de tener desactivado el *Modo de bajo consumo* en `Ajustes del Sistema > Batería`. Con el modo de bajo consumo activo, macOS limita los núcleos de la GPU Metal, reduciendo la velocidad a la mitad.
2. **Memoria Swap:**  
   Antes de procesar libros de más de 300 páginas con el perfil `quality`, cierra navegadores con decenas de pestañas u otras herramientas pesadas para evitar que macOS use swap en disco SSD.
3. **Reanudación instantánea:**  
   Si detienes el script con `Ctrl + C`, puedes volver a ejecutar exactamente el mismo comando más tarde. Gracias a la base de datos SQLite en caché, retomará el trabajo sin esperar.

---

## Licencia

Uso libre para fines educativos, de investigación y personales. Los modelos de lenguaje subyacentes se rigen por sus respectivas licencias comunitarias en Hugging Face.
