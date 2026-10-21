import os
import sys
import subprocess
from typing import List, Tuple, Optional
from src.config import MODEL_PRESETS, DEFAULT_MODEL_ID

def get_available_model_choices() -> List[Tuple[str, str]]:
    """
    Construye la lista de modelos disponibles para la interfaz gráfica, dejando siempre
    TranslateGemma-4B primero y por defecto, e incluyendo modelos locales en caché.
    """
    cached_repos = set()
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf_cache_dir):
        for entry in os.listdir(hf_cache_dir):
            if not entry.startswith("models--"):
                continue
            full_entry = os.path.join(hf_cache_dir, entry)
            has_weights = False
            for root_d, _, files in os.walk(full_entry):
                if any(f.endswith(".safetensors") for f in files):
                    has_weights = True
                    break
            if has_weights:
                parts = entry.split("--")[1:]
                if len(parts) >= 2:
                    cached_repos.add(f"{parts[0]}/{'--'.join(parts[1:])}")

    base_models = [
        (
            MODEL_PRESETS["balanced"],
            "TranslateGemma 4B (Por defecto - Especializado Google, ~2.1 GB)",
        ),
        (
            MODEL_PRESETS["quality"],
            "Hunyuan-MT 7B (Máxima Calidad WMT25, ~4.0 GB)",
        ),
        (
            MODEL_PRESETS["fast"],
            "Qwen 2.5 1.5B Instruct (Ultra-Rápido, ~1.0 GB)",
        ),
    ]

    choices: List[Tuple[str, str]] = []
    seen = set()

    for repo_id, desc in base_models:
        status = "[En caché local]" if repo_id in cached_repos else "[Descargar]"
        choices.append((f"{desc} {status} — {repo_id}", repo_id))
        seen.add(repo_id)

    for repo_id in sorted(cached_repos):
        if repo_id not in seen and "coder" not in repo_id.lower():
            short_name = repo_id.split("/")[-1]
            choices.append((f"{short_name} [En caché local] — {repo_id}", repo_id))
            seen.add(repo_id)

    return choices

def select_files_and_model(
    cli_input: Optional[str] = None,
    cli_output: Optional[str] = None,
    cli_model: Optional[str] = None,
) -> Tuple[str, str, str]:
    model_choices = get_available_model_choices()
    default_model_id = DEFAULT_MODEL_ID

    if cli_input:
        input_pdf = os.path.abspath(cli_input)
        if not os.path.isfile(input_pdf):
            print(f"Error: El archivo de entrada '{input_pdf}' no existe.")
            sys.exit(1)
        if cli_output:
            output_pdf = os.path.abspath(cli_output)
        else:
            base, ext = os.path.splitext(input_pdf)
            output_pdf = f"{base}_ES{ext}"
        return input_pdf, output_pdf, (cli_model or default_model_id)

    try:
        import tkinter as tk
        from tkinter import filedialog, ttk

        root = tk.Tk()
        root.title("Traductor de Libros PDF (MLX Local)")
        root.resizable(False, False)
        root.attributes("-topmost", True)

        chosen_model_id = tk.StringVar(value=cli_model or default_model_id)
        confirmed = {"ok": False}

        frame = ttk.Frame(root, padding=18)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Selecciona el modelo LLM local para la traducción:",
            font=("Helvetica", 12, "bold"),
        ).pack(anchor="w", pady=(0, 10))

        for label_text, repo_id in model_choices:
            rb = ttk.Radiobutton(
                frame,
                text=label_text,
                value=repo_id,
                variable=chosen_model_id,
            )
            rb.pack(anchor="w", pady=3)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=(14, 0))

        def on_accept(event=None):
            confirmed["ok"] = True
            root.quit()

        def on_cancel(event=None):
            root.quit()

        ttk.Button(btn_frame, text="Cancelar", command=on_cancel).pack(side="right", padx=(6, 0))
        btn_ok = ttk.Button(btn_frame, text="Continuar y elegir PDF...", command=on_accept)
        btn_ok.pack(side="right")

        root.bind("<Return>", on_accept)
        root.bind("<Escape>", on_cancel)

        root.update_idletasks()
        w, h = root.winfo_width(), root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (w // 2)
        y = (root.winfo_screenheight() // 2) - (h // 2)
        root.geometry(f"+{x}+{y}")

        root.mainloop()
        if not confirmed["ok"]:
            root.destroy()
            print("Cancelado por el usuario.")
            sys.exit(0)

        selected_model = chosen_model_id.get() or default_model_id
        root.withdraw()

        print("Esperando selección del archivo PDF de origen...")
        input_pdf = filedialog.askopenfilename(
            title="Selecciona el libro en PDF (Inglés)",
            filetypes=[("Archivos PDF", "*.pdf")],
        )
        if not input_pdf:
            root.destroy()
            print("Cancelado: No se seleccionó ningún archivo de entrada.")
            sys.exit(0)

        default_name = os.path.splitext(os.path.basename(input_pdf))[0] + "_ES.pdf"
        output_pdf = filedialog.asksaveasfilename(
            title="Guardar libro traducido (Español) como...",
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")],
        )
        root.destroy()
        if not output_pdf:
            print("Cancelado: No se seleccionó ruta de destino.")
            sys.exit(0)

        return input_pdf, output_pdf, selected_model
    except Exception:
        pass

    if sys.platform == "darwin":
        try:
            selected_model = cli_model or default_model_id
            if not cli_model:
                print("Abriendo selector gráfico de modelo LLM en macOS (Gemma por defecto)...")
                escaped_items = ", ".join(
                    f'"{lbl.replace(chr(34), chr(39))}"' for lbl, _ in model_choices
                )
                default_lbl = model_choices[0][0].replace('"', "'")
                script_model = f'''
                set modelList to {{{escaped_items}}}
                set chosenItem to choose from list modelList with title "Traductor de Libros PDF (MLX)" with prompt "Selecciona el modelo LLM local a utilizar (TranslateGemma por defecto):" default items {{"{default_lbl}"}} OK button name "Continuar" cancel button name "Cancelar"
                if chosenItem is false then
                    return "CANCELED"
                else
                    return item 1 of chosenItem
                end if
                '''
                res_model = subprocess.run(["osascript", "-e", script_model], capture_output=True, text=True)
                if res_model.returncode != 0 or res_model.stdout.strip() == "CANCELED":
                    print("Cancelado: No se seleccionó modelo.")
                    sys.exit(0)
                chosen_lbl = res_model.stdout.strip()
                for lbl, repo_id in model_choices:
                    if lbl.replace('"', "'") == chosen_lbl or repo_id in chosen_lbl:
                        selected_model = repo_id
                        break

            print("Abriendo selector nativo de macOS para el archivo PDF de origen...")
            script_open = '''
            set chosen to choose file with prompt "Selecciona el libro en PDF (Inglés)" of type {"pdf"}
            return POSIX path of chosen
            '''
            res_open = subprocess.run(["osascript", "-e", script_open], capture_output=True, text=True)
            if res_open.returncode == 0 and res_open.stdout.strip():
                input_pdf = res_open.stdout.strip()
                default_name = os.path.splitext(os.path.basename(input_pdf))[0] + "_ES.pdf"

                print("Abriendo selector nativo de macOS para guardar la traducción...")
                script_save = f'''
                set chosen to choose file name with prompt "Guardar libro traducido (Español) como..." default name "{default_name}"
                return POSIX path of chosen
                '''
                res_save = subprocess.run(["osascript", "-e", script_save], capture_output=True, text=True)
                if res_save.returncode == 0 and res_save.stdout.strip():
                    output_pdf = res_save.stdout.strip()
                    if not output_pdf.lower().endswith(".pdf"):
                        output_pdf += ".pdf"
                    return input_pdf, output_pdf, selected_model
                else:
                    print("Cancelado: No se seleccionó ruta de destino.")
                    sys.exit(0)
            elif "User canceled" in res_open.stderr or res_open.returncode != 0:
                print("Cancelado: No se seleccionó archivo de entrada.")
                sys.exit(0)
        except Exception as e:
            print(f"Aviso: No se pudo abrir el selector de macOS ({e}). Intentando alternativa...")

    print("\n--- Selección de Modelo Local ---")
    for idx, (lbl, _) in enumerate(model_choices, 1):
        print(f"  [{idx}] {lbl}")
    sel_idx = input("Elige el número de modelo [1 = TranslateGemma por defecto]: ").strip()
    selected_model = default_model_id
    if sel_idx.isdigit() and 1 <= int(sel_idx) <= len(model_choices):
        selected_model = model_choices[int(sel_idx) - 1][1]

    print("\n--- Entrada manual de rutas ---")
    input_pdf = input("Ruta del archivo PDF de entrada: ").strip().strip("'\"")
    if not os.path.isfile(input_pdf):
        print(f"Error: El archivo '{input_pdf}' no existe.")
        sys.exit(1)
    base, ext = os.path.splitext(input_pdf)
    default_out = f"{base}_ES{ext}"
    output_pdf = input(f"Ruta de guardado [{default_out}]: ").strip().strip("'\"")
    if not output_pdf:
        output_pdf = default_out
    return input_pdf, output_pdf, selected_model
