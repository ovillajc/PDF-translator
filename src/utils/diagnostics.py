import sys
import subprocess
import re
from typing import Tuple

def check_macos_performance_state() -> Tuple[bool, float]:
    """Verifica si macOS está en Modo de Bajo Consumo o con poca RAM disponible."""
    low_power = False
    avail_gb = 8.0
    if sys.platform == "darwin":
        try:
            res = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=3)
            if re.search(r"lowpowermode\s+1", res.stdout):
                low_power = True
        except Exception:
            pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024 ** 3)
        swap = psutil.swap_memory()
        if low_power or swap.percent > 80:
            print("\n[!] AVISO DE RENDIMIENTO DEL SISTEMA:")
            if low_power:
                print("    • Tu Mac tiene activado el 'Modo de bajo consumo' (lowpowermode=1).")
                print("      Desactivarlo en Configuración del Sistema > Batería duplicará la velocidad de la GPU M4.")
            if swap.percent > 80:
                print(f"    • Memoria Swap alta ({swap.used / (1024**3):.1f} GB en uso, {avail_gb:.1f} GB RAM libre).")
                print("      El script ajustará automáticamente el tamaño de lote (batch) para evitar paginación a disco.")
    except ImportError:
        pass
    return low_power, avail_gb
