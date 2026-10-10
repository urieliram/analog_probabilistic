"""
El idioma de las figuras: español por omisión, inglés con ``FIGURAS_IDIOMA=en``.

Cada texto visible se escribe en los dos idiomas con ``t(es, en)``. Las figuras en
inglés van a la subcarpeta ``en/`` de la carpeta de siempre, para no pisar las de
español.

Uso:
    python experiments/figuras_articulo.py                      # español
    FIGURAS_IDIOMA=en python experiments/figuras_articulo.py    # inglés
"""

import os

IDIOMA = os.environ.get("FIGURAS_IDIOMA", "es")
if IDIOMA not in ("es", "en"):
    raise ValueError(f"FIGURAS_IDIOMA debe ser 'es' o 'en', no {IDIOMA!r}")


def t(es, en):
    """El texto en el idioma de la corrida."""
    return en if IDIOMA == "en" else es


def carpeta(base):
    """La carpeta de salida: ``base`` en español, ``base/en`` en inglés, ya creada."""
    if IDIOMA == "es":
        return base
    ruta = base / "en"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta
