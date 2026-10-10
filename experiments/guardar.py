"""
Guarda cada figura dos veces: el PNG de siempre y un EPS para la revista.

El IJF pide las figuras en EPS o PDF. EPS no admite transparencia: una banda
translúcida sale opaca y tapa lo que tiene debajo. Para no perderla:

- en cada eje que tenga piezas translúcidas se dibujan como una sola imagen, a 300
  puntos por pulgada, el fondo blanco del eje y esas piezas, con lo que las bandas y sus
  cruces quedan como en el PNG; las líneas, los ejes y el texto que van encima siguen
  siendo vectores;
- las líneas de la cuadrícula translúcidas toman de antemano el color que tendrían
  sobre blanco;
- las leyendas no cuentan: su marco semitransparente sale blanco, que se ve igual.

Ghostscript comprime el EPS al final; sin eso, la parte que va como imagen lo hace
pesar decenas de megas. El PNG se escribe primero y sin tocar nada, de modo que sale
idéntico al de antes.
"""
import matplotlib
from matplotlib.colors import to_rgb
from matplotlib.legend import Legend
from matplotlib.text import Text

PUNTOS_EPS = 300


def _translucidas(eje):
    """Piezas del eje con transparencia parcial, sin contar leyendas ni textos."""
    return [a for a in eje.get_children()
            if not isinstance(a, (Legend, Text)) and a.get_visible()
            and a.get_alpha() is not None and a.get_alpha() < 1]


def _cuadricula_translucida(eje):
    """Líneas de la cuadrícula con transparencia parcial."""
    return [linea for linea in eje.get_xgridlines() + eje.get_ygridlines()
            if linea.get_alpha() is not None and linea.get_alpha() < 1]


def guarda(fig, ruta, **opciones):
    """Escribe ``ruta`` (PNG) con ``opciones`` y, junto a ella, el mismo dibujo en EPS."""
    fig.savefig(ruta, **opciones)
    zorden, cuadricula = {}, {}
    for eje in fig.axes:
        piezas = _translucidas(eje)
        if piezas:
            zorden[eje] = eje.get_rasterization_zorder()
            eje.set_rasterization_zorder(max(a.get_zorder() for a in piezas) + 0.01)
        for linea in _cuadricula_translucida(eje):
            cuadricula[linea] = (linea.get_color(), linea.get_alpha())
            a = linea.get_alpha()
            linea.set_color([a * c + (1 - a) for c in to_rgb(linea.get_color())])
            linea.set_alpha(None)
    opciones_eps = {k: v for k, v in opciones.items() if k != "dpi"}
    try:
        with matplotlib.rc_context({"ps.usedistiller": "ghostscript"}):
            fig.savefig(ruta.with_suffix(".eps"), dpi=PUNTOS_EPS, **opciones_eps)
    finally:
        for eje, valor in zorden.items():
            eje.set_rasterization_zorder(valor)
        for linea, (color, alfa) in cuadricula.items():
            linea.set_color(color)
            linea.set_alpha(alfa)
