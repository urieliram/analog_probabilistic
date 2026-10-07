"""
El almacén de pronósticos: producir y calificar dejan de ser lo mismo.

Hasta ahora el experimento escribía métricas ya calculadas, de modo que cambiar la
definición de un puntaje obligaba a correrlo todo otra vez —y con los comparativos
caros eso son horas—. Peor: dos cuadros del artículo podían venir de dos corridas
distintas sin que nadie lo notara, y pasó.

Aquí el experimento escribe lo que produjo —los cuantiles, y los miembros cuando el
método entrega una muestra— y la evaluación lo lee. Cambiar una métrica cuesta
segundos, todos los cuadros salen del mismo archivo, y la objeción desaparece.

Formato: un parquet por corrida, una fila por (zona, origen, método, paso), con una
columna por nivel de cuantil. Los miembros van aparte, en su propio archivo, porque
sólo algunos métodos los tienen.
"""

import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ForecastStore:
    """Acumula pronósticos en memoria y los escribe en un solo archivo."""

    def __init__(self, levels, path: Path):
        self.levels = [round(float(a), 4) for a in levels]
        self.path = Path(path)
        self._cuantiles: list = []
        self._miembros: list = []

    def add(self, zone: str, origin, method: str, quantiles: np.ndarray,
            observed: np.ndarray, members: Optional[np.ndarray] = None,
            seconds: float = np.nan) -> None:
        """
        Guarda un pronóstico.

        ``quantiles`` trae una fila por nivel y una columna por paso del horizonte,
        en el orden de ``levels``.
        """
        if quantiles.shape[0] != len(self.levels):
            raise ValueError("los cuantiles no coinciden con los niveles")
        pasos = quantiles.shape[1]

        tabla = pd.DataFrame({f"q{a:g}": quantiles[fila]
                              for fila, a in enumerate(self.levels)})
        tabla.insert(0, "step", np.arange(1, pasos + 1))
        tabla.insert(0, "method", method)
        tabla.insert(0, "origin", origin)
        tabla.insert(0, "zone", zone)
        tabla["observed"] = observed
        tabla["seconds"] = seconds
        self._cuantiles.append(tabla)

        if members is not None:
            m = pd.DataFrame(members.T)
            m.columns = [f"m{i}" for i in range(members.shape[0])]
            m.insert(0, "step", np.arange(1, pasos + 1))
            m.insert(0, "method", method)
            m.insert(0, "origin", origin)
            m.insert(0, "zone", zone)
            self._miembros.append(m)

    def limpiar(self) -> list:
        """
        Borra las partes de una corrida anterior en el mismo destino.

        Las partes se escriben zona por zona, así que una corrida interrumpida deja
        un directorio con unas zonas nuevas y el resto viejas, y la evaluación lo
        lee como si fuera una sola corrida. Ha pasado. Se limpia antes de empezar.
        """
        borrados = []
        for carpeta in (self.path.with_suffix(""),
                        Path(str(self.path.with_suffix("")) + "_members")):
            if carpeta.is_dir():
                borrados.append(str(carpeta))
                shutil.rmtree(carpeta)
        if self.path.exists():
            borrados.append(str(self.path))
            self.path.unlink()
        return borrados

    def flush(self, part: str) -> dict:
        """
        Vuelca lo acumulado como una parte y libera la memoria.

        Hace falta para corridas grandes: los miembros de veinticinco zonas por mil
        ochocientos orígenes son cientos de megabytes, y acumularlos hasta el final
        llena la memoria antes de escribir nada. Las partes se guardan en un
        directorio y se leen como si fueran un solo archivo.
        """
        if not self._cuantiles:
            return {}
        destino = self.path.with_suffix("")
        destino.mkdir(parents=True, exist_ok=True)

        cuantiles = pd.concat(self._cuantiles, ignore_index=True)
        cuantiles.to_parquet(destino / f"{part}.parquet", index=False)
        escrito = {"parte": part, "filas": len(cuantiles)}
        self._cuantiles = []

        if self._miembros:
            carpeta = Path(str(destino) + "_members")
            carpeta.mkdir(parents=True, exist_ok=True)
            pd.concat(self._miembros, ignore_index=True).to_parquet(
                carpeta / f"{part}.parquet", index=False)
            self._miembros = []
        return escrito

    def save(self) -> dict:
        """Escribe el archivo de pronósticos y, si los hay, el de miembros."""
        if not self._cuantiles:
            raise ValueError("no hay nada que guardar")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        cuantiles = pd.concat(self._cuantiles, ignore_index=True)
        cuantiles.to_parquet(self.path, index=False)
        escrito = {"forecasts": str(self.path), "filas": len(cuantiles)}

        if self._miembros:
            ruta = self.path.with_name(self.path.stem + "_members.parquet")
            pd.concat(self._miembros, ignore_index=True).to_parquet(ruta,
                                                                    index=False)
            escrito["members"] = str(ruta)
        return escrito


def load_forecasts(path: Path) -> pd.DataFrame:
    """Lee una corrida, venga en un archivo suelto o repartida en partes."""
    path = Path(path)
    if path.is_dir():
        return pd.read_parquet(path)
    carpeta = path.with_suffix("")
    if not path.exists() and carpeta.is_dir():
        return pd.read_parquet(carpeta)
    return pd.read_parquet(path)


def load_members(path: Path) -> Optional[pd.DataFrame]:
    """Lee los miembros de la corrida, si el método los produjo."""
    path = Path(path)
    suelto = path.with_name(path.stem + "_members.parquet")
    if suelto.exists():
        return pd.read_parquet(suelto)
    carpeta = Path(str(path.with_suffix("")) + "_members")
    return pd.read_parquet(carpeta) if carpeta.is_dir() else None


def member_columns(members: pd.DataFrame) -> list:
    """
    Las columnas de miembro, que son m0, m1, m2…

    Buscarlas por el prefijo «m» a secas atrapa también la columna «method»: el
    error costó una prueba en rojo y habría costado un ensamble con un miembro
    inventado.
    """
    return [c for c in members.columns if re.fullmatch(r"m\d+", c)]


def member_matrix(block: pd.DataFrame, columns: Optional[list] = None) -> np.ndarray:
    """
    Los miembros de un pronóstico como matriz (miembros, pasos), sin los ausentes.

    Un archivo puede traer métodos con distinto número de miembros: el análogo lleva 40 y
    su variante de mezcla 160, porque junta cuatro ventanas. La tabla entonces tiene
    tantas columnas de miembro como el mayor, y ausentes en el resto. Leerlas todas
    devuelve filas de NaN que envenenan cualquier cuantil que se calcule después.

    No es hipotético: por no hacer esto, cuatro de las cinco variantes salieron en NaN en
    el primer cuadro comparativo, con cobertura cero y escape de uno. El archivo estaba
    bien; quien lo leía, no.

    ``block`` trae un pronóstico, una fila por paso del horizonte, ya ordenado por paso.
    """
    columnas = columns if columns is not None else member_columns(block)
    matriz = block[columnas].to_numpy(dtype=float).T
    return matriz[~np.isnan(matriz).any(axis=1)]


def quantile_columns(forecasts: pd.DataFrame) -> tuple:
    """Devuelve las columnas de cuantil y los niveles que representan."""
    columnas = [c for c in forecasts.columns if c.startswith("q")]
    niveles = [float(c[1:]) for c in columnas]
    orden = np.argsort(niveles)
    return ([columnas[i] for i in orden], [niveles[i] for i in orden])
