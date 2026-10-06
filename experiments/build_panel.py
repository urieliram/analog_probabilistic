"""
Arma el panel horario de las zonas de carga desde los archivos del CENACE.

El reporte diario «Precios de Energía en Nodos Distribuidos del MDA» trae las 101
zonas del Sistema Interconectado Nacional, con el precio y sus tres componentes por
hora, y dentro lleva su propia hora de generación. Este extractor recorre esos
archivos, arma una serie por zona y registra el sello de cada archivo.

Escribe un CSV comprimido por zona en data/zones/ y dos índices: el de zonas y el
de sellos de publicación.
"""

import csv
import io
import re
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ARCHIVO_CENACE = Path("/home/uriel/GIT/pml/scraping/nd_mda")
DESTINO = Path(__file__).resolve().parents[1] / "data" / "zones"

COLUMNAS = ["pml", "energy", "losses", "congestion"]
HUSO = ZoneInfo("America/Mexico_City")

MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
         "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}

SELLO = re.compile(r"creado el (\d{1,2})/(\w+)/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})")
FECHA_REPORTE = re.compile(r"Fecha:\s*(\d{1,2})/(\w+)/(\d{4})")


def _fecha(dia: str, mes: str, anio: str) -> datetime:
    return datetime(int(anio), MESES[mes.lower()[:3]], int(dia))


def leer_archivo(ruta: Path):
    """
    Devuelve (fecha de entrega, sello de generación, filas) de un ZIP diario.

    Las filas son (hora, zona, pml, energy, losses, congestion). La hora que trae
    el CENACE va de 1 a 24 y aquí se convierte a inicio de intervalo: la hora 1 es
    las 00:00.
    """
    with zipfile.ZipFile(ruta) as paquete:
        nombre = paquete.namelist()[0]
        texto = paquete.read(nombre).decode("latin-1")

    lineas = texto.splitlines()
    cabecera = "\n".join(lineas[:8])

    m = FECHA_REPORTE.search(cabecera)
    fecha = _fecha(*m.groups()) if m else None

    m = SELLO.search(cabecera)
    sello = None
    if m:
        dia, mes, anio, hh, mm, ss = m.groups()
        sello = _fecha(dia, mes, anio).replace(
            hour=int(hh), minute=int(mm), second=int(ss))

    filas = []
    for fila in csv.reader(io.StringIO("\n".join(lineas[8:]))):
        if len(fila) < 6:
            continue
        try:
            hora = int(fila[0])
            valores = [float(fila[i]) for i in range(2, 6)]
        except ValueError:
            continue
        filas.append((hora, fila[1].strip(), *valores))

    return fecha, sello, filas


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    archivos = sorted(ARCHIVO_CENACE.glob("*/*/*.zip"))
    print(f"{len(archivos)} archivos en {ARCHIVO_CENACE}", flush=True)

    por_zona: dict[str, list] = {}
    sellos = []
    repetidas = []
    saltados = []

    for numero, ruta in enumerate(archivos, 1):
        try:
            fecha, sello, filas = leer_archivo(ruta)
        except (zipfile.BadZipFile, IndexError, UnicodeDecodeError) as error:
            saltados.append((ruta.name, type(error).__name__))
            continue
        if fecha is None or not filas:
            saltados.append((ruta.name, "sin fecha o sin filas"))
            continue

        sellos.append({"archivo": ruta.name, "entrega": fecha, "generado": sello})
        ## El índice va en hora civil local, que es como habla el mercado: el día
        ## de operación tiene sus horas numeradas de 1 a 24.
        ##
        ## Los cinco últimos domingos de octubre de 2018 a 2022 —México tuvo
        ## horario de verano hasta finales de 2022— el reporte trae 25 horas,
        ## porque el reloj se atrasa y una hora se repite. Esa hora 25 se descarta:
        ## si se dejara, caería sobre las 00:00 del día siguiente y machacaría un
        ## dato real. Los domingos de abril, en cambio, el día tiene 23 horas y la
        ## rejilla simplemente no trae esa hora.
        for hora, zona, *valores in filas:
            if hora > 24:
                repetidas.append((fecha.date(), zona))
                continue
            marca = fecha + pd.Timedelta(hours=hora - 1)
            por_zona.setdefault(zona, []).append((marca, *valores))

        if numero % 500 == 0:
            print(f"  {numero}/{len(archivos)}  {len(por_zona)} zonas", flush=True)

    indice = []
    duplicados_totales = 0
    for zona, filas in sorted(por_zona.items()):
        tabla = pd.DataFrame(filas, columns=["ds", *COLUMNAS])
        duplicados_totales += int(tabla.ds.duplicated().sum())
        tabla = tabla.drop_duplicates("ds").sort_values("ds").set_index("ds")
        nombre = re.sub(r"[^a-z0-9]+", "_", zona.lower()).strip("_")
        tabla.to_csv(DESTINO / f"{nombre}.csv.gz", float_format="%.2f",
                     compression="gzip")
        indice.append({"zona": zona, "archivo": f"{nombre}.csv.gz",
                       "horas": len(tabla), "desde": tabla.index.min(),
                       "hasta": tabla.index.max()})

    pd.DataFrame(indice).to_csv(DESTINO.parent / "zones_index.csv", index=False)
    pd.DataFrame(sellos).to_csv(DESTINO.parent / "publication_stamps.csv",
                                index=False)

    print(f"\n{len(indice)} zonas escritas en {DESTINO}")
    print(f"sellos de publicación: {len(sellos)} archivos")
    print(f"marcas duplicadas descartadas: {duplicados_totales}")
    dias = sorted({d for d, _ in repetidas})
    print(f"horas repetidas del cambio de horario descartadas: {len(repetidas)} "
          f"en {len(dias)} días: {dias}")
    if saltados:
        fraccion = len(saltados) / len(archivos)
        print(f"archivos saltados: {len(saltados)} de {len(archivos)} "
              f"({100 * fraccion:.1f}%)  {saltados[:5]}")
        if fraccion > 0.01:
            raise SystemExit(
                "se saltó más del uno por ciento de los archivos: revisa el "
                "formato antes de usar este panel")


if __name__ == "__main__":
    main()
