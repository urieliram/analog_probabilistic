"""
Arma la carpeta de resultados para que un árbitro pueda auditarla sin preguntarnos nada.

La regla es una: **cada cuadro del artículo tiene que poder rastrearse hasta el dato que
lo sostiene y hasta el guión que lo produjo.** Un cuadro sin su detalle es una afirmación
sin evidencia, y un detalle sin su etiqueta es un archivo que nadie sabe leer.

Lo que escribe:

    publicacion/
      LEEME.md               el índice: qué cuadro sale de qué archivo y con qué guión
      DICCIONARIO.csv        qué significa cada columna y en qué unidad está
      HUELLAS.csv            el SHA256 de cada archivo, para verificar que no cambió
      cuadro_NN_*.csv        los cuadros del artículo, uno por archivo
      detalle/*.csv.gz       el dato por pronóstico y por día detrás de cada cuadro

Se regenera entero con una corrida, de modo que no puede quedar desfasado respecto de
los resultados. Si un número del artículo no aparece aquí, el número no existe.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import COMMON_LEVELS, RESULTS  # noqa: E402

DESTINO = RESULTS / "publicacion"
DETALLE = DESTINO / "detalle"

## El modelo y el tratamiento van en COLUMNAS SEPARADAS, no pegados en un nombre
## compuesto. Un cuadro del artículo tiene como sujetos de estudio a los modelos tal
## cual se llaman; «la familia de errores» es taquigrafía nuestra y no es un sujeto.
## Separándolos, el lector ve dos hechos independientes —qué modelo y qué se le hizo— y
## puede agrupar por cualquiera de los dos sin interpretar cadenas de texto.
ESCENARIOS = {
    "propios": "el modelo entrega sus propios escenarios",
    "de sus errores": "escenarios construidos con su mediana más sus errores pasados",
    "sólo cuantiles": "el modelo no entrega escenarios y no se le construyeron",
}

CALIBRACION = {
    "sin calibrar": "los bordes que el modelo entrega de fábrica",
    "split": "conformal por división: una corrección constante de un bloque inicial",
    "enbpi": "conformal con ventana rodante de los últimos orígenes observados",
    "normalizado": "como enbpi, con el error dividido por la dispersión de ese día",
    "asimetrico": "normalizado, con un factor por cada lado de la mediana",
}


def etiqueta(nombre: str) -> tuple:
    """
    Parte el nombre interno en modelo, escenarios y calibración.

    Los nombres internos pegan las tres cosas —«Analog-mezcla+errores+enbpi»— porque es
    cómodo para el código. Para publicar hacen falta separadas.
    """
    capa = "sin calibrar"
    partes = nombre.split("+")
    modelo = partes[0]
    escenarios = "propios"
    for resto in partes[1:]:
        if resto == "errores":
            escenarios = "de sus errores"
        elif resto in CALIBRACION:
            capa = resto
    return modelo, escenarios, capa


## Qué significa cada columna. Sin esto, un archivo de puntajes es ilegible para alguien
## que no escribió el código, y eso incluye a nosotros en seis meses.
DICCIONARIO = [
    ("modelo", "texto", "el sujeto de estudio, con el nombre con que se publica"),
    ("escenarios", "texto", "de dónde salen sus escenarios: propios del modelo, "
     "construidos con su mediana más sus errores pasados, o ninguno"),
    ("calibracion", "texto", "qué capa conformal se le aplicó, o sin calibrar"),
    ("method", "texto", "el método, y la capa de calibración tras un signo de más"),
    ("zone", "texto", "zona de carga del mercado mexicano, nombre público"),
    ("origin", "fecha y hora", "última hora observada; el horizonte son las 24 siguientes"),
    ("pinball_comun", "pesos por MWh", "pérdida pinball media sobre los NUEVE niveles que "
     "TODOS los métodos emiten, de 0.1 a 0.9. Es la columna que compara entre métodos"),
    ("mean_pinball", "pesos por MWh", "pérdida pinball sobre la rejilla PROPIA de cada "
     "método; NO es comparable entre métodos de rejillas distintas"),
    ("crps", "pesos por MWh", "CRPS del ensamble en su forma justa, denominador m(m-1); "
     "ausente en los métodos que no entregan escenarios"),
    ("cover80", "fracción de 0 a 1", "veces que el precio cayó dentro del intervalo "
     "central de 80%, que es el más ancho que TODOS emiten de fábrica. Se prometió 0.80"),
    ("cover90", "fracción de 0 a 1", "lo mismo para el intervalo de 90%; ausente en los "
     "métodos cuya rejilla se detiene en 0.9"),
    ("sharp80", "pesos por MWh", "ancho medio del intervalo de 80%. Menos es mejor, pero "
     "sólo a igualdad de cobertura"),
    ("winkler80", "pesos por MWh", "puntaje de intervalo de Winkler al 80%"),
    ("mae", "pesos por MWh", "error absoluto de la MEDIANA; no depende de la calibración"),
    ("reliability", "fracción de 0 a 1", "desvío medio entre cobertura prometida y "
     "entregada sobre todos los niveles. Menos es mejor"),
    ("hit_0.05 … hit_0.95", "fracción de 0 a 1", "veces que el precio quedó POR DEBAJO de "
     "ese cuantil; si el método está calibrado, hit_a debería valer a"),
    ("costo_real", "pesos", "la cuenta del día: suma de los 24 precios por el perfil de "
     "carga, que aquí es plano, una unidad por hora"),
    ("faltante", "pesos", "cuánto se quedó CORTO el techo anunciado respecto de la cuenta "
     "real; cero si la cuenta cayó dentro. Es el dinero no presupuestado"),
    ("dentro_90", "sí o no", "la cuenta real cayó dentro del intervalo de 90% de la cuenta"),
    ("decil", "entero 1 a 10", "decil de la cuenta REAL del día, con cortes comunes a "
     "todos los métodos; el 10 son los días más caros"),
    ("corr_por_camino", "correlación", "cuánto persiste de una hora a la siguiente lo que "
     "aparta a cada escenario del promedio del ensamble. Alto = cada escenario es un día "
     "posible; cero = son sorteos independientes por hora"),
    ("testigo_corr_global", "correlación", "lo mismo sobre un ensamble construido "
     "permutando cada hora por separado: mismas distribuciones por hora, dependencia "
     "destruida. Es el punto de comparación"),
    ("k_real", "entero", "análogos encontrados de verdad, que puede ser menor que los "
     "pedidos cuando la regla de separación no deja caber tantos"),
    ("niveles_propios", "entero", "cuántos niveles de cuantil emite ese método de fábrica"),
]


def huella(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for trozo in iter(lambda: f.read(1 << 20), b""):
            h.update(trozo)
    return h.hexdigest()


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=RESULTS.parent).stdout.strip()[:12]
    except Exception:
        return "desconocido"


def escribe(nombre: str, tabla: pd.DataFrame, indice: bool = True) -> dict:
    ruta = DESTINO / nombre
    tabla.round(6).to_csv(ruta, index=indice)
    return {"archivo": nombre, "filas": len(tabla), "sha256": huella(ruta)}


def comprime(nombre: str, tabla: pd.DataFrame) -> dict:
    ruta = DETALLE / nombre
    tabla.to_csv(ruta, index=False, compression="gzip")
    return {"archivo": f"detalle/{nombre}", "filas": len(tabla), "sha256": huella(ruta)}


def carga_conformal() -> pd.DataFrame:
    """Todos los puntajes por pronóstico, de todas las corridas calibradas."""
    partes = []
    for ruta in sorted(RESULTS.glob("*_conformal.csv")):
        t = pd.read_csv(ruta, parse_dates=["origin"])
        t["corrida"] = ruta.name.replace("_conformal.csv", "")
        partes.append(t)
    if not partes:
        raise SystemExit("no hay corridas calibradas")
    t = pd.concat(partes, ignore_index=True)
    return t.drop_duplicates(subset=["zone", "origin", "method"])


def main() -> None:
    DETALLE.mkdir(parents=True, exist_ok=True)
    registro = []

    ## ---- el dato por pronóstico, que sostiene todos los cuadros horarios ----
    puntajes = carga_conformal()
    etiquetas = puntajes.method.map(etiqueta)
    puntajes["modelo"] = [e[0] for e in etiquetas]
    puntajes["escenarios"] = [e[1] for e in etiquetas]
    puntajes["calibracion"] = [e[2] for e in etiquetas]
    ## se conservan los nombres viejos para no romper lo que ya los usa
    puntajes["capa"] = puntajes["calibracion"]
    puntajes["metodo_base"] = puntajes["modelo"]
    registro.append(comprime("puntajes_por_pronostico.csv.gz", puntajes))

    columnas = ["pinball_comun", "crps", "cover80", "sharp80", "winkler80",
                "cover90", "sharp90", "mae", "reliability"]
    columnas = [c for c in columnas if c in puntajes.columns]

    ## ---- cuadro 1: el banco en crudo ----
    crudo = puntajes[puntajes.calibracion == "sin calibrar"]
    registro.append(escribe("cuadro_01_banco_sin_calibrar.csv",
                            crudo.groupby(["modelo", "escenarios"])[columnas].mean()
                            .sort_values("pinball_comun")))

    ## ---- cuadro 2: el banco calibrado, una fila por método y capa ----
    registro.append(escribe("cuadro_02_banco_calibrado.csv",
                            puntajes.groupby(["modelo", "escenarios",
                                               "calibracion"])[columnas].mean()))

    ## ---- cuadro 3: filo sujeto a calibración ----
    g = puntajes.groupby(["modelo", "escenarios", "calibracion"])[columnas].mean()
    cumplen = g[(g.cover80 >= 0.78) & (g.cover80 <= 0.82)].sort_values("pinball_comun")
    registro.append(escribe("cuadro_03_filo_a_igual_cobertura.csv", cumplen))

    ## ---- cuadro 4: por año ----
    puntajes["anio"] = puntajes.origin.dt.year
    registro.append(escribe("cuadro_04_por_anio.csv",
                            puntajes.groupby(["modelo", "escenarios", "anio"])
                            [columnas].mean()))

    ## ---- cuadro 5: por zona ----
    registro.append(escribe("cuadro_05_por_zona.csv",
                            puntajes.groupby(["modelo", "escenarios", "zone"])
                            [columnas].mean()))

    ## ---- la cuenta del día, si ya se midió ----
    por_dia = RESULTS / "costo_diario_por_dia.parquet"
    if por_dia.exists():
        d = pd.read_parquet(por_dia)
        cortes = d.costo_real.quantile(np.arange(0, 1.01, 0.1)).to_numpy()
        d["decil"] = np.clip(np.searchsorted(cortes[1:-1], d.costo_real) + 1, 1, 10)
        et = d.metodo.map(etiqueta)
        d["modelo"] = [e[0] for e in et]
        d["escenarios"] = [e[1] for e in et]
        registro.append(comprime("costo_por_dia.csv.gz", d))
        agr = {"dias": ("costo_real", "size"), "cubre_90": ("dentro_90", "mean"),
               "falta_pesos": ("faltante", "mean"), "cuenta": ("costo_real", "mean"),
               "ancho_90": ("ancho_90", "mean")}
        registro.append(escribe("cuadro_06_cuenta_del_dia.csv",
                                d.groupby(["modelo", "escenarios"]).agg(**agr)
                                .sort_values("falta_pesos")))
        registro.append(escribe("cuadro_07_cuenta_por_decil.csv",
                                d.groupby(["modelo", "escenarios", "decil"])
                                .agg(**agr)))

    ## ---- coherencia de trayectorias ----
    coh = RESULTS / "coherencia_trayectorias.csv"
    if coh.exists():
        registro.append(escribe("cuadro_08_coherencia_trayectorias.csv",
                                pd.read_csv(coh).set_index("metodo")))

    ## ---- la malla de selección ----
    for n, f in [("cuadro_09_malla_tramo_pandemia.csv", "grid_2020-04_resumen.csv"),
                 ("cuadro_10_malla_tramo_limpio.csv", "grid_2022-04_resumen.csv")]:
        if (RESULTS / f).exists():
            registro.append(escribe(n, pd.read_csv(RESULTS / f), indice=False))

    ## ---- diccionario, huellas e índice ----
    pd.DataFrame(DICCIONARIO, columns=["columna", "unidad", "qué es"]).to_csv(
        DESTINO / "DICCIONARIO.csv", index=False)
    pd.DataFrame(registro).to_csv(DESTINO / "HUELLAS.csv", index=False)

    lineas = [
        "# Resultados del artículo, para auditoría",
        "",
        f"Generado con el commit `{commit()}` por `experiments/publicar_resultados.py`.",
        "Se regenera entero con una corrida, así que no puede quedar desfasado.",
        "**Si un número del artículo no aparece aquí, el número no existe.**",
        "",
        "## Cómo leerlo",
        "",
        "- `DICCIONARIO.csv` dice qué es cada columna y en qué unidad está. Dos avisos que",
        "  importan: `pinball_comun` es la columna que compara entre métodos, porque se",
        "  calcula sobre los nueve niveles que todos emiten; `mean_pinball` se calcula",
        "  sobre la rejilla propia de cada método y **no** es comparable entre métodos de",
        "  rejillas distintas.",
        "- `HUELLAS.csv` trae el SHA256 y el número de filas de cada archivo.",
        "- `detalle/` trae el dato por pronóstico y por día. Todo cuadro de aquí se puede",
        "  recalcular desde ahí.",
        "",
        "## Qué hay",
        "",
    ]
    for fila in registro:
        lineas.append(f"- `{fila['archivo']}` — {fila['filas']:,d} filas")
    lineas += [
        "",
        "## Convenciones de los cuadros",
        "",
        "| columna | dirección |",
        "|---|---|",
        "| `pinball_comun`, `crps`, `mae`, `reliability`, `winkler80` | menos es mejor |",
        "| `cover80`, `cover90` | más cerca de lo prometido es mejor, no más alto |",
        "| `sharp80` | menos es mejor, **pero sólo a igualdad de cobertura** |",
        "| `falta_pesos` | menos es mejor |",
        "| `corr_por_camino` | más es mejor |",
        "",
        "La cobertura prometida es 0.80 en el intervalo común y 0.90 en el de la cuenta",
        "del día. Un método que cubre 0.93 no es mejor que uno que cubre 0.80: está mal",
        "calibrado del lado caro, porque paga filo por una seguridad que no prometió.",
        "",
        "## Qué NO está aquí, y por qué",
        "",
        "- Los pronósticos crudos y sus escenarios pesan decenas de gigabytes y se",
        "  regeneran con los guiones de `experiments/`. Van a un depósito con",
        "  identificador permanente, no al repositorio.",
        "- Los pesos de los modelos de fundación no se distribuyen aquí: los de Moirai son",
        "  no comerciales y los de TimesFM 3.0 prohíben además redistribuirlos.",
    ]
    (DESTINO / "LEEME.md").write_text("\n".join(lineas) + "\n")

    print(f"escrito en {DESTINO}")
    for fila in registro:
        print(f"  {fila['archivo']:48s} {fila['filas']:>10,d} filas")


if __name__ == "__main__":
    main()
