"""
Calibrar los escenarios ANTES de sumar la cuenta del día, y probar un colchón que crece
con el nivel del precio.

Por qué existe este módulo
--------------------------
La capa de ``conformal.py`` califica las horas, pero no guarda los escenarios
calibrados. La cuenta del día de ``costo_diario.py`` y ``curva_tau.py`` se calculó con
los escenarios CRUDOS: el análogo sin calibrar contra escenarios de errores que se
calibran solos con su ventana de sesenta días, y Moirai sin calibrar. Eso no es trato
igual. Aquí todos los métodos con escenarios pasan por las mismas variantes de
calibración, y la cuenta del día se calcula con los escenarios ya calibrados.

Las cuatro variantes
--------------------
Todas estiran o encogen los escenarios alrededor de su mediana, con un factor por hora
y por lado, de modo que las trayectorias sobreviven. Todas miran sólo los últimos 100
días ya observados al decidir (los precios del día en curso se publicaron la tarde
anterior).

``crudo``    sin calibrar.
``proporcional``  el ajuste lineal: un factor por hora del día y por lado que
             multiplica el abanico propio del método, el mismo factor para cualquier
             nivel de precio. Para alcanzar los picos ensancha también la madrugada.
             El puntaje es el exceso dividido por la media anchura que el ensamble
             traía ese día, de modo que el factor sale directo. (Se probó primero la
             variante ``enbpi`` de ``conformal.py`` tal cual, y cubría 0.93 a 0.95
             cuando prometía 0.90, porque normaliza por la desviación estándar del
             ensamble y no por su media anchura; no era un rival justo.)
``raiz``     el colchón del vendedor de periódicos: el ancho es Q por la raíz del nivel
             pronosticado, ``w = Q * sqrt(max(mediana, 100))``, con un Q por lado. Es
             la fórmula ``gamma * sqrt(pronóstico)``; gamma no se fija en 2 porque eso
             vale para conteos (Poisson, sigma = raíz de la media) y no para pesos: aquí
             Q se elige conformalmente para que el pasado cumpla lo prometido.
``nivel``    el ancho sale de los propios datos, por tramo de nivel: los puntos de los
             últimos 100 días se parten en diez tramos según la mediana pronosticada, y
             en cada tramo se toma el cuantil conformal del exceso. No supone ninguna
             forma; si el error crece como raíz abajo y más que lineal arriba, como se
             midió, este tramo por tramo lo captura.

``nivel_bloque``  como ``nivel``, pero por bloque de horas del reloj (0-5, 6-11, 12-17,
             18-23) y con cinco tramos de nivel dentro de cada bloque: unos 120 puntos
             del pasado por celda. Se agregó después de ver que el nivel solo cubre de
             más en la madrugada (0.95) y de menos en el pico de la tarde (0.83): a un
             mismo nivel, la tarde se equivoca más.

El piso de 100 pesos en ``raiz`` evita la raíz de precios negativos o casi cero. Los
días con menos de 30 días de historia se omiten en TODAS las variantes, para que las
comparaciones sean sobre los mismos días.

Hipótesis declaradas antes de correr
------------------------------------
H1. Con ``raiz`` o ``nivel``, el ancho horario en las horas de precio bajo es menor que
    con ``proporcional``, con cobertura horaria parecida.
H2. ``raiz`` o ``nivel`` mejoran la CRPS ponderada de la cuenta del día respecto de
    ``proporcional`` (pesos uniforme y cola alta), método por método.
H3. Después de un día conocido caro, la cobertura de la cuenta de los escenarios de
    errores se acerca a 0.90 con ``nivel``.
H4. (declarada después de H1-H3, antes de correr ``nivel_bloque``) ``nivel_bloque``
    reduce la diferencia de cobertura horaria entre la madrugada y la tarde respecto de
    ``nivel``, y mejora la CRPS ponderada de la cuenta del día respecto de ``nivel``.
Comparación principal entre métodos: Analog-mezcla contra t0-beta+errores, LEAR y
PatchTST-FM+errores, cada uno con la MISMA variante.

Uso:
    python experiments/calibra_escenarios.py calcula
    python experiments/calibra_escenarios.py analiza
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.conformal import (ALFA, MINIMO, TOPE_FACTOR,  # noqa: E402
                                   VENTANA_RODANTE, cuantil_conformal, escala)
from experiments.forecast_store import member_columns, member_matrix  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import pinball  # noqa: E402

NIVELES = np.round(np.arange(0.05, 0.96, 0.05), 2)
VARIANTES = ("crudo", "proporcional", "raiz", "nivel", "nivel_bloque")
BLOQUES = (range(0, 6), range(6, 12), range(12, 18), range(18, 24))
TRAMOS_BLOQUE = 5
PISO = 100.0
TRAMOS = 10
HISTORIA_MINIMA = 30
DETALLE = RESULTS / "calibrados_por_dia.parquet"
POR_HORA = RESULTS / "calibrados_por_hora.csv"
POR_NIVEL = RESULTS / "calibrados_por_nivel.csv"

## qué métodos se toman de cada corrida; "Analog" está en dos corridas y se toma una vez
CORRIDAS = {
    "picos_prueba": None, "forecasts_prueba": ["Analog-minimos-cuadrados"],
    "memoria_larga": None, "euclidiana": None,
    "bench_prueba": None, "lear_prueba": None, "fundacion_Moirai": None,
    "err_picos": None, "err_lear": None, "err_t0-beta": None,
    "err_PatchTST-FM": None, "err_TiRex": None, "err_TimesFM-2.5": None,
    "err_Moirai": None,
    ## MOMENT (err_MOMENT) se probó el 2026-10-08 y se dejó fuera del artículo por
    ## decisión de los autores: quedó 20 de 23, empatado con el ingenuo diario.
}


def _por_lado(exceso_pasado, escala_pasada, alfa):
    """Cuantil conformal del exceso hacia arriba y hacia abajo, cada uno a alfa/2."""
    arriba = np.where(exceso_pasado > 0, exceso_pasado / escala_pasada, 0.0)
    abajo = np.where(exceso_pasado < 0, -exceso_pasado / escala_pasada, 0.0)
    return (cuantil_conformal(arriba.ravel(), alfa / 2),
            cuantil_conformal(abajo.ravel(), alfa / 2))


def _por_tramos(med_pasado, exceso_pasado, med_hoy, tramos):
    """Ancho por lado para cada hora de hoy, según el tramo de nivel en que cae."""
    cortes = np.quantile(med_pasado, np.linspace(0, 1, tramos + 1)[1:-1])
    tramo_pasado = np.searchsorted(cortes, med_pasado)
    tramo_hoy = np.searchsorted(cortes, med_hoy)
    w_arr, w_aba = np.empty(len(med_hoy)), np.empty(len(med_hoy))
    for t in np.unique(tramo_hoy):
        q_a, q_b = _por_lado(exceso_pasado[tramo_pasado == t], 1.0, ALFA)
        w_arr[tramo_hoy == t], w_aba[tramo_hoy == t] = q_a, q_b
    return w_arr, w_aba


def factores(variante, H, med, arr, aba, disp):
    """Factor por hora y por lado que lleva el ensamble de hoy a la anchura objetivo."""
    exceso = H["obs"] - H["med"]
    if variante == "proporcional":
        ## un factor por hora (eje 0 = días): el exceso en unidades de la media
        ## anchura que el ensamble anunciaba ese día, por lado
        arriba = np.where(exceso > 0, exceso / np.maximum(H["arr"], MINIMO), 0.0)
        abajo = np.where(exceso < 0, -exceso / np.maximum(H["aba"], MINIMO), 0.0)
        return (cuantil_conformal(arriba, ALFA / 2), cuantil_conformal(abajo, ALFA / 2))
    if variante == "raiz":
        q_arr, q_aba = _por_lado(exceso, np.sqrt(np.maximum(H["med"], PISO)), ALFA)
        g = np.sqrt(np.maximum(med, PISO))
        w_arr, w_aba = q_arr * g, q_aba * g
    elif variante == "nivel":
        w_arr, w_aba = _por_tramos(H["med"].ravel(), exceso.ravel(), med, TRAMOS)
    elif variante == "nivel_bloque":
        w_arr, w_aba = np.empty(len(med)), np.empty(len(med))
        for bloque in BLOQUES:
            h = list(bloque)
            w_arr[h], w_aba[h] = _por_tramos(H["med"][:, h].ravel(),
                                             exceso[:, h].ravel(), med[h],
                                             TRAMOS_BLOQUE)
    else:
        raise ValueError(variante)
    return w_arr / np.maximum(arr, MINIMO), w_aba / np.maximum(aba, MINIMO)


def _una_tarea(args):
    corrida, parte, metodos, cortes_nivel = args
    miembros = pd.read_parquet(parte)
    if metodos:
        miembros = miembros[miembros.method.isin(metodos)]
    obs = pd.read_parquet(RESULTS / corrida / parte.name,
                          columns=["zone", "origin", "step", "observed"])
    obs = obs.drop_duplicates(["zone", "origin", "step"])
    observados = {(z, o): b.sort_values("step")["observed"].to_numpy()
                  for (z, o), b in obs.groupby(["zone", "origin"])}
    columnas = member_columns(miembros)
    filas, horas, tramos = [], [], []
    for metodo, bm in miembros.groupby("method"):
        H = {k: [] for k in ("obs", "med", "arr", "aba")}
        acum_h = {v: np.zeros((24, 5)) for v in VARIANTES}
        acum_n = {v: np.zeros((len(cortes_nivel) + 1, 3)) for v in VARIANTES}
        for origen, b in sorted(bm.groupby("origin"), key=lambda kv: kv[0]):
            zona = b.zone.iloc[0]
            y = observados.get((zona, origen))
            E = member_matrix(b.sort_values("step"), columnas)
            if y is None or len(y) != 24 or E.shape[1] != 24 or len(E) < 2:
                continue
            med = np.median(E, axis=0)
            q05, q95 = np.quantile(E, [0.05, 0.95], axis=0)
            arr, aba, disp = q95 - med, med - q05, E.std(axis=0)
            if len(H["obs"]) >= HISTORIA_MINIMA:
                HH = {k: np.array(v[-VENTANA_RODANTE:]) for k, v in H.items()}
                real = float(y.sum())
                for variante in VARIANTES:
                    if variante == "crudo":
                        nuevo, acotados = E, 0
                    else:
                        f_a, f_b = factores(variante, HH, med, arr, aba, disp)
                        if not (np.all(np.isfinite(f_a)) and np.all(np.isfinite(f_b))):
                            continue
                        acotados = int(np.sum(f_a > TOPE_FACTOR) + np.sum(f_b > TOPE_FACTOR))
                        nuevo = escala(E, med, np.minimum(f_a, TOPE_FACTOR),
                                       np.minimum(f_b, TOPE_FACTOR))
                    costos = nuevo.sum(axis=1)
                    cq = np.quantile(costos, NIVELES)
                    h05, h50, h95 = np.quantile(nuevo, [0.05, 0.5, 0.95], axis=0)
                    dentro = (y >= h05) & (y <= h95)
                    fila = {"metodo": metodo, "variante": variante, "zona": zona,
                            "origen": origen, "costo_real": real,
                            "n_escenarios": len(E), "acotados": acotados,
                            "cq_0.05": cq[0], "cq_0.95": cq[-1],
                            "cob_hora_90": dentro.mean(),
                            "ancho_hora_90": float((h95 - h05).mean()),
                            "pin_hora_0.05": float(pinball(y, h05, 0.05).mean()),
                            "pin_hora_0.50": float(pinball(y, h50, 0.5).mean()),
                            "pin_hora_0.95": float(pinball(y, h95, 0.95).mean())}
                    for q, a in zip(cq, NIVELES):
                        fila[f"pin_{a:.2f}"] = float(pinball(np.array([real]),
                                                              np.array([q]), a)[0])
                    filas.append(fila)
                    acum_h[variante] += np.column_stack(
                        [dentro, h95 - h05, pinball(y, h95, 0.95), pinball(y, h05, 0.05),
                         np.ones(24)])
                    tr = np.searchsorted(cortes_nivel, h50)
                    np.add.at(acum_n[variante], tr,
                              np.column_stack([dentro, h95 - h05, np.ones(24)]))
            H["obs"].append(y)
            H["med"].append(med)
            H["arr"].append(arr)
            H["aba"].append(aba)
        for v in VARIANTES:
            for h in range(24):
                s = acum_h[v][h]
                horas.append({"metodo": metodo, "variante": v, "zona": parte.stem,
                              "hora": h, "cubre": s[0], "ancho": s[1], "pin95": s[2],
                              "pin05": s[3], "n": s[4]})
            for t in range(len(cortes_nivel) + 1):
                s = acum_n[v][t]
                tramos.append({"metodo": metodo, "variante": v, "zona": parte.stem,
                               "tramo": t, "cubre": s[0], "ancho": s[1], "n": s[2]})
    return filas, horas, tramos, f"{corrida}/{parte.stem}"


def calcula(solo_una_zona: bool = False) -> None:
    ## cortes comunes de nivel para el desglose por hora: deciles del precio observado
    obs = pd.concat(pd.read_parquet(f, columns=["observed"])
                    for f in sorted((RESULTS / "lear_prueba").glob("*.parquet")))
    cortes_nivel = np.quantile(obs.observed, np.linspace(0, 1, 11)[1:-1])
    tareas = []
    for corrida, metodos in CORRIDAS.items():
        partes = sorted((RESULTS / f"{corrida}_members").glob("*.parquet"))
        if solo_una_zona:
            partes = partes[:1]
        tareas += [(corrida, p, metodos, cortes_nivel) for p in partes]
    ## las tareas pesadas primero, para que no queden solas al final
    tareas.sort(key=lambda t: -t[1].stat().st_size)
    print(f"{len(tareas)} tareas", flush=True)
    filas, horas, tramos = [], [], []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (f, h, t, nombre) in enumerate(pool.map(_una_tarea, tareas), 1):
            filas += f
            horas += h
            tramos += t
            if i % 20 == 0 or i == len(tareas):
                print(f"  {i}/{len(tareas)} {nombre}", flush=True)
    sufijo = "_prueba" if solo_una_zona else ""
    pd.DataFrame(filas).to_parquet(str(DETALLE).replace(".parquet", f"{sufijo}.parquet"),
                                   index=False)
    pd.DataFrame(horas).to_csv(str(POR_HORA).replace(".csv", f"{sufijo}.csv"), index=False)
    t = pd.DataFrame(tramos)
    t.attrs["cortes"] = cortes_nivel.tolist()
    t.to_csv(str(POR_NIVEL).replace(".csv", f"{sufijo}.csv"), index=False)
    pd.Series(cortes_nivel).to_csv(RESULTS / f"calibrados_cortes_nivel{sufijo}.csv",
                                   index=False, header=["corte"])
    print(f"{len(filas):,d} filas por día", flush=True)


## ---------------------------------------------------------------------------
## Análisis: las hipótesis declaradas arriba, en el orden declarado
## ---------------------------------------------------------------------------

CLAVE = ["Analog-mezcla", "Analog", "t0-beta+errores", "PatchTST-FM+errores", "LEAR",
         "Moirai", "Moirai+errores"]
RIVALES = ["LEAR", "t0-beta+errores", "PatchTST-FM+errores"]
CALIBRADAS = ("proporcional", "raiz", "nivel", "nivel_bloque")


def analiza() -> None:
    from experiments.curva_tau import PESOS, compara
    from experiments.dilema import con_dia_anterior, deciles

    pub = RESULTS / "publicacion"
    pub.mkdir(exist_ok=True)
    t = pd.read_parquet(DETALLE)
    t["cob_cuenta"] = (t.costo_real >= t["cq_0.05"]) & (t.costo_real <= t["cq_0.95"])
    pines = t[[f"pin_{a:.2f}" for a in NIVELES]].to_numpy()
    for nombre, w in PESOS.items():
        t[f"qw_{nombre}"] = 2 * (pines * w(NIVELES)).mean(axis=1)
    t["etiqueta"] = t.metodo + " | " + t.variante
    x = t.rename(columns={"metodo": "metodo_base"}).rename(columns={"etiqueta": "metodo"})

    print("=== 0. calibración: lo prometido es 0.90 por hora y por cuenta del día ===")
    cal = t.groupby(["metodo", "variante"]).agg(
        dias=("origen", "size"), cob_hora=("cob_hora_90", "mean"),
        ancho_hora=("ancho_hora_90", "mean"), cob_cuenta=("cob_cuenta", "mean"),
        qw_uniforme=("qw_uniforme", "mean"), qw_cola_alta=("qw_cola_alta", "mean"),
        pin_0_95=("pin_0.95", "mean"), pin_0_50=("pin_0.50", "mean"),
        pin_0_05=("pin_0.05", "mean"))
    cal.round(3).to_csv(pub / "cuadro_calibrados.csv")
    print(cal.round(3).to_string())

    print("\n=== H1. ancho y cobertura por hora del día y por tramo de nivel ===")
    h = pd.read_csv(POR_HORA).groupby(["metodo", "variante", "hora"])[
        ["cubre", "ancho", "pin95", "pin05", "n"]].sum()
    for c in ["cubre", "ancho", "pin95", "pin05"]:
        h[c] = h[c] / h.n
    h.round(3).to_csv(pub / "cuadro_calibrados_por_hora.csv")
    n = pd.read_csv(POR_NIVEL).groupby(["metodo", "variante", "tramo"])[
        ["cubre", "ancho", "n"]].sum()
    n["cubre"], n["ancho"] = n.cubre / n.n, n.ancho / n.n
    n.round(3).to_csv(pub / "cuadro_calibrados_por_nivel.csv")
    for m in ["Analog-mezcla", "t0-beta+errores", "LEAR"]:
        print(f"\n--- {m}: ancho del intervalo horario de 90% (cobertura) por tramo de"
              " nivel pronosticado, tramo 0 = más barato ---")
        a = n.loc[m]
        tabla = pd.DataFrame({v: [f"{r.ancho:6.0f} ({r.cubre:.2f})" for r in
                                  a.loc[v].itertuples()] for v in ("crudo",) + CALIBRADAS})
        print(tabla.to_string())
        print(f"--- {m}: por hora del día ---")
        a = h.loc[m]
        tabla = pd.DataFrame({v: [f"{r.ancho:6.0f} ({r.cubre:.2f})" for r in
                                  a.loc[v].itertuples()] for v in ("crudo",) + CALIBRADAS})
        print(tabla.iloc[[1, 3, 5, 8, 12, 16, 18, 19, 20, 21, 22]].to_string())

    print("\n=== H2. dentro de cada método: raiz y nivel contra proporcional ===")
    print("(por ciento de proporcional; negativo = mejor que el ajuste lineal)")
    filas = []
    for m in sorted(t.metodo.unique()):
        for v in ("raiz", "nivel", "nivel_bloque"):
            for medida in ["qw_uniforme", "qw_cola_alta", "qw_cola_baja", "pin_0.95",
                           "pin_0.05"]:
                r = compara(x, f"{m} | {v}", f"{m} | proporcional", medida)
                r.update({"metodo_base": m, "variante": v})
                filas.append(r)
    h2 = pd.DataFrame(filas)
    h2.round(4).to_csv(pub / "cuadro_calibrados_h2.csv", index=False)
    for m, g in h2.groupby("metodo_base"):
        celdas = "  ".join(f"{r.variante}/{r.medida[:12]}: {r.pct:+.1f} ({r.t:+.1f})"
                           for r in g.itertuples() if r.medida in
                           ("qw_uniforme", "qw_cola_alta"))
        print(f"  {m:30s} {celdas}")

    print("\n=== comparación principal: Analog-mezcla contra rivales, misma variante ===")
    filas = []
    for v in ("crudo",) + CALIBRADAS:
        for rival in RIVALES:
            for medida in ["qw_uniforme", "qw_cola_alta", "qw_cola_baja", "pin_0.05",
                           "pin_0.50", "pin_0.90", "pin_0.95"]:
                r = compara(x, f"Analog-mezcla | {v}", f"{rival} | {v}", medida)
                r.update({"variante": v, "rival": rival})
                filas.append(r)
    pr = pd.DataFrame(filas)
    pr.round(4).to_csv(pub / "cuadro_calibrados_principal.csv", index=False)
    for v, g in pr.groupby("variante", sort=False):
        print(f"\n--- variante {v} ---")
        tabla = g.pivot_table(index="medida", columns="rival", values="pct").round(1)
        tt = g.pivot_table(index="medida", columns="rival", values="t").round(1)
        for rival in RIVALES:
            tabla[rival] = [f"{a:+.1f}% ({b:+.1f})" for a, b in zip(tabla[rival], tt[rival])]
        print(tabla[RIVALES].to_string())

    print("\n=== el mejor de cada método, por CRPS ponderada hacia la cola alta ===")
    mejor = cal.reset_index().sort_values("qw_cola_alta").groupby("metodo").head(1)
    print(mejor.sort_values("qw_cola_alta")[["metodo", "variante", "cob_cuenta",
                                             "qw_uniforme", "qw_cola_alta"]]
          .round(3).to_string(index=False))

    print("\n=== H3. cobertura de la cuenta por decil del día conocido al decidir ===")
    base = t[t.variante == "crudo"][["metodo", "zona", "origen", "costo_real"]]
    conocido = con_dia_anterior(base)[["zona", "origen", "costo_conocido"]] \
        .drop_duplicates(["zona", "origen"])
    y = t.merge(conocido, on=["zona", "origen"])
    y["dk"] = deciles(y.costo_conocido)
    tabla = y[y.metodo.isin(CLAVE)].pivot_table(index=["metodo", "variante"],
                                                 columns="dk", values="cob_cuenta")
    tabla.round(4).to_csv(pub / "cuadro_calibrados_dia_conocido.csv")
    print(tabla.round(3).to_string())

    print("\n=== H4. nivel_bloque contra nivel ===")
    print("cobertura horaria por bloque de horas (lo prometido es 0.90)")
    hb = pd.read_csv(POR_HORA)
    hb["bloque"] = hb.hora // 6
    hb = hb.groupby(["metodo", "variante", "bloque"])[["cubre", "ancho", "n"]].sum()
    hb["cubre"], hb["ancho"] = hb.cubre / hb.n, hb.ancho / hb.n
    hb.round(3).to_csv(pub / "cuadro_calibrados_por_bloque.csv")
    for m in ["Analog-mezcla", "t0-beta+errores", "PatchTST-FM+errores", "LEAR"]:
        a = hb.loc[m]
        tabla = pd.DataFrame({v: [f"{r.ancho:6.0f} ({r.cubre:.2f})" for r in
                                  a.loc[v].itertuples()]
                              for v in ("proporcional", "nivel", "nivel_bloque")},
                             index=["0-5 h", "6-11 h", "12-17 h", "18-23 h"])
        print(f"\n--- {m}: ancho (cobertura) ---")
        print(tabla.to_string())
    filas = []
    for m in sorted(t.metodo.unique()):
        for medida in ["qw_uniforme", "qw_cola_alta", "qw_cola_baja"]:
            r = compara(x, f"{m} | nivel_bloque", f"{m} | nivel", medida)
            r.update({"metodo_base": m})
            filas.append(r)
    h4 = pd.DataFrame(filas)
    h4.round(4).to_csv(pub / "cuadro_calibrados_h4.csv", index=False)
    print("\nnivel_bloque menos nivel, en % de nivel (negativo = mejor):")
    for m, g in h4.groupby("metodo_base"):
        print(f"  {m:30s} " + "  ".join(f"{r.medida[3:]}: {r.pct:+.1f} ({r.t:+.1f})"
                                       for r in g.itertuples()))


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "calcula"
    if modo == "calcula":
        calcula()
    elif modo == "prueba":
        calcula(solo_una_zona=True)
    elif modo == "analiza":
        analiza()
