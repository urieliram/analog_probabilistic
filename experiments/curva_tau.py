"""
La curva en tau: qué tan bien describe cada método cada parte de la distribución de la
cuenta del día.

Para cada día se calcula la pérdida pinball de la cuenta (perfil plano: la suma de los 24
precios) en 19 niveles, de 0.05 a 0.95. El nivel tau dice de qué parte de la distribución
se habla: tau bajo es la cola de los días baratos, 0.5 el centro, tau alto la cola de los
días caros. La pinball es un puntaje propio: se evalúa sobre TODOS los días, sin escoger
los que salieron caros, de modo que no premia a quien pronostica siempre alto (el dilema
del pronosticador, Lerch et al. 2017).

La curva NO es el costo de una compra. El costo en pesos de equivocarse en el precio
depende de la cantidad comprada y del precio de tiempo real, y eso es otro trabajo. Aquí
se mide la calidad del insumo, parte por parte.

Dos cosas más:

* **Control por número de escenarios.** Analog-mezcla trae 160 escenarios y los métodos
  armados con sus errores traen hasta 60. Con más escenarios una cola se estima mejor, y eso
  sería ventaja de tamaño, no de método. Por eso cada método con más de 60 escenarios
  se califica también con 60 sorteados de los suyos, con semilla fija por día.
* **Un solo número para cada región.** La CRPS ponderada por cuantiles (Gneiting y
  Ranjan, 2011): el promedio de las pinball de los 19 niveles, multiplicado por dos, con
  un peso por nivel. Se reportan los cinco pesos usuales, fijados antes de calcular.

Uso:
    python experiments/curva_tau.py calcula   # una pasada por los escenarios (lento)
    python experiments/curva_tau.py analiza   # cuadros y figura desde el detalle (rápido)
"""

import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import member_columns, member_matrix  # noqa: E402
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import diebold_mariano, pinball  # noqa: E402

NIVELES = np.round(np.arange(0.05, 0.96, 0.05), 2)
PERFIL = np.ones(24)
TAMANO_CONTROL = 60
DETALLE = RESULTS / "curva_tau_por_dia.parquet"

## Los cinco pesos de la CRPS ponderada por cuantiles. Se fijan aquí, antes de ver un
## solo número, para que nadie pueda decir que se eligió el que favorece a alguien.
PESOS = {
    "uniforme": lambda u: np.ones_like(u),
    "centro": lambda u: u * (1 - u),
    "colas": lambda u: (2 * u - 1) ** 2,
    "cola_alta": lambda u: u ** 2,
    "cola_baja": lambda u: (1 - u) ** 2,
}

## Los rivales de la comparación principal y el método que se compara contra ellos
PRINCIPAL = "Analog-mezcla"
RIVALES = ["LEAR", "t0-beta+errores", "PatchTST-FM+errores"]


def _fila(metodo, zona, origen, real, costos):
    cuantiles = np.quantile(costos, NIVELES)
    fila = {"metodo": metodo, "zona": zona, "origen": origen,
            "costo_real": real, "n_escenarios": len(costos)}
    for q, a in zip(cuantiles, NIVELES):
        fila[f"pin_{a:.2f}"] = float(pinball(np.array([real]), np.array([q]), a)[0])
    return fila


def _de_una_zona(args):
    ruta, parte = args
    miembros = pd.read_parquet(parte)
    pronosticos = pd.read_parquet(Path(ruta) / parte.name,
                                  columns=["zone", "origin", "step", "observed"])
    columnas = member_columns(miembros)
    unicos = pronosticos.drop_duplicates(subset=["zone", "origin", "step"])
    observados = {(z, o): b.sort_values("step")["observed"].to_numpy()
                  for (z, o), b in unicos.groupby(["zone", "origin"])}
    filas, saltados = [], 0
    for (zona, origen, metodo), bloque in miembros.groupby(
            ["zone", "origin", "method"], sort=False):
        observado = observados.get((zona, origen))
        if observado is None or len(observado) != len(PERFIL):
            saltados += 1
            continue
        escenarios = member_matrix(bloque.sort_values("step"), columnas)
        if len(escenarios) < 2:
            saltados += 1
            continue
        costos = escenarios @ PERFIL
        real = float(observado @ PERFIL)
        filas.append(_fila(metodo, zona, origen, real, costos))
        if len(costos) > TAMANO_CONTROL:
            ## semilla fija por día y método: el sorteo se puede repetir exacto
            semilla = zlib.crc32(f"{zona}|{origen}|{metodo}".encode())
            elegidos = np.random.default_rng(semilla).choice(
                len(costos), TAMANO_CONTROL, replace=False)
            filas.append(_fila(f"{metodo} ({TAMANO_CONTROL} esc.)", zona, origen,
                               real, costos[elegidos]))
    return filas, saltados, f"{Path(ruta).name}/{parte.name}"


def calcula() -> None:
    corridas = [Path(str(p).replace("_members", ""))
                for p in sorted(RESULTS.glob("*_members")) if p.is_dir()]
    tareas = [(str(c), parte) for c in corridas
              for parte in sorted(Path(str(c) + "_members").glob("*.parquet"))]
    print(f"{len(corridas)} corridas, {len(tareas)} archivos por zona", flush=True)
    filas, saltados = [], 0
    ## cuatro procesos: la máquina también corre producción
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (nuevas, s, nombre) in enumerate(pool.map(_de_una_zona, tareas), 1):
            filas.extend(nuevas)
            saltados += s
            if i % 25 == 0 or i == len(tareas):
                print(f"  {i}/{len(tareas)} {nombre}", flush=True)
    t = pd.DataFrame(filas)
    t.to_parquet(DETALLE, index=False)
    print(f"{len(t):,d} filas en {DETALLE.name}; {saltados:,d} días descartados")


def qwcrps(t: pd.DataFrame) -> pd.DataFrame:
    """CRPS ponderada por cuantiles, discretizada en los 19 niveles, por día."""
    pines = t[[f"pin_{a:.2f}" for a in NIVELES]].to_numpy()
    salida = t[["metodo", "zona", "origen"]].copy()
    for nombre, w in PESOS.items():
        salida[f"qw_{nombre}"] = 2 * (pines * w(NIVELES)).mean(axis=1)
    return salida


def compara(t: pd.DataFrame, a: str, b: str, columna: str) -> dict:
    """
    A contra B en una columna de pérdida, pareado por zona y día de decisión.

    La diferencia se suma sobre las zonas de cada día y la prueba de Diebold-Mariano
    corre sobre esa serie de días, con varianza de Newey-West. Negativo favorece a A.
    """
    pa = t[t.metodo == a].set_index(["zona", "origen"])[columna]
    pb = t[t.metodo == b].set_index(["zona", "origen"])[columna]
    comun = pa.index.intersection(pb.index)
    ## Se SUMA por día de decisión, no se promedia: en un subconjunto de días (un decil)
    ## cada día trae un número distinto de zonas, y promediar le daría a un día con una
    ## zona el mismo peso que a uno con veinte. Con sumas, el porcentaje coincide con el
    ## de los promedios simples sobre las mismas zonas y días.
    dif = (pa.loc[comun] - pb.loc[comun]).groupby(level="origen").sum().sort_index()
    base = pb.loc[comun].groupby(level="origen").sum().reindex(dif.index)
    estadistico, _ = diebold_mariano(dif.to_numpy())
    media = float(dif.mean())
    error = abs(media / estadistico) if estadistico else np.nan
    nivel = float(base.mean())
    return {"a": a, "b": b, "medida": columna, "dias": len(dif),
            "zona_dias": len(comun), "nivel_a": float(pa.loc[comun].mean()),
            "nivel_b": float(pb.loc[comun].mean()), "diferencia_por_dia": media,
            "t": estadistico,
            "pct": 100 * media / nivel,
            "pct_bajo": 100 * (media - 1.96 * error) / nivel,
            "pct_alto": 100 * (media + 1.96 * error) / nivel}


def analiza() -> None:
    t = pd.read_parquet(DETALLE)
    pub = RESULTS / "publicacion"
    pub.mkdir(exist_ok=True)

    print("=== escenarios por método ===")
    print(t.groupby("metodo").n_escenarios.agg(["min", "median", "max", "size"])
          .to_string())

    ## 1. el nivel de cada método en cada tau (menos es mejor)
    niveles = t.groupby("metodo")[[f"pin_{a:.2f}" for a in NIVELES]].mean()
    niveles.columns = [f"{a:.2f}" for a in NIVELES]
    niveles.round(2).to_csv(pub / "cuadro_curva_tau_niveles.csv")
    print("\n=== pinball media de la cuenta del día por tau (menos es mejor) ===")
    print(niveles.round(0).to_string())

    ## 2. la curva: el principal contra cada rival, en cada tau, con su banda
    filas = []
    for principal in [PRINCIPAL, f"{PRINCIPAL} ({TAMANO_CONTROL} esc.)"]:
        for rival in RIVALES:
            for a in NIVELES:
                r = compara(t, principal, rival, f"pin_{a:.2f}")
                r["tau"] = a
                filas.append(r)
    curva = pd.DataFrame(filas)
    curva.round(4).to_csv(pub / "cuadro_curva_tau.csv", index=False)
    print("\n=== Analog-mezcla menos el rival, en % del rival, con IC 95% ===")
    print("(negativo: gana el análogo)")
    for principal, g in curva.groupby("a", sort=False):
        print(f"\n--- {principal} ---")
        tabla = g.pivot_table(index="tau", columns="b", values="pct").round(1)
        tt = g.pivot_table(index="tau", columns="b", values="t").round(1)
        for rival in RIVALES:
            tabla[rival] = [f"{p:+.1f}% (t={x:+.1f})"
                            for p, x in zip(tabla[rival], tt[rival])]
        print(tabla[RIVALES].to_string())

    ## 3. un número por región: la CRPS ponderada por cuantiles
    qw = qwcrps(t)
    resumen = qw.groupby("metodo")[[f"qw_{n}" for n in PESOS]].mean()
    resumen.round(2).to_csv(pub / "cuadro_crps_ponderada.csv")
    print("\n=== CRPS ponderada por cuantiles de la cuenta (menos es mejor) ===")
    print(resumen.sort_values("qw_cola_alta").round(1).to_string())
    pruebas = [compara(qw, p, r, f"qw_{n}") for p in
               [PRINCIPAL, f"{PRINCIPAL} ({TAMANO_CONTROL} esc.)"]
               for r in RIVALES for n in PESOS]
    pruebas = pd.DataFrame(pruebas)
    pruebas.round(4).to_csv(pub / "cuadro_crps_ponderada_pruebas.csv", index=False)
    print("\n=== Analog-mezcla menos rival en CRPS ponderada (% del rival, t) ===")
    for principal, g in pruebas.groupby("a", sort=False):
        print(f"\n--- {principal} ---")
        for _, r in g.iterrows():
            print(f"  {r.b:22s} {r.medida:14s} {r.pct:+6.1f}%  "
                  f"[{r.pct_bajo:+.1f}, {r.pct_alto:+.1f}]  t={r.t:+.2f}")

    figura(curva)


def figura(curva: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(1, len(RIVALES), figsize=(13, 4.2), sharey=True)
    for eje, rival in zip(ejes, RIVALES):
        for principal, color in [(PRINCIPAL, "C0"),
                                 (f"{PRINCIPAL} ({TAMANO_CONTROL} esc.)", "C1")]:
            g = curva[(curva.a == principal) & (curva.b == rival)].sort_values("tau")
            eje.plot(g.tau, g.pct, color=color, marker="o", ms=3, label=principal)
            eje.fill_between(g.tau, g.pct_bajo, g.pct_alto, color=color, alpha=0.15)
        eje.axhline(0, color="black", lw=0.8)
        eje.set_title(f"contra {rival}")
        eje.set_xlabel("τ (parte de la distribución de la cuenta)")
        eje.grid(alpha=0.3)
    ejes[0].set_ylabel("diferencia de pinball, % del rival\n(abajo de cero gana el análogo)")
    ejes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    salida = Path(__file__).resolve().parents[1] / "figuras" / "figura_curva_tau.png"
    fig.savefig(salida, dpi=160)
    print(f"\nfigura en {salida}")


if __name__ == "__main__":
    {"calcula": calcula, "analiza": analiza}[sys.argv[1] if len(sys.argv) > 1
                                            else "analiza"]()
