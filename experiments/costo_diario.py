"""
¿Qué método sabe decir cuánto puede costar el DÍA, y no sólo cada hora?

La liquidación del mercado es hora por hora: cada hora se paga su propio precio por la
energía de esa hora. De modo que para el costo ESPERADO del día bastan las
distribuciones por hora, y la dependencia entre horas no agrega nada. Eso es cierto y hay
que decirlo.

Lo que la dependencia sí cambia es el RIESGO del total. Con un perfil de carga
comprometido q, el costo del día es la suma de q_h por p_h, y la varianza de esa suma
lleva todas las covarianzas entre horas:

    Var(costo) = suma_h suma_k  q_h q_k Cov(p_h, p_k)

Si las horas fueran independientes, el total se concentraría: promediar veinticuatro
tiros reduce la dispersión. Pero los precios no son independientes — un pico se extiende
por toda una tarde — y entonces el total tiene una cola mucho más gorda de lo que un
modelo de horas desatadas puede imaginar.

La consecuencia es medible y es la que importa para cubrirse: **un método con horas
desatadas subestima sistemáticamente el riesgo de un día caro, aunque acierte cada hora
por separado.** Esto lo mide.

Y hay algo anterior: un método que sólo entrega cuantiles por hora **no puede** formar
una distribución del costo del día. No es que la haga mal: no la puede hacer. Hacen falta
escenarios. Cuatro de los cinco modelos de fundación del banco quedan fuera por eso.

El perfil usado es plano, una unidad de energía cada hora. Es la elección neutral: con un
perfil real la conclusión se mueve en la misma dirección, porque concentra peso en las
horas de la tarde, que son las que más covarían.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (load_forecasts, load_members,  # noqa: E402
                                        member_columns, member_matrix)
from experiments.protocol import RESULTS  # noqa: E402
from experiments.scores import pinball  # noqa: E402

NIVELES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
PERFIL = np.ones(24)


def de_una_corrida(ruta: Path, perfil: np.ndarray = PERFIL) -> pd.DataFrame:
    """
    Cobertura del costo del día bajo los escenarios de cada método.

    Se lee zona por zona y no la corrida entera: los escenarios de las variantes de
    picos pesan 2.4 GB y los de Moirai 3.8 GB, y cargarlos juntos no cabe en memoria.
    """
    carpeta_m = Path(str(ruta) + "_members")
    partes = sorted(carpeta_m.glob("*.parquet")) if carpeta_m.is_dir() else []
    if not partes:
        ## Un método sin escenarios no se omite del cuadro: se declara que NO PUEDE
        ## contestar la pregunta. Omitirlo dejaría una tabla con huecos que se leen como
        ## "no medido", cuando lo que hay que leer es "este método no produce una
        ## distribución del costo del día ni puede producirla".
        propios = sorted(pd.read_parquet(ruta, columns=["method"]).method.unique()) \
            if Path(ruta).is_dir() else []
        return pd.DataFrame([{"metodo": m, "no_puede": True} for m in propios])
    filas, descartados = [], 0
    for parte in partes:
        nuevas, saltados = _de_una_zona(ruta, parte, perfil)
        filas.extend(nuevas)
        descartados += saltados
    if descartados:
        print(f"    ({descartados:,d} días descartados por largo o escenarios)")
    return pd.DataFrame(filas)


def _de_una_zona(ruta: Path, parte: Path, perfil: np.ndarray) -> list:
    """Una zona: los escenarios y los pronósticos de esa misma parte."""
    miembros = pd.read_parquet(parte)
    pronosticos = pd.read_parquet(Path(str(ruta)) / parte.name)
    columnas = member_columns(miembros)

    ## Lo observado es el mismo para todos los métodos, así que se queda una sola copia
    ## por paso. Sin esto, una corrida con cinco métodos devolvía 120 valores donde el
    ## perfil espera 24, y la comprobación de largo descartaba TODOS los días sin decir
    ## nada: el cuadro salía vacío y parecía que no había escenarios.
    unicos = pronosticos.drop_duplicates(subset=["zone", "origin", "step"])
    observados = {(z, o): b.sort_values("step")["observed"].to_numpy()
                  for (z, o), b in unicos.groupby(["zone", "origin"])}

    filas, saltados = [], 0
    for (zona, origen, metodo), bloque in miembros.groupby(
            ["zone", "origin", "method"], sort=False):
        observado = observados.get((zona, origen))
        if observado is None or len(observado) != len(perfil):
            saltados += 1
            continue
        escenarios = member_matrix(bloque.sort_values("step"), columnas)
        if len(escenarios) < 2:
            saltados += 1
            continue
        ## el costo del día de cada escenario, y el que de verdad ocurrió
        costos = escenarios @ perfil
        real = float(observado @ perfil)
        cuantiles = np.quantile(costos, NIVELES)
        fila = {"metodo": metodo, "zona": zona, "origen": origen,
                "costo_real": real, "costo_mediano": float(np.median(costos)),
                "ancho_90": float(cuantiles[6] - cuantiles[0]),
                "dentro_90": bool(cuantiles[0] <= real <= cuantiles[6]),
                "dentro_80": bool(cuantiles[1] <= real <= cuantiles[5]),
                "se_pasa_arriba": bool(real > cuantiles[6]),
                ## Cuánto te quedaste CORTO, en pesos: la cuenta real menos el techo
                ## que el método anunció. Es la cifra que le importa a quien se cubre,
                ## porque es el dinero que no estaba presupuestado. Cero cuando la
                ## cuenta cayó dentro.
                "faltante": float(max(real - cuantiles[6], 0.0)),
                "pinball": float(np.mean([
                    pinball(np.array([real]), np.array([q]), a)[0]
                    for q, a in zip(cuantiles, NIVELES)]))}
        ## La pérdida pinball de la cuenta a CADA nivel, por separado. Es lo que permite
        ## evaluar con un puntaje propio a un comprador de pérdida asimétrica: si quedarse
        ## corto le cuesta c_u y pasarse le cuesta c_o, su reserva óptima es el cuantil
        ## tau = c_u / (c_u + c_o) de la cuenta, y el puntaje propio para esa decisión es
        ## la pinball a ese nivel tau, sobre TODOS los días, sin condicionar en lo que
        ## pasó. Condicionar en los días que resultaron caros premia al que siempre
        ## pronostica alto (el dilema del pronosticador, Lerch et al. 2017).
        for q, a in zip(cuantiles, NIVELES):
            fila[f"pin_{a:g}"] = float(pinball(np.array([real]), np.array([q]), a)[0])
            fila[f"cq_{a:g}"] = float(q)
        filas.append(fila)
    return filas, saltados


def main() -> None:
    corridas = sys.argv[1:] or [str(p).replace("_members", "")
                                for p in sorted(RESULTS.glob("*_members"))
                                if p.is_dir()]
    partes = []
    for corrida in corridas:
        t = de_una_corrida(Path(corrida))
        if not t.empty:
            partes.append(t)
            print(f"  {Path(corrida).name}: {t.metodo.nunique()} métodos, "
                  f"{len(t):,d} días", flush=True)
    if not partes:
        raise SystemExit("ninguna corrida con escenarios")

    t = pd.concat(partes, ignore_index=True)
    if "no_puede" in t.columns:
        sin_escenarios = sorted(t[t.no_puede == True].metodo.unique())  # noqa: E712
        if sin_escenarios:
            print("\n=== NO PUEDEN CONTESTAR: sólo entregan cuantiles por hora ===")
            print("sin escenarios no hay distribución del costo del día, y no es que la")
            print("hagan mal: no la pueden hacer\n")
            for m in sin_escenarios:
                print(f"  {m}")
        t = t[t.no_puede != True]  # noqa: E712
    if t.empty:
        raise SystemExit("ningún método del banco entregó escenarios")
    def tabla(sub: pd.DataFrame) -> pd.DataFrame:
        r = sub.groupby("metodo").agg(
            dias=("costo_real", "size"),
            cubre_90=("dentro_90", "mean"), se_pasa_arriba=("se_pasa_arriba", "mean"),
            falta_pesos=("faltante", "mean"), cuenta=("costo_real", "mean"),
            ancho_90=("ancho_90", "mean"), pinball=("pinball", "mean"))
        r["falta_pct_de_la_cuenta"] = 100 * r.falta_pesos / r.cuenta
        return r

    print("\n=== LA CUENTA DEL DÍA ===")
    print("perfil plano de una unidad por hora, así que la cuenta del día es la SUMA")
    print("de los 24 precios. 'cubre_90' es la fracción de días en que la cuenta real")
    print("cayó dentro del intervalo anunciado: se prometió 0.90. 'falta_pesos' es")
    print("cuánto te quedaste corto respecto del techo anunciado, en pesos, y es cero")
    print("cuando la cuenta cayó dentro.")
    print("\nmás es mejor: cubre_90.   menos es mejor: se_pasa_arriba, falta_pesos,")
    print("ancho_90, pinball.\n")
    print("--- todos los días ---")
    print(tabla(t).sort_values("falta_pesos").round(2).to_string())

    ## Se guarda el detalle por día, no sólo el resumen: mirar otro corte —por decil,
    ## por año, por zona— no debe costar otra pasada sobre los escenarios, que son
    ## gigabytes. Por no tenerlo hubo que recalcular sólo para ver el decil caro.
    detalle = RESULTS / "costo_diario_por_dia.parquet"
    t.to_parquet(detalle, index=False)
    print(f"\n(detalle por día en {detalle.name}: {len(t):,d} filas)")

    ## Los deciles se cortan sobre la cuenta REAL de cada día, con los cortes comunes a
    ## todos los métodos: si cada método cortara por su propia distribución, los deciles
    ## no serían los mismos días y las filas no se podrían comparar.
    cortes = t.costo_real.quantile(np.arange(0, 1.01, 0.1)).to_numpy()
    t["decil"] = np.clip(np.searchsorted(cortes[1:-1], t.costo_real) + 1, 1, 10)

    print("\n--- por decil de la cuenta del día: cuánto te quedas corto, en pesos ---")
    print("el decil 10 son los días más caros\n")
    print(t.pivot_table(index="metodo", columns="decil", values="faltante")
          .round(0).to_string())
    print("\n--- por decil: cobertura de la cuenta, se prometió 0.90 ---\n")
    print(t.pivot_table(index="metodo", columns="decil", values="dentro_90")
          .round(3).to_string())
    print("\n--- los tres deciles más caros, ordenados por el pesos que faltan ---\n")
    cerca = tabla(t[t.decil >= 8]).sort_values("falta_pesos")
    print(cerca.round(2).to_string())
    resumen = tabla(t)
    salida = RESULTS / "costo_diario.csv"
    resumen.round(6).to_csv(salida)
    print(f"\nescrito {salida.name}")


if __name__ == "__main__":
    main()
