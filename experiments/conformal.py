"""
Calibración conformal del ensamble: ¿se arregla barato el abanico estrecho?

El intervalo que promete cubrir 90 veces de cada 100 cubre 82. Esta es la prueba de
si eso se arregla con una corrección sencilla o no, y es una prueba, no un parche: si
el método más simple lo arregla, el hallazgo del artículo es menor y hay que decirlo.

La idea del conformal, en llano: en vez de creerle al abanico que el ensamble opina,
se mide qué tan equivocado ha estado en los días ya observados y se ensancha hasta que
la promesa se cumpla sobre esos días. No supone nada sobre la forma de la distribución
de precios —ni normal, ni simétrica—; sólo pide que el pasado reciente se parezca al
futuro inmediato.

Se escalan los MIEMBROS alrededor de la mediana y no se recalibran los cuantiles uno
por uno:

    Y~(i, tau) = m(tau) + c(tau) * (Y(i, tau) - m(tau))

así el abanico se ensancha y el orden de los escenarios no cambia —el miembro que iba
más alto sigue yendo más alto—, de modo que las trayectorias sobreviven y los puntajes
que las usan siguen significando algo.

Cuatro variantes, y están elegidas por lo que ya se midió:

``split``        una corrección constante, calculada una vez sobre un bloque inicial.
                 Es el conformal por división, el más simple. Se incluye sabiendo que
                 probablemente falle: la falla de cobertura cambia de signo con el
                 régimen —en los días tranquilos se escapa por abajo, en los movidos
                 por arriba— y una constante no puede seguir eso.
``enbpi``        ventana rodante de los últimos orígenes observados, corrección nueva
                 cada día (Xu y Xie, ICML 2021). Sigue al régimen con retraso.
``normalizado``  como enbpi, pero cada error se divide por la dispersión que el
                 ensamble traía ese día antes de tomar el cuantil. El ancho entonces
                 escala con la agitación del momento, que es lo que el diagnóstico
                 pide, y es lo que Tweedie habría intentado dar por el lado equivocado
                 —haciendo crecer la varianza con el nivel del precio en vez de con el
                 movimiento—.
``asimetrico``   normalizado, con un factor para arriba de la mediana y otro para
                 abajo, porque los dos lados no fallan igual.

Ninguna variante mira un error posterior al origen que está pronosticando.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.forecast_store import (member_columns,  # noqa: E402
                                        member_matrix, quantile_columns)
from experiments.protocol import COMMON_LEVELS, RESULTS  # noqa: E402
from experiments.scores import (LEVELS, evaluate,  # noqa: E402
                                mean_pinball)

VENTANA_RODANTE = 100
BLOQUE_SPLIT = 100
ALFA = 0.10
MINIMO = 1e-6
TOPE_FACTOR = 10.0
VARIANTES = ("split", "enbpi", "normalizado", "asimetrico")


def pinball_comun(observado: np.ndarray, cuantiles: np.ndarray,
                  niveles: list) -> float:
    """
    La pérdida pinball sobre la rejilla que TODOS los métodos emiten, de 0.1 a 0.9.

    Hace falta porque promediar sobre rejillas distintas no compara nada: TiRex y
    TimesFM emiten nueve niveles y el análogo diecinueve, y el promedio de nueve
    pérdidas no es del mismo tamaño que el de diecinueve. Esta columna es la que lleva
    la comparación entre métodos; la que se calcula sobre la rejilla propia de cada uno
    sirve para su propio registro, no para el cuadro.
    """
    posicion = {round(float(a), 4): i for i, a in enumerate(niveles)}
    indices, usados = [], []
    for a in COMMON_LEVELS:
        clave = round(float(a), 4)
        if clave in posicion and np.all(np.isfinite(cuantiles[posicion[clave]])):
            indices.append(posicion[clave])
            usados.append(a)
    if len(usados) != len(COMMON_LEVELS):
        return float("nan")
    return mean_pinball(observado, cuantiles[indices], usados)


def cuantil_conformal(puntajes: np.ndarray, alfa: float) -> np.ndarray:
    """
    El cuantil conformal de los puntajes, con la corrección de muestra finita.

    Se toma el ``ceil((n+1)(1-alfa))``-ésimo valor ordenado y no el percentil
    empírico: es esa corrección la que da la garantía de cobertura con n finito, y
    sin ella esto no es conformal sino un promedio de errores con buena intención.
    Cuando los puntajes no alcanzan para el nivel pedido, el cuantil es infinito,
    que es la respuesta honesta y no un número inventado.
    """
    n = puntajes.shape[0]
    if n == 0:
        return np.full(puntajes.shape[1:], np.inf)
    posicion = int(np.ceil((n + 1) * (1 - alfa)))
    if posicion > n:
        return np.full(puntajes.shape[1:], np.inf)
    return np.sort(puntajes, axis=0)[posicion - 1]


def factor_simetrico(observado: np.ndarray, mediana: np.ndarray,
                     dispersion: np.ndarray, alfa: float,
                     normalizar: bool) -> np.ndarray:
    """Cuánto ensanchar cada paso del horizonte, un solo factor por paso."""
    desvio = np.abs(observado - mediana)
    puntajes = desvio / np.maximum(dispersion, MINIMO) if normalizar else desvio
    cuantil = cuantil_conformal(puntajes, alfa)
    if normalizar:
        return cuantil
    ## sin normalizar el cuantil está en pesos: se vuelve factor con la dispersión
    ## media que el ensamble trajo en esos mismos días
    return cuantil / np.maximum(dispersion.mean(axis=0), MINIMO)


def factores_asimetricos(observado: np.ndarray, mediana: np.ndarray,
                         arriba: np.ndarray, abajo: np.ndarray,
                         alfa: float) -> tuple:
    """
    Un factor para cada lado, cada uno al nivel 1 - alfa/2.

    Dos puntajes de una cola al nivel 1 - alfa/2 dan cobertura de al menos 1 - alfa,
    y hace falta separarlos porque los dos lados no fallan igual: en los días
    tranquilos el precio se escapa por abajo y en los movidos por arriba.
    """
    exceso = observado - mediana
    s_arriba = np.where(exceso > 0, exceso / np.maximum(arriba, MINIMO), 0.0)
    s_abajo = np.where(exceso < 0, -exceso / np.maximum(abajo, MINIMO), 0.0)
    return (cuantil_conformal(s_arriba, ALFA / 2),
            cuantil_conformal(s_abajo, ALFA / 2))


def escala(miembros: np.ndarray, mediana: np.ndarray,
           c_arriba: np.ndarray, c_abajo: np.ndarray = None) -> np.ndarray:
    """Estira el ensamble alrededor de su mediana, por lado si se dan dos factores."""
    desvio = miembros - mediana[None, :]
    if c_abajo is None:
        return mediana[None, :] + np.clip(c_arriba, 1.0, None)[None, :] * desvio
    arriba = np.clip(c_arriba, 1.0, None)[None, :]
    abajo = np.clip(c_abajo, 1.0, None)[None, :]
    return mediana[None, :] + np.where(desvio > 0, arriba * desvio, abajo * desvio)


def una_zona_cuantiles(pronosticos: pd.DataFrame, metodo: str, niveles: list,
                       pasos: int) -> list:
    """
    Calibra un método que sólo entrega bordes, con regresión cuantílica conformalizada.

    Es la ruta de los cuatro modelos de fundación que no dan escenarios. La idea es la
    misma que para los que sí los dan —medir lo que falló en los días ya observados y
    corregir por ese tanto— y las cuatro variantes son las mismas, para que las dos
    familias se comparen con idénticas reglas.
    """
    columnas = {round(float(a), 4): f"q{a:g}" for a in niveles}
    historial = {"obs": [], "q": {round(float(a), 4): [] for a in niveles}}
    salida = []

    for numero, (origen, p) in enumerate(
            sorted(pronosticos.groupby("origin"), key=lambda kv: kv[0])):
        p = p.sort_values("step")
        if len(p) != pasos:
            continue
        observado = p["observed"].to_numpy()
        hoy = np.array([p[columnas[round(float(a), 4)]].to_numpy() for a in niveles])

        ## se califica sobre los niveles que este método emite de verdad: pasarle los
        ## ausentes a la evaluación devuelve NaN en el puntaje principal y en la
        ## confiabilidad, y entonces el método desaparece del cuadro sin decir por qué
        vivos = [a for i, a in enumerate(niveles) if np.all(np.isfinite(hoy[i]))]
        indices = [niveles.index(a) for a in vivos]
        if len(vivos) < 3:
            continue

        if historial["obs"]:
            for variante in VARIANTES:
                ajustado = calibra_cuantiles(historial, hoy, niveles, variante,
                                            VENTANA_RODANTE, BLOQUE_SPLIT)
                if not np.all(np.isfinite(ajustado[indices])):
                    continue
                puntajes = evaluate(observado, ajustado[indices], vivos)
                puntajes["pinball_comun"] = pinball_comun(observado, ajustado,
                                                          niveles)
                puntajes.update({"zone": p.zone.iloc[0], "origin": origen,
                                 "method": f"{metodo}+{variante}",
                                 "factor_medio": float(np.nanmean(
                                     np.abs(ajustado - hoy))),
                                 "pasos_acotados": 0,
                                 "niveles_propios": len(vivos)})
                salida.append(puntajes)

        if numero > 0:
            puntajes = evaluate(observado, hoy[indices], vivos)
            puntajes["pinball_comun"] = pinball_comun(observado, hoy, niveles)
            puntajes.update({"zone": p.zone.iloc[0], "origin": origen,
                             "method": metodo, "factor_medio": 0.0,
                             "pasos_acotados": 0, "niveles_propios": len(vivos)})
            salida.append(puntajes)

        historial["obs"].append(observado)
        for a in niveles:
            historial["q"][round(float(a), 4)].append(hoy[niveles.index(a)])
        ## sólo se conserva lo que las variantes pueden mirar
        tope = max(VENTANA_RODANTE, BLOQUE_SPLIT)
        if len(historial["obs"]) > tope:
            historial["obs"] = historial["obs"][-tope:]
            for a in niveles:
                clave = round(float(a), 4)
                historial["q"][clave] = historial["q"][clave][-tope:]
    return salida


def una_zona(pronosticos: pd.DataFrame, miembros: pd.DataFrame, metodo: str,
             columnas_m: list, niveles: list, pasos: int) -> list:
    """Recorre los orígenes de una zona en orden y calibra con lo ya observado."""
    cuantiles_por_nivel = {round(float(a), 4): f"q{a:g}" for a in niveles}
    fila_mediana = cuantiles_por_nivel[0.5]
    q_alta, q_baja = cuantiles_por_nivel[0.95], cuantiles_por_nivel[0.05]

    bloques_p = {o: b.sort_values("step") for o, b in pronosticos.groupby("origin")}
    bloques_m = {o: b.sort_values("step") for o, b in miembros.groupby("origin")}
    origenes = sorted(set(bloques_p) & set(bloques_m))

    historial = {"obs": [], "med": [], "disp": [], "arr": [], "aba": []}
    salida = []
    for numero, origen in enumerate(origenes):
        p, m = bloques_p[origen], bloques_m[origen]
        if len(p) != pasos or len(m) != pasos:
            continue
        ## sin los miembros ausentes: un archivo con métodos de distinto número de
        ## miembros deja columnas en NaN, y leerlas envenena todos los cuantiles
        ensamble = member_matrix(m, columnas_m)
        if len(ensamble) < 2:
            continue
        observado = p["observed"].to_numpy()
        mediana = p[fila_mediana].to_numpy()
        dispersion = ensamble.std(axis=0)
        arriba = p[q_alta].to_numpy() - mediana
        abajo = mediana - p[q_baja].to_numpy()

        if historial["obs"]:
            H = {k: np.array(v) for k, v in historial.items()}
            ## split: la corrección se congela con el bloque inicial y no se mueve
            tope = min(len(H["obs"]), BLOQUE_SPLIT)
            factores = {
                "split": (factor_simetrico(H["obs"][:tope], H["med"][:tope],
                                           H["disp"][:tope], ALFA, False), None),
                "enbpi": (factor_simetrico(H["obs"][-VENTANA_RODANTE:],
                                           H["med"][-VENTANA_RODANTE:],
                                           H["disp"][-VENTANA_RODANTE:],
                                           ALFA, False), None),
                "normalizado": (factor_simetrico(H["obs"][-VENTANA_RODANTE:],
                                                 H["med"][-VENTANA_RODANTE:],
                                                 H["disp"][-VENTANA_RODANTE:],
                                                 ALFA, True), None),
                "asimetrico": factores_asimetricos(
                    H["obs"][-VENTANA_RODANTE:], H["med"][-VENTANA_RODANTE:],
                    H["arr"][-VENTANA_RODANTE:], H["aba"][-VENTANA_RODANTE:], ALFA),
            }
            for nombre, (c_arr, c_aba) in factores.items():
                acotados = 0
                if nombre == "normalizado":
                    ## Un ensamble degenerado —cuando el recorte pega casi todos los
                    ## miembros al máximo histórico, o cuando la ventana presente es
                    ## casi constante— deja q95 - mediana cerca de cero, y dividir por
                    ## eso infló el ancho a 1.28e7 sobre las 25 zonas. No pasó con dos.
                    ## El factor se acota y se cuenta cuántas veces se acota, porque un
                    ## tope silencioso se lee como si nada hubiera pasado.
                    ## El cuantil normalizado viene en unidades de desviación del
                    ## ensamble: el intervalo que define es mediana +- Q * desviación
                    ## de hoy. Para lograrlo escalando miembros hace falta el factor
                    ## que lleva la media-anchura de hoy a ese número, y la
                    ## media-anchura es q95 - mediana, no la desviación media de los
                    ## miembros respecto de la mediana. Confundirlas infló el ancho a
                    ## 1957 y la cobertura a 0.976: el intervalo quedaba cumpliendo
                    ## de más, que también es estar mal calibrado.
                    c_arr = c_arr * dispersion / np.maximum(arriba, MINIMO)
                    acotados = int(np.sum(c_arr > TOPE_FACTOR))
                    c_arr = np.minimum(c_arr, TOPE_FACTOR)
                if not np.all(np.isfinite(c_arr)):
                    continue
                if nombre == "asimetrico":
                    nuevo = escala(ensamble, mediana, c_arr, c_aba)
                else:
                    nuevo = escala(ensamble, mediana, c_arr)
                nuevo = np.sort(nuevo, axis=0)
                cuantiles_nuevos = np.quantile(nuevo, niveles, axis=0)
                puntajes = evaluate(observado, cuantiles_nuevos, niveles, nuevo)
                puntajes["pinball_comun"] = pinball_comun(observado,
                                                          cuantiles_nuevos, niveles)
                puntajes.update({"zone": p.zone.iloc[0], "origin": origen,
                                 "method": f"{metodo}+{nombre}",
                                 "factor_medio": float(np.mean(c_arr)),
                                 "pasos_acotados": acotados})
                salida.append(puntajes)

        ## el crudo, para comparar sobre exactamente los mismos orígenes
        if numero > 0:
            cuantiles_crudos = np.quantile(ensamble, niveles, axis=0)
            puntajes = evaluate(observado, cuantiles_crudos, niveles, ensamble)
            puntajes["pinball_comun"] = pinball_comun(observado, cuantiles_crudos,
                                                      niveles)
            puntajes.update({"zone": p.zone.iloc[0], "origin": origen,
                             "method": metodo, "factor_medio": 1.0})
            salida.append(puntajes)

        for clave, valor in [("obs", observado), ("med", mediana),
                             ("disp", dispersion), ("arr", arriba), ("aba", abajo)]:
            historial[clave].append(valor)
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrida")
    parser.add_argument("--metodo", default=None,
                        help="por omisión, todos los que traiga la corrida")
    parser.add_argument("--zonas", type=int, default=None)
    args = parser.parse_args()

    ruta = Path(args.corrida)
    partes = sorted((ruta if ruta.is_dir() else ruta.with_suffix("")).glob("*.parquet"))
    if args.zonas:
        partes = partes[: args.zonas]
    print(f"{len(partes)} zonas, método {args.metodo}", flush=True)

    ## por omisión se calibran TODOS los métodos de la corrida: la capa tiene que
    ## aplicarse a todo el banco con las mismas reglas, o la comparación queda amañada
    ## a favor del método propio
    metodos = ([args.metodo] if args.metodo
               else sorted(pd.read_parquet(partes[0], columns=["method"])
                           .method.unique()))
    print(f"métodos: {', '.join(metodos)}", flush=True)

    todo = []
    for numero, parte in enumerate(partes, 1):
        completo = pd.read_parquet(parte)
        ## los miembros de la corrida viven en un directorio hermano, una parte por
        ## zona con el mismo nombre
        ## los miembros viven en un directorio hermano; un método que sólo entrega
        ## bordes no lo tiene, y entonces va por la ruta de cuantiles
        carpeta_m = Path(str(ruta).rstrip("/") + "_members")
        archivo_m = carpeta_m / parte.name
        completo_m = pd.read_parquet(archivo_m) if archivo_m.exists() else None
        columnas_q, niveles = quantile_columns(completo)

        filas_zona = 0
        for metodo in metodos:
            p = completo[completo.method == metodo]
            if p.empty:
                continue
            m = (completo_m[completo_m.method == metodo]
                 if completo_m is not None else None)
            if m is None or m.empty:
                filas = una_zona_cuantiles(p, metodo, niveles, int(p.step.max()))
            else:
                filas = una_zona(p, m, metodo, member_columns(m), niveles,
                                 int(p.step.max()))
            todo.extend(filas)
            filas_zona += len(filas)
        print(f"  {numero}/{len(partes)} {parte.stem:20s} {filas_zona:,d} filas",
              flush=True)

    tabla = pd.DataFrame(todo)
    destino = RESULTS / f"{ruta.name.replace('.parquet','')}_conformal.csv"
    tabla.to_csv(destino, index=False)
    print(f"\nescrito {destino.name}: {len(tabla):,d} filas")



## ---------------------------------------------------------------------------
## Calibración de los métodos que sólo entregan cuantiles
## ---------------------------------------------------------------------------
##
## Cuatro de los cinco modelos de fundación —y cualquier modelo de regresión
## cuantílica— no entregan escenarios: entregan bordes. No hay miembros que escalar, así
## que la capa de arriba no se les puede aplicar, y aplicarles otra distinta sería
## comparar métodos con tratamientos distintos, que es justo lo que se quiere evitar.
##
## Lo que sí se les aplica es la MISMA idea con la herramienta que corresponde a su forma
## de salida: regresión cuantílica conformalizada (Romano, Patterson y Candès, 2019). Para
## cada par simétrico de bordes se mide, en los días ya observados, cuánto se salió el
## precio del intervalo que esos dos bordes definían; se toma el cuantil conformal de esas
## violaciones; y se ensancha el par por ese tanto. Un valor negativo encoge el intervalo,
## que es lo correcto cuando el método venía siendo demasiado ancho.
##
## Las cuatro variantes son las mismas que arriba, para que las dos familias se comparen
## con las mismas reglas: constante, rodante, normalizada por el ancho del día, y
## asimétrica con un ajuste por lado.

def _pares_simetricos(niveles) -> list:
    """Los pares (bajo, alto) que definen un intervalo central, de más ancho a más angosto."""
    bajos = sorted(a for a in niveles if a < 0.5 - 1e-9)
    pares = []
    for bajo in bajos:
        alto = round(1.0 - bajo, 4)
        if any(abs(alto - a) < 1e-9 for a in niveles):
            pares.append((bajo, alto))
    return pares


def violaciones(observado: np.ndarray, bajo: np.ndarray,
                alto: np.ndarray) -> tuple:
    """
    Cuánto se salió el precio del intervalo, por lado y en conjunto.

    El valor conjunto es negativo cuando el precio quedó cómodamente dentro, y entonces
    el cuantil conformal encoge el intervalo en vez de ensancharlo. Eso es deseable: un
    intervalo demasiado ancho también está mal calibrado.
    """
    por_abajo = bajo - observado
    por_arriba = observado - alto
    return np.maximum(por_abajo, por_arriba), por_abajo, por_arriba


def calibra_cuantiles(historial: dict, cuantiles_hoy: np.ndarray, niveles: list,
                      variante: str, ventana: int, bloque: int) -> np.ndarray:
    """
    Ajusta cada par de bordes con las violaciones de los días ya observados.

    ``historial`` trae, por nivel, las listas de bordes y observaciones pasadas. Nada
    posterior al origen entra aquí.
    """
    posicion = {round(float(a), 4): i for i, a in enumerate(niveles)}
    salida = np.array(cuantiles_hoy, dtype=float, copy=True)
    obs = np.array(historial["obs"])
    if len(obs) == 0:
        return salida

    for bajo, alto in _pares_simetricos(niveles):
        i, j = posicion[bajo], posicion[alto]
        if np.isnan(cuantiles_hoy[i]).any() or np.isnan(cuantiles_hoy[j]).any():
            continue
        q_bajo = np.array(historial["q"][bajo])
        q_alto = np.array(historial["q"][alto])
        if np.isnan(q_bajo).any() or np.isnan(q_alto).any():
            continue

        alfa = round(2 * bajo, 4)            ## el par define una cobertura de 1 - alfa
        recorte = slice(None, bloque) if variante == "split" else slice(-ventana, None)
        o, b, a = obs[recorte], q_bajo[recorte], q_alto[recorte]
        conjunta, por_abajo, por_arriba = violaciones(o, b, a)

        if variante == "asimetrico":
            q_lo = cuantil_conformal(por_abajo, alfa / 2)
            q_hi = cuantil_conformal(por_arriba, alfa / 2)
        elif variante == "normalizado":
            ancho = np.maximum(a - b, MINIMO)
            escala_hoy = max(float(np.mean(cuantiles_hoy[j] - cuantiles_hoy[i])), MINIMO)
            q_lo = q_hi = cuantil_conformal(conjunta / ancho, alfa) * escala_hoy
        else:                                 ## constante y rodante
            q_lo = q_hi = cuantil_conformal(conjunta, alfa)

        if not (np.all(np.isfinite(q_lo)) and np.all(np.isfinite(q_hi))):
            continue
        salida[i] = cuantiles_hoy[i] - q_lo
        salida[j] = cuantiles_hoy[j] + q_hi

    ## Ensanchar cada par por su cuenta puede cruzar los bordes: un par interior podría
    ## recibir más ajuste que el que lo contiene. Se fuerza el orden, que es una
    ## propiedad de cualquier función de cuantiles y no una corrección cosmética.
    ##
    ## El orden se fuerza SÓLO entre los niveles que el método emite. Acumular sobre
    ## todos borraba las cuatro variantes en silencio: el nivel más bajo de la rejilla
    ## es ausente en los modelos que se detienen en 0.9, y maximum.accumulate propaga
    ## el ausente a toda la columna, de modo que el resultado salía entero en NaN y se
    ## descartaba sin aviso.
    presentes = [i for i, a in enumerate(niveles)
                 if np.all(np.isfinite(salida[i]))]
    if len(presentes) > 1:
        orden = sorted(presentes, key=lambda i: float(niveles[i]))
        salida[orden] = np.maximum.accumulate(salida[orden], axis=0)
    return salida


if __name__ == "__main__":
    main()
