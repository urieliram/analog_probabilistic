"""
LEAR: el comparativo que un revisor de pronóstico de precios va a pedir por nombre.

Es el modelo de referencia de la literatura de precios eléctricos —Lago, Marcjasz,
De Schutter y Weron, *Forecasting day-ahead electricity prices: A review of state-of-
the-art algorithms, best practices and an open-access benchmark*, Applied Energy 293
(2021) 116983—. En palabras llanas: para cada hora del día siguiente ajusta una
regresión lineal sobre los precios de los días -1, -2, -3 y -7 a todas sus horas, más
el día de la semana, y deja que un **lazo** —una penalización que empuja a cero los
coeficientes que no ganan su lugar— decida cuáles de esas 103 variables se quedan. Se
reajusta cada día, igual que en operación.

Tres decisiones que hay que declarar, porque cambian lo que significa el resultado:

1. **Sin exógenas.** El LEAR del artículo original usa el pronóstico de demanda y de
   generación que publica el operador. Aquí no entran: el análogo tampoco los ve, y la
   demanda de este mercado es regional mientras la zona es una fracción. Lo que se
   compara son dos métodos con el mismo conjunto de información, precios y calendario.
2. **La misma historia que el análogo**, dos años por omisión, aunque el LEAR original
   promedia cuatro ventanas de calibración de hasta cuatro años. Darle más historia que
   al análogo lo favorecería, y aquí el interés es que la comparación no esté inclinada
   en ninguna de las dos direcciones.
3. **Transformación estabilizadora de varianza.** Los precios eléctricos tienen picos
   que arrastran una regresión por mínimos cuadrados. Se ajusta sobre
   `asinh((p - mediana) / MAD)` y se deshace al predecir, que es lo recomendado por
   Uniejewski, Weron y Ziel (IEEE Trans. Power Systems 33, 2018) y lo que usa el LEAR
   de referencia.

LEAR es un pronóstico puntual: entrega un número por hora, no una distribución. Los
cuantiles salen de sus propios errores recientes, que `LEARRodante` acumula origen a
origen, y por eso nunca mira un error que no estuviera ya observado.
"""

from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LassoLarsIC

from experiments.benchmarks.naive import _cuantiles_de_errores

HORAS = 24
REZAGOS_DIA = (1, 2, 3, 7)
DIAS_CALIBRACION = 728
VENTANA_ERRORES = 60


def _estabiliza(precios: np.ndarray) -> tuple:
    """
    Lleva los precios a una escala donde los picos no arrastran la regresión.

    Devuelve la serie transformada y los dos números que hacen falta para deshacerlo.
    La desviación absoluta mediana se normaliza por 0.6745 para que, en una normal,
    valga lo mismo que la desviación estándar.
    """
    centro = float(np.median(precios))
    escala = float(np.median(np.abs(precios - centro))) / 0.6745
    if escala <= 0:
        escala = float(np.std(precios)) or 1.0
    return np.arcsinh((precios - centro) / escala), centro, escala


def _deshace(valores: np.ndarray, centro: float, escala: float) -> np.ndarray:
    return np.sinh(valores) * escala + centro


def _matriz_de_dias(dias: np.ndarray, indices: np.ndarray,
                    dia_semana: np.ndarray) -> np.ndarray:
    """
    Una fila por día a predecir: los rezagos diarios a todas sus horas y el calendario.

    ``dias`` es la serie en forma de días por horas. ``indices`` son los días cuya
    fila se construye; cada uno usa los días que tiene antes, nunca él mismo.
    """
    bloques = [dias[indices - rezago] for rezago in REZAGOS_DIA]
    semana = np.zeros((len(indices), 7))
    semana[np.arange(len(indices)), dia_semana[indices]] = 1.0
    return np.hstack(bloques + [semana])


def lear_punto(historia: np.ndarray, horizonte: int = HORAS,
               dias_calibracion: int = DIAS_CALIBRACION,
               dia_semana_origen: int = 0) -> np.ndarray:
    """
    Pronóstico puntual de las 24 horas del día siguiente al origen.

    ``historia`` termina en la hora 23 del último día observado, de modo que el día a
    predecir es el siguiente y todos sus rezagos ya están dentro.

    ``dia_semana_origen`` es el día de la semana de ese último día observado, en 0 a 6.
    Se cuenta desde el origen hacia atrás y no desde el principio de la historia a
    propósito: la historia se recorta a un número entero de días por el final, así que
    su primera hora cae en un día distinto según el recorte, y amarrar el calendario a
    ella es cómo se produce un error de uno que nadie ve.
    """
    if horizonte != HORAS:
        raise ValueError("LEAR es un modelo diario: el horizonte es de 24 horas")
    completos = len(historia) // HORAS
    if completos < max(REZAGOS_DIA) + 2:
        raise ValueError("historia demasiado corta para los rezagos de LEAR")

    dias = np.asarray(historia[len(historia) - completos * HORAS:],
                      dtype=float).reshape(completos, HORAS)
    transformados, centro, escala = _estabiliza(dias.ravel())
    dias_t = transformados.reshape(completos, HORAS)

    ## el día `completos - 1` es el del origen y el `completos` el que se predice
    dia_semana = (dia_semana_origen
                  - (completos - 1 - np.arange(completos + 1))) % 7

    ## se entrena sobre los días que tienen todos sus rezagos y ya están observados,
    ## y se predice el día siguiente al último, que es el que no está en la historia
    primero = max(REZAGOS_DIA)
    entrenamiento = np.arange(primero, completos)
    if dias_calibracion:
        entrenamiento = entrenamiento[-dias_calibracion:]
    X = _matriz_de_dias(dias_t, entrenamiento, dia_semana)
    objetivo = np.array([completos])
    X_futuro = _matriz_de_dias(dias_t, objetivo, dia_semana)

    ## El criterio de información necesita más días que variables —103: cuatro
    ## rezagos diarios a 24 horas más siete de calendario— porque si no, no hay con
    ## qué estimar la varianza del ruido y la penalización queda indefinida. El LEAR
    ## de referencia promedia también ventanas de 56 y 84 días, que caen debajo de
    ## ese límite y exigen una estimación aparte de la varianza; esa variante queda
    ## fuera de aquí y se declara, en vez de resolverse con un número inventado.
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"calibración demasiado corta para el criterio AIC: "
            f"{X.shape[0]} días contra {X.shape[1]} variables; "
            f"hacen falta más de {X.shape[1]}")

    media, desviacion = X.mean(axis=0), X.std(axis=0)
    desviacion[desviacion == 0] = 1.0
    X = (X - media) / desviacion
    X_futuro = (X_futuro - media) / desviacion

    prediccion = np.empty(HORAS)
    for hora in range(HORAS):
        y = dias_t[entrenamiento, hora]
        modelo = LassoLarsIC(criterion="aic").fit(X, y)
        prediccion[hora] = modelo.predict(X_futuro)[0]
    return _deshace(prediccion, centro, escala)


class LEARRodante:
    """
    LEAR con una distribución, construida con sus propios errores ya observados.

    LEAR entrega un número por hora. Para que compita contra un método
    probabilístico hacen falta cuantiles, y la forma honesta de obtenerlos es la
    misma que usan los comparativos ingenuos: los errores que el modelo cometió en
    los días anteriores, paso por paso del horizonte.

    Esos errores no se pueden recalcular en cada origen —serían sesenta ajustes de
    LEAR por pronóstico— así que se acumulan a medida que la corrida avanza. Hay que
    llamar a ``observa`` con lo que de verdad pasó, después de pedir el pronóstico y
    nunca antes. Los primeros días no tienen errores todavía y el intervalo sale
    degenerado: se marcan con ``listo`` y se reportan aparte.
    """

    def __init__(self, dias_calibracion: int = DIAS_CALIBRACION,
                 ventana_errores: int = VENTANA_ERRORES):
        self.dias_calibracion = dias_calibracion
        self.ventana_errores = ventana_errores
        self._errores: list = []
        self._ultimo: Optional[np.ndarray] = None

    @property
    def listo(self) -> bool:
        """Si ya hay errores suficientes para que el intervalo signifique algo."""
        return len(self._errores) >= HORAS

    def predice(self, historia: np.ndarray, horizonte: int,
                levels: Sequence[float], dia_semana_origen: int = 0) -> np.ndarray:
        centro = lear_punto(historia, horizonte, self.dias_calibracion,
                            dia_semana_origen)
        self._ultimo = centro
        errores = np.array(self._errores[-self.ventana_errores:]) if self._errores \
            else np.empty((0, horizonte))
        return _cuantiles_de_errores(centro, errores, levels)

    def observa(self, observado: np.ndarray) -> None:
        """Guarda el error del último pronóstico, una vez que el día ya pasó."""
        if self._ultimo is None:
            raise ValueError("no hay pronóstico pendiente que observar")
        self._errores.append(np.asarray(observado, dtype=float) - self._ultimo)
        self._ultimo = None

    def reinicia(self) -> None:
        """Entre zonas: los errores de una zona no dicen nada de otra."""
        self._errores = []
        self._ultimo = None
