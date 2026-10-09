"""
LEAR: el comparativo que un revisor de pronóstico de precios va a pedir por nombre.

Es el modelo de referencia de la literatura de precios eléctricos —Lago, Marcjasz,
De Schutter y Weron, *Forecasting day-ahead electricity prices: A review of state-of-
the-art algorithms, best practices and an open-access benchmark*, Applied Energy 293
(2021) 116983—. En palabras llanas: para cada hora del día siguiente ajusta una
regresión lineal sobre los precios de los días -1, -2, -3 y -7 a todas sus horas, más
el día de la semana, y deja que un **lazo** —una penalización que empuja a cero los
coeficientes que no ganan su lugar— decida cuáles de esas 103 variables se quedan. Se
reajusta cada ``refit_dias`` días y entre reajustes reusa sus coeficientes; en el
artículo, cada semana (``run_bench.py --refit 7``).

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


class AjusteLEAR:
    """
    Los 24 modelos ya ajustados, más lo que hace falta para aplicarlos.

    Existe porque reajustar cada día cuesta 26 segundos por origen —324 horas para el
    panel entero, trece días y medio— y eso no cabe. Separando ajustar de aplicar, el
    ajuste se reusa entre reajustes y el costo baja en proporción al periodo.

    Se guardan también el centro y la escala de la transformación estabilizadora y la
    estandarización de las variables. Si se recalcularan con la historia nueva, las
    variables dejarían de estar en la escala con la que se ajustaron los coeficientes y
    el pronóstico saldría torcido sin que nada avisara.
    """

    __slots__ = ("modelos", "centro", "escala", "media", "desviacion")

    def __init__(self, modelos, centro, escala, media, desviacion):
        self.modelos = modelos
        self.centro = centro
        self.escala = escala
        self.media = media
        self.desviacion = desviacion


def _dias_y_calendario(historia: np.ndarray, dia_semana_origen: int) -> tuple:
    """
    La historia en forma de días por horas, y el día de la semana de cada uno.

    El día ``completos - 1`` es el del origen y el ``completos`` es el que se predice.
    El calendario se cuenta desde el origen hacia atrás, no desde el principio de la
    historia: la historia se recorta a un número entero de días por el final, así que
    su primera hora cae en un día distinto según el recorte, y amarrar el calendario a
    ella es cómo se produce un error de uno que nadie ve.
    """
    completos = len(historia) // HORAS
    if completos < max(REZAGOS_DIA) + 2:
        raise ValueError("historia demasiado corta para los rezagos de LEAR")
    dias = np.asarray(historia[len(historia) - completos * HORAS:],
                      dtype=float).reshape(completos, HORAS)
    dia_semana = (dia_semana_origen
                  - (completos - 1 - np.arange(completos + 1))) % 7
    return dias, dia_semana, completos


def ajusta_lear(historia: np.ndarray, dias_calibracion: int = DIAS_CALIBRACION,
                dia_semana_origen: int = 0) -> AjusteLEAR:
    """Ajusta los 24 modelos, uno por cada hora del día siguiente."""
    dias, dia_semana, completos = _dias_y_calendario(historia, dia_semana_origen)
    transformados, centro, escala = _estabiliza(dias.ravel())
    dias_t = transformados.reshape(completos, HORAS)

    ## se entrena sobre los días que tienen todos sus rezagos y ya están observados
    entrenamiento = np.arange(max(REZAGOS_DIA), completos)
    if dias_calibracion:
        entrenamiento = entrenamiento[-dias_calibracion:]
    X = _matriz_de_dias(dias_t, entrenamiento, dia_semana)

    ## El criterio de información necesita más días que variables —103: cuatro rezagos
    ## diarios a 24 horas más siete de calendario— porque si no, no hay con qué estimar
    ## la varianza del ruido y la penalización queda indefinida. El LEAR de referencia
    ## promedia también ventanas de 56 y 84 días, que caen debajo de ese límite y
    ## exigen una estimación aparte de la varianza; esa variante queda fuera de aquí y
    ## se declara, en vez de resolverse con un número inventado.
    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"calibración demasiado corta para el criterio AIC: "
            f"{X.shape[0]} días contra {X.shape[1]} variables; "
            f"hacen falta más de {X.shape[1]}")

    media, desviacion = X.mean(axis=0), X.std(axis=0)
    desviacion[desviacion == 0] = 1.0
    X = (X - media) / desviacion

    modelos = [LassoLarsIC(criterion="aic").fit(X, dias_t[entrenamiento, hora])
               for hora in range(HORAS)]
    return AjusteLEAR(modelos, centro, escala, media, desviacion)


def aplica_lear(ajuste: AjusteLEAR, historia: np.ndarray,
                dia_semana_origen: int = 0) -> np.ndarray:
    """
    Aplica un ajuste ya hecho a la historia de hoy.

    La transformación y la estandarización vienen del ajuste y no de la historia nueva:
    es lo que mantiene las variables en la escala de los coeficientes.
    """
    dias, dia_semana, completos = _dias_y_calendario(historia, dia_semana_origen)
    dias_t = np.arcsinh((dias.ravel() - ajuste.centro)
                        / ajuste.escala).reshape(completos, HORAS)
    X = _matriz_de_dias(dias_t, np.array([completos]), dia_semana)
    X = (X - ajuste.media) / ajuste.desviacion
    return _deshace(np.array([m.predict(X)[0] for m in ajuste.modelos]),
                    ajuste.centro, ajuste.escala)


def lear_punto(historia: np.ndarray, horizonte: int = HORAS,
               dias_calibracion: int = DIAS_CALIBRACION,
               dia_semana_origen: int = 0) -> np.ndarray:
    """
    Pronóstico puntual de las 24 horas del día siguiente al origen, ajustando ahora.

    ``historia`` termina en la hora 23 del último día observado, de modo que el día a
    predecir es el siguiente y todos sus rezagos ya están dentro. Es el camino de un
    solo uso: para una corrida larga conviene ``LEARRodante``, que reusa el ajuste.
    """
    if horizonte != HORAS:
        raise ValueError("LEAR es un modelo diario: el horizonte es de 24 horas")
    ajuste = ajusta_lear(historia, dias_calibracion, dia_semana_origen)
    return aplica_lear(ajuste, historia, dia_semana_origen)


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
                 ventana_errores: int = VENTANA_ERRORES,
                 refit_dias: int = 1):
        self.dias_calibracion = dias_calibracion
        self.ventana_errores = ventana_errores
        ## Cada cuántos días se reajustan los 24 modelos. Por omisión, CADA DÍA, que
        ## es lo que hace el LEAR publicado: no hay desviación que declarar.
        ##
        ## Estuvo a punto de haberla. Un ajuste parecía costar 18.5 segundos, lo que daba
        ## 324 horas para el panel entero y forzaba a reajustar cada semana. Pero esos
        ## 18.5 segundos eran un artefacto de los hilos de álgebra: con los hilos libres,
        ## la coordinación domina por completo en una matriz de 723 por 103. Con UN hilo
        ## el mismo ajuste cuesta 0.29 segundos, sesenta y cuatro veces menos, y el panel
        ## entero sale en media hora. Por eso este módulo se corre con OMP_NUM_THREADS=1.
        self.refit_dias = refit_dias
        self._errores: list = []
        self._ultimo: Optional[np.ndarray] = None
        self._ajuste: Optional[AjusteLEAR] = None
        self._desde_ajuste = 0
        self.ajustes = 0

    @property
    def listo(self) -> bool:
        """Si ya hay errores suficientes para que el intervalo signifique algo."""
        return len(self._errores) >= HORAS

    def predice(self, historia: np.ndarray, horizonte: int,
                levels: Sequence[float], dia_semana_origen: int = 0) -> np.ndarray:
        if horizonte != HORAS:
            raise ValueError("LEAR es un modelo diario: el horizonte es de 24 horas")
        if self._ajuste is None or self._desde_ajuste >= self.refit_dias:
            self._ajuste = ajusta_lear(historia, self.dias_calibracion,
                                       dia_semana_origen)
            self._desde_ajuste = 0
            self.ajustes += 1
        self._desde_ajuste += 1
        centro = aplica_lear(self._ajuste, historia, dia_semana_origen)
        self._ultimo = centro
        errores = np.array(self._errores[-self.ventana_errores:]) if self._errores \
            else np.empty((0, horizonte))
        return _cuantiles_de_errores(centro, errores, levels)

    def trayectorias(self) -> Optional[np.ndarray]:
        """
        El centro del último pronóstico más cada vector de error ya observado.

        Se emite así para que LEAR reciba el mismo tratamiento que los métodos con
        muestra: una trayectoria por error pasado, con su forma completa. Devuelve
        ``None`` mientras no haya errores, que es cuando el intervalo sería un punto.
        """
        if self._ultimo is None or not self._errores:
            return None
        errores = np.array(self._errores[-self.ventana_errores:])
        return self._ultimo[None, :] + errores

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
        self._ajuste = None
        self._desde_ajuste = 0
