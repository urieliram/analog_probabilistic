"""
PatchTST-FM r2: el modelo de fundación que ya trae su propia distribución.

Es un modelo de series de tiempo preentrenado por IBM Granite que pronostica sin
ajustarse a la serie: se le dan las horas pasadas y devuelve las futuras, sin
entrenar nada —lo que se llama *zero-shot*—. Parte la historia en tramos de
dieciséis horas, uno cada ocho, y los procesa con bloques que combinan atención
multi-cabeza con convolución temporal —bloques *conformer*—, de modo que ve a la
vez la forma del día y la del mes.

Qué entrega
-----------
Noventa y nueve cuantiles por cada hora del horizonte: 0.01, 0.02 y así de
centésima en centésima hasta 0.99. No es una banda calculada después del
pronóstico: la cabeza del modelo emite los noventa y nueve de una sola pasada.
Aquí se piden todos, porque pedir uno y pedir los noventa y nueve cuesta lo mismo
—el modelo los calcula de todos modos—, y porque así 0.05 y 0.95 —las colas que le
importan a quien se cubre contra el precio— salen del modelo y no de una
interpolación entre vecinos.

No entrega muestras de trayectorias. Sólo da cuantiles hora por hora, de manera
que no hay caminos que extraerle: `predice` devuelve `None` en el lugar de los
miembros. Fabricarlos a partir de los cuantiles sería inventar una dependencia
entre horas que el modelo nunca declaró.

Tampoco admite variables externas: no ve la demanda, ni el calendario, ni el
precio del mercado de día en adelanto.

Licencia
--------
Pesos bajo Apache-2.0 más OpenMDW-1.0, permisivas y con uso comercial incluido.
Es la única salida comercial del banco de modelos de fundación: TimesFM-3 y TabPFN
son de uso no comercial, así que sus números sirven para el artículo pero no para
un producto.

Lo que hay que saber para leer sus números
-----------------------------------------
1. **El punto de este modelo es la media, no la mediana.** El pronóstico puntual
   que el envoltorio devuelve —y que la producción del proyecto guarda— es la media
   de la distribución predictiva. El error absoluto se compara contra la mediana,
   que es el cuantil 0.5; hay que pedirlo entre los niveles, porque `predice` no lo
   agrega solo. No da igual: en Caborca, con origen el 15 de junio de 2023, la
   media promedia 1018 pesos por MWh y la mediana 976, y en la hora donde más se
   separan hay 100 pesos entre una y otra.
2. **Cuánta historia ve es una decisión declarada, y cambia el resultado.** Por
   omisión el modelo ve las últimas 2048 horas, que es lo que corre en producción;
   admite hasta 8192. No son equivalentes: en ese mismo origen de Caborca, pasar a
   8192 mueve la mediana 116 pesos por MWh en promedio de las 24 horas, y el cuantil
   0.95 se mueve 207. El valor por omisión no se eligió mirando el tramo de prueba,
   se heredó de producción, y queda abierto en `carga` para quien quiera el otro.
3. **El modelo no mira el calendario.** Se comprobó moviendo el índice de horas
   veinticuatro años hacia atrás: el pronóstico no cambió ni un decimal. El índice
   que `_marco` construye es sintético y sólo existe porque el envoltorio de
   inferencia exige una columna de tiempo.
4. **Los cuantiles no se ordenan a mano.** Salen del modelo y se entregan como
   salen. Si alguna vez se cruzaran, sería señal de que algo está mal configurado, y
   taparlo con un ordenamiento escondería la falla en vez de mostrarla.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd

PESOS = "ibm-granite/granite-timeseries-patchtst-fm-r2"

## Los niveles que la cabeza del modelo emite de fábrica: la rejilla completa de
## centésimas, sin el 0 ni el 1. Cualquier otro nivel —0.025, 0.975— no existe para
## este modelo y sale en NaN.
NIVELES_NATIVOS = tuple(round(0.01 * paso, 2) for paso in range(1, 100))

HORAS_CONTEXTO = 2048
HORAS_CONTEXTO_MAXIMO = 8192

## El envoltorio nombra las columnas de cuantil `y_prediction_q0.05` y así.
_PREFIJO = "y_prediction_q"

## Sólo absorbe el error de representación binaria: 0.15 escrito a mano y 15/100
## calculado no son el mismo número flotante. No sirve para acercar un nivel que el
## modelo no emite al nivel más parecido que sí emite.
_TOLERANCIA = 1e-9

## Hora en la que termina el índice sintético de `_marco`. Es fija a propósito.
_FIN_SINTETICO = pd.Timestamp("2000-01-01 00:00")


class ModeloPatchTST:
    """
    Los pesos ya en memoria, más los envoltorios de inferencia ya construidos.

    El envoltorio de granite-tsfm fija el largo del contexto y el horizonte al
    construirse, así que no puede armarse en `carga`, que todavía no los conoce. Se
    arma en la primera llamada de cada combinación y se reusa: cuesta cinco
    milésimas de segundo contra setenta de un pronóstico, y no vuelve a leer pesos.
    """

    __slots__ = ("pesos", "dispositivo", "horas_contexto", "envoltorios")

    def __init__(self, pesos, dispositivo: str, horas_contexto: int):
        self.pesos = pesos
        self.dispositivo = dispositivo
        self.horas_contexto = horas_contexto
        self.envoltorios = {}

    def envoltorio(self, horas_contexto: int, horizonte: int):
        """El envoltorio de esta combinación de contexto y horizonte."""
        clave = (horas_contexto, horizonte)
        if clave not in self.envoltorios:
            from tsfm_public import TimeSeriesForecastingPipeline

            self.envoltorios[clave] = TimeSeriesForecastingPipeline(
                model=self.pesos,
                context_length=horas_contexto,
                prediction_length=horizonte,
                quantile_levels=list(NIVELES_NATIVOS),
                timestamp_column="ds",
                id_columns=["unique_id"],
                target_columns=["y"],
                freq="h",
                device=self.dispositivo,
            )
        return self.envoltorios[clave]


def carga(dispositivo: str = "cuda", horas_contexto: int = HORAS_CONTEXTO):
    """
    Carga el modelo una vez y devuelve lo que haga falta para predecir.

    `horas_contexto` es cuánta historia ve el modelo en cada origen: 2048 por
    omisión, 8192 como máximo, y el punto 2 de la nota de arriba dice por qué el
    valor importa.

    `torch` y `tsfm_public` se importan aquí y no al principio del archivo para que
    leer `NIVELES_NATIVOS` no arrastre medio gigabyte de dependencias.
    """
    import torch
    from tsfm_public import PatchTSTFMForPrediction

    if horas_contexto > HORAS_CONTEXTO_MAXIMO:
        raise ValueError(
            f"el modelo admite {HORAS_CONTEXTO_MAXIMO} horas de contexto como "
            f"máximo, y se pidieron {horas_contexto}")
    if dispositivo.startswith("cuda") and not torch.cuda.is_available():
        dispositivo = "cpu"
    return ModeloPatchTST(PatchTSTFMForPrediction.from_pretrained(PESOS),
                          dispositivo, int(horas_contexto))


def _marco(contexto: np.ndarray) -> pd.DataFrame:
    """
    La historia en la forma que pide el envoltorio: tiempo, identificador y valor.

    Las horas son sintéticas y terminan siempre en la misma fecha. El modelo no usa
    el calendario —se comprobó: mover el índice veinticuatro años no cambió ni un
    decimal—, de modo que ponerle la fecha real del origen no cambiaría el
    pronóstico y sí daría a entender que el modelo la aprovecha.
    """
    horas = pd.date_range(end=_FIN_SINTETICO, periods=len(contexto), freq="h")
    return pd.DataFrame({"unique_id": "serie", "ds": horas,
                         "y": contexto.astype(np.float32)})


def _niveles_emitidos(prediccion: pd.DataFrame) -> dict:
    """
    Los niveles que el modelo devolvió de hecho, leídos de los nombres de columna.

    Se leen de la salida en vez de darlos por sabidos. Si una versión del paquete
    cambiara la rejilla, los niveles que dejara de emitir saldrían en NaN en lugar
    de salir con el valor de otro nivel.
    """
    return {float(nombre[len(_PREFIJO):]): nombre
            for nombre in prediccion.columns if nombre.startswith(_PREFIJO)}


def _columna_del_nivel(emitidos: dict, nivel: float) -> Optional[str]:
    """
    La columna del nivel pedido, o `None` si el modelo no lo emite.

    Un nivel fuera de la rejilla —0.025, por ejemplo— no se acerca al vecino más
    próximo ni se interpola entre los dos que lo encierran: devuelve `None` y
    termina en NaN. Que un nivel no exista es un hecho del modelo, y el artículo lo
    reporta como tal en vez de rellenarlo.
    """
    for emitido, columna in emitidos.items():
        if abs(emitido - nivel) < _TOLERANCIA:
            return columna
    return None


def predice(modelo: ModeloPatchTST, historia: np.ndarray, horizonte: int,
            niveles: Sequence[float]):
    """
    Devuelve `(cuantiles, miembros)` para las `horizonte` horas tras el origen.

    `historia` son los precios horarios hasta el origen inclusive, y es lo único
    que el pronóstico ve: de ahí se toman las últimas `modelo.horas_contexto`
    horas, o todas si hay menos. Nada del futuro entra por ninguna parte.

    `cuantiles` tiene forma `(len(niveles), horizonte)`, una fila por nivel y en el
    orden pedido. Los niveles que el modelo no emite van en NaN. Si entre los
    niveles viene 0.5, esa fila es la mediana, que es el centro a usar para error
    absoluto; el punto que la producción guarda de este modelo es la media, y la
    media no está aquí.

    `miembros` es siempre `None`: el modelo no entrega trayectorias.
    """
    historia = np.asarray(historia, dtype=float).ravel()
    horas = min(modelo.horas_contexto, len(historia))
    envoltorio = modelo.envoltorio(horas, int(horizonte))
    prediccion = envoltorio(_marco(historia[-horas:]))

    fila = prediccion.iloc[0]
    emitidos = _niveles_emitidos(prediccion)
    cuantiles = np.full((len(niveles), int(horizonte)), np.nan)
    for renglon, nivel in enumerate(niveles):
        columna = _columna_del_nivel(emitidos, float(nivel))
        if columna is not None:
            cuantiles[renglon] = np.asarray(fila[columna], dtype=float)
    return cuantiles, None
