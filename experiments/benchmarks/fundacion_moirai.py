"""
Moirai: el único del banco que entrega una muestra, pero sin atar las horas.

Qué es. Un transformador preentrenado sobre millones de series de otros dominios que
pronostica sin reentrenar nada —cero disparos, *zero-shot*: los pesos llegan hechos y
aquí no se ajusta ninguno—. Lee la serie en trozos de varias horas, llamados parches, y
de ahí escribe una distribución para el día siguiente.

Qué entrega. No entrega cuantiles: entrega mil valores por cada hora del día siguiente,
en una matriz de mil por veinticuatro. Es la única salida del banco con esa forma, y por
eso la única a la que se le pueden calcular los puntajes que miden caminos completos.

Qué NO significa esa matriz, y es lo primero que hay que saber. **Las horas de un mismo
miembro no están atadas entre sí.** Moirai pronostica las 24 horas en una sola pasada y
su cabeza entrega, para cada hora por separado, una mezcla de cuatro distribuciones; al
muestrear, cada hora saca su propio componente y su propio valor sin mirar lo que sacó
la hora anterior. Está en el código de la biblioteca y se midió aquí, en Caborca con mil
miembros:

    correlación entre horas vecinas dentro de los miembros      -0.01
    dispersión de la media diaria del miembro / la que daría
      la independencia                                           1.01
    lo mismo en los precios observados de los últimos 120 días   +0.82  y  2.93

Es decir: cada miembro es un sorteo del producto de las 24 distribuciones de cada hora,
no un día posible. Se probó con los cuatro tamaños de parche del modelo y con la
selección automática; en todos sale igual, y la biblioteca no ofrece un muestreo
conjunto en esta versión.

Qué hacer con eso. Los miembros se devuelven enteros y sin colapsar, porque son la
salida del modelo: reducirlos a media y tres pares de cuantiles —lo que hace el pipeline
de producción— tiraría lo que sí hay, que es la distribución completa de cada hora. Lo
que no se puede es leer su puntaje de energía o de variograma como el de un método que
propone días posibles. Contra el análogo, que arma sus miembros con días observados
enteros, es de esperarse que Moirai pierda en los puntajes de camino por construcción, y
el artículo tiene que reportarlo como una propiedad de cómo muestrea Moirai y no como
que pronostica peor.

Qué no ve. Sólo los precios de las últimas 720 horas. Ni el calendario —no sabe si el
día que pronostica es domingo—, ni la demanda, ni la generación, ni de qué nodo se
trata. Se verificó en la biblioteca: la transformación de entrada de Moirai no agrega
una sola variable de calendario. El comparativo ingenuo y el LEAR sí usan el día de la
semana; Moirai no, y eso hay que tenerlo presente al leer su puntaje.

Licencia. Los pesos son CC-BY-NC-4.0: sólo uso no comercial. No se copian ni se
empaquetan en este repositorio; se leen de la caché local de Hugging Face.

Cuatro cosas más para leer sus números
--------------------------------------
1. Son un sorteo, y el sorteo se nota. Cada llamada vuelve a muestrear, así que dos
   corridas sobre el mismo origen no dan el mismo número. Medido sobre un origen de
   Caborca, en dos tandas de ocho repeticiones: con mil miembros el cuantil del 5% se
   mueve del 11 al 14% de su valor entre llamadas y la media un 2%; con cien miembros,
   del 29 al 39% y un 6%. Quien necesite repetir una corrida miembro a miembro fija la
   semilla de torch antes de llamar; este archivo no la toca, para no alterarle el
   sorteo a otro modelo que corra en el mismo proceso.
2. Mil miembros, no cien, y la razón es la cola. GluonTS no interpola entre muestras:
   para el nivel q toma la muestra ordenada en la posición redondeada de (n-1)·q. Con
   cien miembros el 5% es la sexta muestra más baja y la cola queda medida con el ruido
   de dos o tres observaciones; con mil es la muestra 51 de 1000. Medido aquí, subir de
   cien a mil cuesta menos de un 10% de tiempo, no el 40% que uno esperaría: el tiempo
   se va en elegir el tamaño de parche, no en muestrear.
3. El tamaño de parche lo elige el modelo. Con `patch_size='auto'`, uni2ts prueba los
   cinco tamaños del modelo —8, 16, 32, 64 y 128 horas—, puntúa cada uno pronosticando
   las últimas 24 horas YA OBSERVADAS antes del origen y se queda con el de menor
   pérdida. No hay fuga: esas 24 horas son pasado. Cuesta unas diez veces lo que un
   parche fijo, porque muestrea una vez por candidato, y aun así son 0.4 segundos por
   pronóstico en tarjeta. Se paga a cambio de no elegir a mano un número que el artículo
   tendría que defender.
4. Aquí no se recorta nada. El pipeline de producción acota los precios a los límites
   del mercado antes y después de pronosticar; este adaptador no, ni por abajo ni por
   arriba. Lo que se reporta es lo que el modelo emitió, incluidos los miembros que
   caigan fuera de esos límites.
"""

import warnings
from typing import Sequence

import numpy as np
import pandas as pd

PESOS = "Salesforce/moirai-1.0-R-large"
CONTEXTO_HORAS = 720
NUM_MUESTRAS = 1000
TAMANO_PARCHE = "auto"

## El lote de GluonTS cuenta series por pasada, no miembros. Como cada llamada manda un
## solo origen, 32 no acelera nada: ganaría tiempo de tarjeta sólo si se le pasaran
## varios orígenes juntos en un mismo conjunto, y el contrato de `predice` es de uno en
## uno. Se declara para que quede dicho y nadie le atribuya una ganancia que no hubo.
LOTE = 32

## Moirai no emite una rejilla fija de cuantiles: emite muestras. De una muestra de mil,
## cualquier nivel entre 0.001 y 0.999 es una posición de la muestra ordenada del propio
## modelo, no una interpolación entre cuantiles emitidos. Esta lista son los 19 niveles
## que pide el artículo, todos dentro de ese rango: para Moirai ninguna fila sale en
## NaN. La constante existe para hacerle a cada adaptador del banco la misma pregunta.
NIVELES_NATIVOS = tuple(round(0.05 * k, 2) for k in range(1, 20))

## Moirai no mira el calendario, así que la fecha de arranque que GluonTS exige no entra
## al pronóstico. Se usa una fija en vez de la verdadera para que nadie lea en el código
## que el modelo sabe en qué día está parado.
ARRANQUE = pd.Period("2000-01-01 00:00", freq="h")


class ModeloMoirai:
    """Los pesos ya en memoria, y un predictor de GluonTS armado por horizonte."""

    def __init__(self, pesos, dispositivo: str, contexto: int, num_muestras: int):
        self.pesos = pesos
        self.dispositivo = dispositivo
        self.contexto = contexto
        self.num_muestras = num_muestras
        self._predictores = {}

    def predictor(self, horizonte: int):
        """El predictor de ese horizonte; se arma la primera vez y se reusa."""
        if horizonte not in self._predictores:
            from uni2ts.model.moirai import MoiraiForecast

            pronosticador = MoiraiForecast(
                module=self.pesos,
                prediction_length=horizonte,
                context_length=self.contexto,
                patch_size=TAMANO_PARCHE,
                num_samples=self.num_muestras,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            self._predictores[horizonte] = pronosticador.create_predictor(
                batch_size=LOTE, device=self.dispositivo,
            )
        return self._predictores[horizonte]


def carga(dispositivo: str = "cuda", contexto: int = CONTEXTO_HORAS,
          num_muestras: int = NUM_MUESTRAS) -> ModeloMoirai:
    """
    Trae los pesos a memoria una sola vez y deja listo lo necesario para predecir.

    Los pesos salen de la caché local de Hugging Face; no se copian al repositorio ni
    se vuelven a bajar si ya están ahí.

    ``contexto`` son las horas de historia que el modelo lee en cada pronóstico. Las 720
    de omisión —treinta días, cuatro ciclos semanales— son las del pipeline del
    proyecto, no un número ajustado aquí: ajustarlo mirando el tramo de prueba
    inclinaría la comparación a favor de Moirai.
    """
    ## torch y uni2ts tardan cuatro segundos en importarse. Se importan dentro para que
    ## listar los adaptadores del banco no los pague si Moirai no se va a correr.
    import torch
    from uni2ts.model.moirai import MoiraiModule

    if str(dispositivo).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            "no hay tarjeta disponible: Moirai corre en CPU y cada pronóstico tarda "
            "bastante más que los 0.4 segundos de referencia",
            stacklevel=2,
        )
        dispositivo = "cpu"

    pesos = MoiraiModule.from_pretrained(PESOS).to(dispositivo)
    return ModeloMoirai(pesos, dispositivo, contexto, num_muestras)


def predice(modelo: ModeloMoirai, historia: np.ndarray, horizonte: int,
            niveles: Sequence[float]) -> tuple:
    """
    Devuelve (cuantiles, miembros).

    ``historia`` es un arreglo de una dimensión con los precios horarios hasta el origen
    inclusive: la última hora observada es el último elemento. ``niveles`` son los
    niveles de cuantil pedidos.

    ``cuantiles`` tiene forma (len(niveles), horizonte), una fila por nivel en el orden
    pedido. Cada fila es el corte hora por hora de la muestra: la posición de la muestra
    ordenada que GluonTS asigna a ese nivel, sin interpolar. Un nivel que la muestra no
    alcanza a resolver —más fino que una posición, con mil miembros por debajo de 0.0005
    o por encima de 0.9995— sale en NaN, porque ahí la respuesta sería el mínimo o el
    máximo de la muestra y eso no es una estimación de ese cuantil.

    ``miembros`` tiene forma (num_muestras, horizonte): la muestra del modelo entera,
    sin colapsar. Los cuantiles salen de estos mismos miembros, así que las dos salidas
    son consistentes entre sí. Las horas de un miembro no están atadas entre sí —ver la
    nota del principio, con la medición—, así que un puntaje de trayectoria calculado
    sobre ellos mide una muestra sin estructura de camino.
    """
    from gluonts.dataset.common import ListDataset

    historia = np.asarray(historia, dtype=np.float32).ravel()
    minimo = modelo.contexto + horizonte
    if len(historia) < minimo:
        raise ValueError(
            f"Moirai necesita {minimo} horas de historia ({modelo.contexto} de "
            f"contexto más {horizonte} para elegir el tamaño de parche) y recibió "
            f"{len(historia)}; con menos, GluonTS rellenaría con ceros, que en una "
            f"serie de precios son un precio y no un hueco"
        )

    ## La ventana que se le pasa son las últimas contexto+horizonte horas, todas pasado
    ## observado. El modelo condiciona el pronóstico en las últimas `contexto` horas,
    ## las que terminan en el origen; las `horizonte` horas finales son además la
    ## ventana con la que 'auto' puntúa los tamaños de parche.
    ventana = historia[-minimo:]
    conjunto = ListDataset(
        [{"start": ARRANQUE, "target": ventana}], freq="h",
    )
    pronostico = next(modelo.predictor(horizonte).predict(conjunto))

    miembros = np.asarray(pronostico.samples, dtype=float)
    resolucion = 0.5 / (len(miembros) - 1)
    cuantiles = np.full((len(niveles), horizonte), np.nan)
    for fila, nivel in enumerate(niveles):
        if resolucion <= float(nivel) <= 1.0 - resolucion:
            cuantiles[fila] = pronostico.quantile(float(nivel))
    return cuantiles, miembros
