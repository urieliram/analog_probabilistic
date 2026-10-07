"""
t0-beta: veintiún cuantiles por hora, y ninguna trayectoria.

Qué es. Un modelo de pronóstico de series ya entrenado —de los que se llaman modelos de
fundación: se entrenan una vez sobre muchísimas series y se aplican a una serie nueva
sin volver a ajustarlos, lo que se llama cero disparos o *zero-shot*— publicado por The
Forecasting Company. Son 256 millones de parámetros (arXiv:2609.24559) que leen la
serie en parches de 32 horas. Los pesos son theforecastingcompany/t0-beta y se toman de
la caché local de Hugging Face; este archivo nunca los copia al repositorio.

Qué entrega, exactamente. Veintiún cuantiles por hora: 0.01, de 0.05 a 0.95 en pasos de
0.05, y 0.99. Son los niveles con los que fue entrenado —la configuración de los pesos
los trae listados— y son los únicos que este adaptador devuelve con un número. El
paquete sabe interpolar los niveles de en medio y sabe extrapolar las colas más allá de
0.01 y 0.99; aquí las dos cosas quedan apagadas a propósito. Un 0.975 sacado entre el
0.95 y el 0.99 del modelo no es una medición del modelo: es una recta nuestra puesta en
su nombre. Los niveles que se le pidan y no tenga salen en NaN. Tampoco entrega
trayectorias: no publica muestras de caminos, así que no se puede saber si la hora en
que el precio sube alto viene acompañada de otra alta a la hora siguiente, y ``predice``
devuelve ``None`` donde otros métodos devuelven su muestra. Fabricar muestras a partir
de los cuantiles inventaría esa dependencia entre horas, que es justo una de las cosas
que el artículo mide.

Cómo leer sus números. Cubre los nueve niveles de la comparación en crudo, los nueve, y
once de los trece de la rejilla completa: le faltan el 0.025 y el 0.975. La celda vacía
en esos dos no es un cero ni una corrida que falló; es que el modelo no emite ahí. Los
precios se le pasan como vienen, en pesos por megavatio-hora, con sus picos y sus
valores negativos: normaliza por su cuenta con una transformación de seno hiperbólico
inverso, así que no hay que estabilizarlos antes, al contrario de lo que necesita una
regresión lineal. Un hueco se le puede pasar como NaN, y lo lee como observación
ausente en vez de como un cero.

Licencia. Pesos y paquete bajo Apache-2.0, que permite el uso comercial. Es lo que lo
separa de los modelos de fundación del banco cuyos pesos son sólo para investigación:
el número de t0-beta se puede llevar a un producto y el de ellos no.

Cuánto cuesta un pronóstico. Medido el 2026-10-06 en este equipo —un Core i7-11700 de
ocho hilos y una RTX 5060 Ti—, con 2048 horas de contexto y horizonte de 24: 0.03 s en
la tarjeta, y entre 22 y 45 s en el procesador según lo ocupada que esté la máquina. En
los dos casos sin contar la primera llamada, que tarda un segundo más porque inicializa
los núcleos de cálculo. Son casi mil veces de diferencia, y eso decide el dispositivo:
el banco completo en procesador no cabe en el tiempo del artículo.
"""

from typing import Optional, Sequence

import numpy as np
import torch
from t0 import T0Forecaster

PESOS = "theforecastingcompany/t0-beta"

## La rejilla entrenada. Verificada el 2026-10-06 contra la configuración de los pesos,
## ``model.config.quantile_levels``: coincide nivel por nivel.
NIVELES_NATIVOS = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)

## Horas de historia que entran al modelo, el mismo valor que usa el pipeline de
## producción del proyecto. No se eligió por tiempo: medido el 2026-10-06, pasar de 1024
## a 4096 horas sube el costo de 20.6 s a 24.9 s por pronóstico en el procesador. Se
## conserva el valor de producción en vez de estrenar uno que aquí no se evaluó contra
## el acierto. El tope de 1024 que trae el paquete es del horizonte, no del contexto.
HORAS_DE_CONTEXTO = 2048

## Los niveles llegan de aritmética de punto flotante —0.05 sumado tres veces no da 0.15
## exacto— así que la comparación contra la rejilla lleva holgura. La separación más
## corta de la rejilla es 0.04, entre 0.01 y 0.05: millones de veces mayor que esta
## holgura, de modo que no hay manera de confundir un nivel con su vecino.
HOLGURA_DE_NIVEL = 1e-9


def _posicion_nativa(nivel: float) -> Optional[int]:
    """Dónde cae el nivel en la rejilla nativa, o ``None`` si el modelo no lo emite."""
    for posicion, nativo in enumerate(NIVELES_NATIVOS):
        if abs(nivel - nativo) <= HOLGURA_DE_NIVEL:
            return posicion
    return None


def carga(dispositivo: str = "cuda"):
    """
    Carga los pesos una vez y devuelve el modelo listo para predecir.

    No hace falta devolver el dispositivo aparte: el modelo lo lleva consigo y
    ``predice`` lo lee de sus propios parámetros. Si se pide la tarjeta y no hay, se cae
    al procesador, que entrega los mismos cuantiles —iguales a un decimal en la
    verificación del 2026-10-06— cientos de veces más despacio.
    """
    if dispositivo.startswith("cuda") and not torch.cuda.is_available():
        dispositivo = "cpu"
    return T0Forecaster.from_pretrained(PESOS).eval().to(dispositivo)


def predice(modelo, historia: np.ndarray, horizonte: int,
            niveles: Sequence[float]) -> tuple:
    """
    Devuelve ``(cuantiles, miembros)`` para las ``horizonte`` horas tras el origen.

    ``historia`` es la serie horaria de precios hasta el origen inclusive, y es lo único
    que el modelo ve: el origen es su última hora. De ese final se toman las
    ``HORAS_DE_CONTEXTO`` últimas horas.

    ``cuantiles`` tiene forma ``(len(niveles), horizonte)``, una fila por nivel en el
    orden en que se pidieron. Las filas de los niveles que el modelo no trae entrenados
    quedan en NaN, sin interpolar ni extrapolar.

    ``miembros`` es siempre ``None``: t0-beta no publica muestras de trayectoria.

    Se le pide siempre la rejilla nativa completa y de ahí se toman los niveles que
    hagan falta. Verificado el 2026-10-06 sobre Caborca: pedir los veintiún niveles y
    pedir cinco de ellos devuelve los mismos valores, con diferencia máxima de 0, así
    que pedir de más no mueve ningún número.
    """
    contexto = np.asarray(historia, dtype=np.float32)[-HORAS_DE_CONTEXTO:]
    dispositivo = next(modelo.parameters()).device
    with torch.no_grad():
        emitidos = modelo.predict(
            torch.tensor(contexto[None, :], device=dispositivo),
            horizon=int(horizonte),
            quantile_levels=list(NIVELES_NATIVOS),
        ).quantiles
    ## del modelo sale (1, horizonte, 21) y aquí se quiere (21, horizonte)
    nativos = emitidos[0].float().cpu().numpy().T

    cuantiles = np.full((len(niveles), int(horizonte)), np.nan)
    for fila, nivel in enumerate(niveles):
        posicion = _posicion_nativa(float(nivel))
        if posicion is not None:
            cuantiles[fila] = nativos[posicion]
    return cuantiles, None
