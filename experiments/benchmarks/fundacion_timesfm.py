"""
TimesFM 2.5 de Google: nueve cuantiles por hora y ninguna trayectoria.

Qué entrega. Por cada hora del horizonte el modelo saca nueve números: los cuantiles
0.1, 0.2 y así hasta 0.9. Eso es todo lo que tiene. No entrega muestras de caminos
posibles del día, de modo que `predice` devuelve None en el lugar de los miembros y los
puntajes de trayectoria no se le pueden calcular. Los niveles que el artículo pide
fuera de esa rejilla -0.05, 0.15 y los demás múltiplos impares de 0.05- salen en NaN.
No se interpolan: que el modelo no tenga un nivel es un hecho del modelo, y el artículo
lo reporta tal cual.

Qué versión, y por qué. El paquete instalado es timesfm 3.0.1, pero aquí se cargan los
pesos de la 2.5. La nota de licencia del propio paquete lo explica: el código es
Apache-2.0 y los pesos hasta la 2.5 también, mientras los de la 3.0 se distribuyen bajo
timesfm-non-commercial-license-v1.0, limitados a uso no comercial y fuera de
producción. Esa limitación alcanza a lo que el modelo produce, así que un número salido
de la 3.0 no podría entrar al servicio.

Cómo leer sus números. El abanico depende de cinco banderas de inferencia, y tres de
las que usa la tarjeta oficial del modelo no son el valor por omisión del paquete. Lo
que queda declarado aquí, con lo medido el 2026-10-06 sobre la zona caborca en 60
orígenes de 24 horas que terminan el 2023-06-15, esto es, 11520 pares de cuantiles
vecinos:

  - Cabeza continua de cuantiles y arreglo de cruces, las dos encendidas, como en la
    tarjeta. Un cruce es un cuantil alto que queda por debajo de uno más bajo, el 0.6
    debajo del 0.5, y es justo lo que el artículo mide. Con las dos encendidas: cero
    cruces de 11520. Con la cabeza continua sola: 71. Con las dos apagadas, que es el
    valor por omisión del paquete: 20.
  - Invariancia al signo, encendida, que sí es el valor por omisión. Promedia el
    pronóstico de la serie con el de la misma serie con el signo volteado. Cuesta el
    doble de tiempo, 0.10 s contra 0.05 s por pronóstico, y mueve cuantiles hasta
    194 MXN/MWh, así que no es un detalle cosmético.
  - Piso en cero, apagada, y es la única desviación respecto de la tarjeta. La bandera
    recorta la salida en cero cuando toda la ventana de entrada es no negativa, y el
    precio de este mercado sí puede ser negativo: 39 de las 17520 horas de caborca
    hasta el 2023-06-15 lo son. Un piso que el mercado no tiene apretaría el cuantil
    bajo, que es parte de lo que se está midiendo. Apagarla no movió ningún número en
    los 40 orígenes probados: diferencia máxima de 0.0000 MXN/MWh.
  - Normalización de la entrada, encendida como en la tarjeta. Movió a lo más 0.0007
    MXN/MWh, consistente con que el modelo ya es invariante a la escala; se deja porque
    protege de magnitudes extremas y no cuesta nada.

El contexto son 2048 horas, unas doce semanas. Es una elección declarada y no ajustada:
escogerla por su error en el tramo de prueba sería decidir mirando lo que se quiere
medir. El tiempo no la limita; medido en una RTX 5060 Ti, un pronóstico tarda 0.06 s
con 512 horas de contexto, 0.10 s con 2048 y 0.24 s con 8192.

Hace falta GPU. Ese mismo pronóstico de 24 horas tarda 0.10 s en la RTX 5060 Ti y unos
50 s en CPU, quinientas veces más, así que el banco completo no es viable en CPU.
"""

from typing import Optional, Sequence

import numpy as np

REPO_PESOS = "google/timesfm-2.5-200m-pytorch"
NIVELES_NATIVOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CONTEXTO_HORAS = 2048

## El modelo decodifica en bloques de 128 horas y compila al múltiplo que cubra el
## horizonte pedido, así que un horizonte de 24 se compila igual a 128.
HORIZONTE_COMPILADO = 128

## Los niveles llegan como flotantes de otro cálculo, y 0.1 puede venir como
## 0.10000000000000002. Se comparan por cercanía, no por igualdad.
TOLERANCIA_NIVEL = 1e-6


def carga(dispositivo: str = "cuda"):
    """Carga los pesos de la 2.5 desde el caché y compila el modelo para inferencia."""
    import timesfm
    import torch
    from huggingface_hub import hf_hub_download
    from timesfm.configs import ForecastConfig

    try:
        pesos = hf_hub_download(
            repo_id=REPO_PESOS,
            filename=timesfm.TimesFM_2p5_200M_torch.WEIGHTS_FILENAME,
            local_files_only=True,
        )
    except Exception:
        ## Primero el caché y sin red; sólo si los pesos no están se bajan, una vez.
        pesos = hf_hub_download(
            repo_id=REPO_PESOS,
            filename=timesfm.TimesFM_2p5_200M_torch.WEIGHTS_FILENAME,
        )

    ## Sin compilación JIT de torch: 0.10 s por pronóstico ya alcanza, y compilar
    ## agrega una espera larga en la primera llamada de cada proceso.
    modelo = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)

    if dispositivo != "cuda" or not torch.cuda.is_available():
        ## El módulo elige cuda:0 en su constructor. Para correr en CPU hay que
        ## cambiárselo antes de cargar los pesos, que es cuando los mueve al aparato.
        modelo.model.device = torch.device("cpu")
        modelo.model.device_count = 1

    modelo.load_checkpoint(pesos)
    modelo.compile(ForecastConfig(
        max_context=CONTEXTO_HORAS,
        max_horizon=HORIZONTE_COMPILADO,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        fix_quantile_crossing=True,
        force_flip_invariance=True,
        infer_is_positive=False,
    ))
    return modelo


def predice(modelo, historia: np.ndarray, horizonte: int,
            niveles: Sequence[float]) -> tuple:
    """
    Cuantiles del modelo para las próximas ``horizonte`` horas.

    ``historia`` son los precios horarios hasta el origen inclusive; el modelo sólo ve
    las últimas ``CONTEXTO_HORAS``. Devuelve el arreglo de cuantiles, con NaN en los
    niveles que el modelo no emite, y None en los miembros, porque no entrega muestras
    de trayectorias.
    """
    contexto = np.asarray(historia, dtype=float).ravel()[-CONTEXTO_HORAS:]
    _, abanico = modelo.forecast(horizon=horizonte, inputs=[contexto])

    ## abanico[0] tiene forma (horizonte, 10). La columna 0 es la cabeza puntual del
    ## modelo y no un cuantil; las columnas 1 a 9 son los niveles 0.1 a 0.9 en orden.
    nativos = np.asarray(abanico[0])[:, 1:].T

    nativos_pedidos = np.asarray(NIVELES_NATIVOS)
    cuantiles = np.full((len(niveles), horizonte), np.nan)
    for fila, nivel in enumerate(niveles):
        coincide = np.abs(nativos_pedidos - float(nivel)) < TOLERANCIA_NIVEL
        if coincide.any():
            cuantiles[fila] = nativos[int(np.argmax(coincide))]

    miembros: Optional[np.ndarray] = None
    return cuantiles, miembros
