"""
TiRex: nueve cuantiles por hora, y ninguna cola.

Qué es. Un modelo de pronóstico de series ya entrenado —de los que se llaman modelos
de fundación, entrenados una vez sobre muchísimas series y aplicados a una serie nueva
sin volver a ajustarlos— publicado por NXAI y construido sobre xLSTM. Aquí no se
entrena nada: se le entregan los precios de las últimas horas y devuelve el día
siguiente. Los pesos son NX-AI/TiRex, 35 millones de parámetros en 141 MB, y se leen
de la caché local de Hugging Face; este archivo nunca los copia al repositorio.

Qué entrega, exactamente. Nueve cuantiles por hora y nada más: 0.1, 0.2, ... 0.9. Los
nueve vienen grabados en el punto de control —la última capa tiene nueve salidas por
cada paso del horizonte— así que no son un argumento que se pueda cambiar al llamarlo.
No existe el 0.05 ni el 0.95. Este adaptador devuelve NaN en los niveles que se le
pidan y el modelo no emita, y no los interpola ni los extrapola: que un nivel no
exista es un hecho del modelo y así se reporta. Tampoco entrega trayectorias, de modo
que `predice` devuelve ``None`` donde otros métodos devuelven su muestra. Armar
muestras a partir de nueve cuantiles por hora sería inventar la dependencia entre
horas, que es precisamente una de las cosas que el artículo mide.

Cómo leer sus números. Su banda más ancha es de 0.1 a 0.9, el 80% central. Para quien
compra energía y se cubre, el número que decide es el extremo alto —el 0.95—, y de ese
TiRex no dice nada. La celda vacía en su renglón de la tabla no es un cero ni una
corrida que falló: es que el modelo no tiene nada que decir ahí. Y lo que su interfaz
llama "media" es la mediana: el propio código la toma de la columna del cuantil 0.5,
así que no hay un valor esperado aparte de los nueve. Este adaptador la descarta para
no dar dos nombres al mismo número.

Cuánta historia mira. Sólo las últimas 2048 horas, 85 días, por mucha historia que se
le pase: el modelo recorta por su cuenta a la ventana con la que fue entrenado. La
función recorta igual antes de llamarlo, para que el recorte se vea en el código y
para no mover dos años de datos a la tarjeta en cada pronóstico. Verificado el
2026-10-06 con la zona caborca: el pronóstico con 17520 horas de historia y el
pronóstico con las últimas 2048 salen idénticos, cuantil por cuantil.

Licencia. NXAI Community License Agreement, que no es Apache ni MIT. Permite usar,
copiar, modificar y también vender lo que se haga con los pesos, con dos condiciones
que conviene tener presentes antes de llevar esto a producción: quien facture más de
cien millones de euros al año e incorpore los pesos a un producto o servicio comercial
necesita una licencia aparte de NXAI, y quien redistribuya los pesos o un derivado
tiene que mostrar "Built with technology from NXAI". Para publicar un artículo no
estorba.

Equipo. Corre con el paquete tirex-ts, probado con la versión 1.4.1, y pide una tarjeta
con capacidad de cómputo 8.0 o más. Medido el 2026-10-06 en esta máquina, una RTX 5060
Ti de capacidad 12.0: 0.21 segundos por pronóstico de 24 horas en la tarjeta contra 50
segundos en el procesador, 238 veces más. Y tampoco salen los mismos números: los
cuantiles de una misma ventana difieren hasta en 18 pesos por MWh entre tarjeta y
procesador, sobre bandas de unos 1300, así que una corrida hecha a medias entre los dos
no es comparable consigo misma. Por eso `carga` falla cuando se le pide GPU y no hay
una que sirva, en lugar de cambiar sola al procesador sin que nada avise. Quien quiera
procesador lo pide con ``carga(dispositivo="cpu")``.
"""

from typing import Sequence

import numpy as np

PESOS = "NX-AI/TiRex"
NIVELES_NATIVOS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
CAPACIDAD_MINIMA = 8.0


def carga(dispositivo: str = "cuda"):
    """
    Carga el modelo una vez y devuelve lo que haga falta para predecir.

    Los imports van adentro porque este adaptador no se usa en todas las corridas del
    repositorio y el resto no tiene por qué exigir torch ni tirex instalados.
    """
    import torch
    from huggingface_hub.errors import LocalEntryNotFoundError
    from tirex import load_model

    if dispositivo.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("se pidió GPU y PyTorch no detecta ninguna")
        mayor, menor = torch.cuda.get_device_capability(0)
        if mayor + menor / 10 < CAPACIDAD_MINIMA:
            raise RuntimeError(
                f"TiRex requiere capacidad de cómputo {CAPACIDAD_MINIMA} y esta "
                f"tarjeta tiene {mayor}.{menor}"
            )

    try:
        modelo = load_model(PESOS, device=dispositivo, backend="torch",
                            hf_kwargs={"local_files_only": True})
    except LocalEntryNotFoundError:
        ## la caché de Hugging Face ya tiene los pesos en esta máquina y por eso se
        ## piden sin red; en una máquina nueva no están y hay que bajarlos una vez
        modelo = load_model(PESOS, device=dispositivo, backend="torch")

    ## NIVELES_NATIVOS es dos cosas a la vez: lo que el artículo reporta como rejilla
    ## de este modelo y el orden de las columnas que devuelve. Si un punto de control
    ## trajera otra rejilla, las columnas se leerían corridas y los cuantiles saldrían
    ## mal sin que nada avisara
    if tuple(modelo.config.quantiles) != NIVELES_NATIVOS:
        raise RuntimeError(
            "el punto de control emite los cuantiles "
            f"{tuple(modelo.config.quantiles)} y este adaptador declara "
            f"{NIVELES_NATIVOS}"
        )
    return modelo


def predice(modelo, historia: np.ndarray, horizonte: int,
            niveles: Sequence[float]) -> tuple:
    """
    Devuelve (cuantiles, miembros).

    historia: arreglo de numpy de una dimensión con los precios horarios hasta el
              origen inclusive. El origen es la última hora observada.
    horizonte: 24.
    niveles: lista de niveles de cuantil pedidos, por ejemplo
             [0.05, 0.1, 0.15, ... 0.9, 0.95].

    cuantiles: arreglo de numpy de forma (len(niveles), horizonte), una fila por nivel
               en el orden pedido. Los niveles que el modelo no emite de fábrica van
               en NaN, sin interpolar ni extrapolar.
    miembros: siempre ``None``, porque TiRex no entrega trayectorias.
    """
    contexto = np.asarray(historia, dtype=np.float32)[-modelo.config.train_ctx_len:]

    ## el segundo valor que devuelve forecast se llama "mean" en la interfaz de TiRex
    ## pero es la columna del cuantil 0.5, que ya viene en el primero
    nativos, _ = modelo.forecast(
        context=contexto[None, :],
        prediction_length=horizonte,
        output_type="numpy",
    )
    nativos = np.asarray(nativos[0])

    columna_de = {round(nivel, 6): j for j, nivel in enumerate(NIVELES_NATIVOS)}
    cuantiles = np.full((len(niveles), horizonte), np.nan)
    for i, nivel in enumerate(niveles):
        j = columna_de.get(round(float(nivel), 6))
        if j is not None:
            cuantiles[i] = nativos[:, j]

    return cuantiles, None
