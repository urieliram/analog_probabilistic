"""
MOMENT-1-large: un pronóstico puntual por hora, y ninguna trayectoria.

MOMENT (Goswami et al., ICML 2024; pesos AutonLab/MOMENT-1-large) es un codificador T5 que
recibe siempre 512 pasos, los corta en parches de 8 y reconstruye los parches que se le
ocultan. De fábrica sólo trae entrenada la cabeza de reconstrucción, no la de pronóstico,
así que el pronóstico sin entrenamiento consiste en ocultarle las horas futuras y pedirle
que las reconstruya (``short_forecast``). Con horizonte 24 se ocultan 24 pasos y quedan
488 horas de historia.

El horizonte es 24, como para todos los modelos del banco. La producción del proyecto
(LLM11) usa 72 y recorta; en MOMENT eso es otro pronosticador, porque el horizonte decide
cuántas horas se ocultan y cuánta historia ve el modelo.

Sólo da un número por hora. Se guarda como la mediana para que entre a la receta de
escenarios de sus errores (mediana más los errores de los últimos sesenta días), igual
que los demás modelos sin escenarios.

Disposición de la ventana, la misma de la producción:

    ventana de 512:   [ hueco de 24 | historia de 488 ]
    input_mask:         0 ......... 0   1 ........... 1

Licencia. Pesos bajo MIT, que permite el uso comercial.

Los pesos se verifican al cargar, tensor por tensor, contra el archivo: con algunas
versiones de ``transformers`` el modelo nombra distinto sus capas y queda con capas al
azar sin que la carga avise.
"""

import math
from typing import Sequence

import numpy as np
import torch

PESOS = "AutonLab/MOMENT-1-large"
LARGO = 512
PARCHE = 8
NIVELES_NATIVOS = (0.5,)


def _ocultos(horizonte: int) -> int:
    return math.ceil(horizonte / PARCHE) * PARCHE


def carga(dispositivo: str = "cuda"):
    from huggingface_hub import hf_hub_download
    from momentfm import MOMENTPipeline
    from safetensors.torch import load_file

    if dispositivo.startswith("cuda") and not torch.cuda.is_available():
        dispositivo = "cpu"
    modelo = MOMENTPipeline.from_pretrained(
        PESOS, model_kwargs={"task_name": "reconstruction"})
    referencia = load_file(hf_hub_download(PESOS, "model.safetensors"))
    propios = modelo.state_dict()
    distintos = [k for k in set(propios) & set(referencia)
                 if not torch.equal(propios[k].detach().cpu(), referencia[k].cpu())]
    if set(propios) != set(referencia) or distintos:
        raise RuntimeError("pesos de MOMENT mal cargados: el modelo tiene capas que no "
                           "salieron del archivo")
    modelo.requires_grad_(False)
    return modelo.eval().to(dispositivo)


def predice(modelo, historia: np.ndarray, horizonte: int,
            niveles: Sequence[float]) -> tuple:
    """Devuelve ``(cuantiles, None)``; sólo la fila del nivel 0.5 trae valores."""
    contexto_largo = LARGO - _ocultos(horizonte)
    valores = np.asarray(historia, dtype=float)[-contexto_largo:]
    valores = np.where(np.isfinite(valores), valores, np.nan)
    import pandas as pd
    valores = pd.Series(valores).ffill().bfill().to_numpy()
    ventana = np.zeros((1, 1, LARGO), dtype=np.float32)
    mascara = np.zeros((1, LARGO), dtype=np.float32)
    ventana[0, 0, LARGO - len(valores):] = valores
    mascara[0, LARGO - len(valores):] = 1.0
    dispositivo = next(modelo.parameters()).device
    with torch.no_grad():
        salida = modelo.short_forecast(x_enc=torch.from_numpy(ventana).to(dispositivo),
                                       input_mask=torch.from_numpy(mascara).to(dispositivo),
                                       forecast_horizon=int(horizonte))
    punto = salida.forecast[0, 0, :].float().cpu().numpy()
    cuantiles = np.full((len(niveles), int(horizonte)), np.nan)
    for fila, nivel in enumerate(niveles):
        if abs(float(nivel) - 0.5) < 1e-9:
            cuantiles[fila] = punto
    return cuantiles, None
