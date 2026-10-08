"""
Las figuras del artículo: cómo se ve el abanico de cada método en un día de pico.

Un cuadro de cobertura dice cuántas veces falló cada método. Una figura dice **cómo**
falló, y en un artículo de decisión eso importa: el lector tiene que poder ver que el
abanico del análogo llega arriba donde el del rival se queda corto.

Los días se eligen con una regla declarada, no a mano: del decil de cuentas más caras, los
que el método propuesto cubre y el comparativo más preciso no. Elegir el día que mejor se
ve sería dibujar la conclusión en vez de mostrarla, así que la regla y el número de días
que la cumplen se reportan en el pie de figura.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.protocol import RESULTS  # noqa: E402

FIGURAS = RESULTS.parent / "figuras"

## de dónde sale el abanico de cada método que entra a la figura
FUENTES = {
    "Analog-mezcla": ("picos_prueba", "Analog-mezcla"),
    "Analog": ("picos_prueba", "Analog"),
    "LEAR": ("lear_prueba", "LEAR"),
    "t0-beta": ("err_t0-beta", "t0-beta+errores"),
}
COLORES = {"Analog-mezcla": "#1b5e20", "Analog": "#7cb342",
           "LEAR": "#ef6c00", "t0-beta": "#5e35b1"}


def abanico(corrida: str, metodo: str, zona: str, origen) -> dict:
    """Los bordes de 90% y la mediana de un pronóstico, más lo observado."""
    t = pd.read_parquet(Path(RESULTS) / corrida / f"{zona}.parquet")
    p = t[(t.method == metodo) & (t.origin == origen)].sort_values("step")
    if p.empty:
        return None
    return {"bajo": p["q0.05"].to_numpy(), "alto": p["q0.95"].to_numpy(),
            "medio": p["q0.5"].to_numpy(), "real": p["observed"].to_numpy()}


def una_figura(dias: list, nombre: str, titulo: str, nota: str,
               perfil=None) -> Path:
    """
    Dos renglones: las horas arriba, la cuenta del día abajo.

    El renglón de abajo no es decoración. La afirmación del trabajo es sobre la CUENTA
    DEL DÍA, no sobre el pico horario, y una figura que sólo muestre las horas invita a
    leer otra cosa: en un día de pico puede ocurrir que ningún método cubra la hora más
    cara y que aun así unos cubran la cuenta y otros no. Mostrar sólo arriba sería
    ilustrar una afirmación distinta de la que se mide.
    """
    perfil = np.ones(24) if perfil is None else perfil
    n = len(dias)
    fig, ejes = plt.subplots(2, n, figsize=(4.3 * n, 6.4),
                             gridspec_kw={"height_ratios": [2.4, 1]})
    ejes = np.atleast_2d(ejes)
    if n == 1:
        ejes = ejes.reshape(2, 1)

    for col, (zona, origen) in enumerate(dias):
        arriba, abajo = ejes[0, col], ejes[1, col]
        horas = np.arange(1, 25)
        real, cuentas = None, {}

        for etiqueta, (corrida, metodo) in FUENTES.items():
            d = abanico(corrida, metodo, zona, origen)
            if d is None:
                continue
            real = d["real"]
            arriba.fill_between(horas, d["bajo"], d["alto"], alpha=0.10,
                                color=COLORES[etiqueta], linewidth=0)
            arriba.plot(horas, d["alto"], color=COLORES[etiqueta], lw=1.8,
                        label=etiqueta)
            cuentas[etiqueta] = (float(d["bajo"] @ perfil),
                                 float(d["medio"] @ perfil),
                                 float(d["alto"] @ perfil))
        if real is None:
            continue
        arriba.plot(horas, real, color="black", lw=2.6, label="precio observado")
        arriba.set_title(f"{zona.replace('_', ' ')} · {pd.Timestamp(origen).date()}",
                         fontsize=10)
        arriba.set_xlabel("hora del día siguiente")
        arriba.grid(alpha=0.25, lw=0.5)

        ## la cuenta del día: el intervalo de cada método y lo que de verdad costó
        cuenta_real = float(real @ perfil)
        for fila, (etiqueta, (lo, me, hi)) in enumerate(cuentas.items()):
            y = len(cuentas) - fila
            cubre = lo <= cuenta_real <= hi
            abajo.plot([lo, hi], [y, y], color=COLORES[etiqueta],
                       lw=4.5 if cubre else 2.0, alpha=0.9 if cubre else 0.55,
                       solid_capstyle="butt")
            abajo.plot([me], [y], "o", color=COLORES[etiqueta], ms=5)
            if not cubre:
                abajo.annotate("se queda corto", (hi, y), fontsize=7.5,
                               color="#b71c1c", xytext=(4, -3),
                               textcoords="offset points")
        abajo.axvline(cuenta_real, color="black", lw=2.2)
        abajo.set_yticks(range(1, len(cuentas) + 1))
        abajo.set_yticklabels(list(cuentas)[::-1], fontsize=8)
        abajo.set_xlabel("cuenta del día, pesos")
        abajo.grid(alpha=0.25, lw=0.5, axis="x")
        abajo.set_title(f"cuenta real: {cuenta_real:,.0f}", fontsize=9)

    ejes[0, 0].set_ylabel("precio, pesos por MWh")
    ejes[0, 0].legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.suptitle(titulo, fontsize=12.5, y=0.99)
    fig.text(0.5, -0.03, nota, ha="center", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIGURAS.mkdir(exist_ok=True)
    ruta = FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight", dpi=170)
    plt.close(fig)
    return ruta


def elige_dias(cuantos: int = 4) -> tuple:
    """
    Del decil de cuentas más caras, los que el propuesto cubre y el rival no.

    Devuelve los días y cuántos cumplen la regla, para que el pie de figura lo diga.
    """
    d = pd.read_parquet(RESULTS / "costo_diario_por_dia.parquet")
    cortes = d.costo_real.quantile(np.arange(0, 1.01, 0.1)).to_numpy()
    d["decil"] = np.clip(np.searchsorted(cortes[1:-1], d.costo_real) + 1, 1, 10)
    alto = d[d.decil == 10]
    piv = alto.pivot_table(index=["zona", "origen"], columns="metodo",
                           values="dentro_90")
    cumplen = piv[(piv["Analog-mezcla"] == 1) & (piv["t0-beta+errores"] == 0)]
    cuenta = alto.pivot_table(index=["zona", "origen"], values="costo_real",
                              aggfunc="first")
    orden = cumplen.join(cuenta).sort_values("costo_real", ascending=False)
    ## Un día distinto por panel, y además de meses distintos. Tomar el más caro sin
    ## más daba cuatro zonas del MISMO 23 de mayo de 2024: eso muestra un episodio
    ## cuatro veces y no cuatro casos, y un lector con razón lo leería como un solo
    ## dato presentado cuatro veces.
    vistos, dias = set(), []
    for (zona, origen), _ in orden.iterrows():
        mes = pd.Timestamp(origen).strftime("%Y-%m")
        if mes in vistos or zona in vistos:
            continue
        vistos.update({mes, zona})
        dias.append((zona, origen))
        if len(dias) == cuantos:
            break
    return dias, len(cumplen), len(piv)


def main() -> None:
    dias, cumplen, total = elige_dias(4)
    nota = (f"Días elegidos por regla, no a mano: del decil de cuentas diarias más caras, "
            f"los que Analog-mezcla cubre y t0-beta no. {cumplen:,d} de {total:,d} días "
            f"del decil cumplen la regla; se muestran cuatro, de zonas y meses distintos, el "
            f"más caro de cada uno. ARRIBA: bandas del intervalo central de 90% por hora, "
            f"con su borde superior en línea. ABAJO: el intervalo de 90% de la CUENTA DEL "
            f"DÍA de cada método, con la cuenta real en negro. Nótese que en un día de "
            f"pico puede fallar la hora más cara en todos y aun así unos cubrir la cuenta "
            f"y otros no: lo que este trabajo mide es la cuenta.")
    ruta = una_figura(dias, "figura_picos_cubiertos.png",
                      "Días caros que el análogo cubre y el comparativo más preciso no",
                      nota)
    print(f"escrito {ruta}")
    for zona, origen in dias:
        print(f"  {zona} {pd.Timestamp(origen).date()}")

    ## y el contraste: días normales, donde el rival va bien y el análogo sobra
    d = pd.read_parquet(RESULTS / "costo_diario_por_dia.parquet")
    cortes = d.costo_real.quantile(np.arange(0, 1.01, 0.1)).to_numpy()
    d["decil"] = np.clip(np.searchsorted(cortes[1:-1], d.costo_real) + 1, 1, 10)
    medio = d[d.decil.isin([4, 5])]
    piv = medio.pivot_table(index=["zona", "origen"], columns="metodo",
                            values="dentro_90")
    ambos = piv[(piv["Analog-mezcla"] == 1) & (piv["t0-beta+errores"] == 1)]
    vistos, dias2 = set(), []
    for (zona, origen) in ambos.index:
        mes = pd.Timestamp(origen).strftime("%Y-%m")
        if mes in vistos or zona in vistos:
            continue
        vistos.update({mes, zona})
        dias2.append((zona, origen))
        if len(dias2) == 4:
            break
    nota2 = (f"Días del medio de la distribución —deciles 4 y 5— donde los dos cubren. "
             f"{len(ambos):,d} de {len(piv):,d} días cumplen. Aquí el abanico del análogo "
             f"sobra, y es la razón por la que pierde en los deciles bajos.")
    ruta2 = una_figura(dias2, "figura_dias_normales.png",
                       "Días normales: los dos cubren, y el análogo paga filo de más",
                       nota2)
    print(f"escrito {ruta2}")


if __name__ == "__main__":
    main()
