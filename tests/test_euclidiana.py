"""La métrica euclidiana busca por nivel y forma; Pearson sólo por forma."""

import numpy as np

from analog_probabilistic.analog_probabilistic import find_analogs


def _serie_con_episodio_caro():
    rng = np.random.default_rng(0)
    x = np.sin(np.arange(5000) * 2 * np.pi / 24) * 100 + 500 + rng.normal(0, 10, 5000)
    x[3000:3200] += 1000
    x[-48:] += 1000
    return x


def test_euclidiana_elige_el_episodio_del_mismo_nivel():
    a = find_analogs(_serie_con_episodio_caro(), 48, 24, 5, 0.5, metric="euclidiana")
    assert np.all(a.windows.mean(axis=1) > 1200)


def test_pearson_no_distingue_el_nivel():
    a = find_analogs(_serie_con_episodio_caro(), 48, 24, 5, 0.5)
    assert np.all(a.windows.mean(axis=1) < 800)


def test_euclidiana_respeta_la_frontera():
    x = _serie_con_episodio_caro()
    a = find_analogs(x, 48, 24, 40, 0.5, metric="euclidiana")
    assert a.positions.max() + 48 + 24 <= len(x) - 48


def test_la_mascara_de_admisibles_se_respeta():
    x = _serie_con_episodio_caro()
    n = len(x)
    last = n - 2 * 48 - 24 + 1
    admisibles = np.zeros(last, dtype=bool)
    admisibles[1000:1500] = True
    a = find_analogs(x, 48, 24, 10, 0.5, admisibles=admisibles)
    assert np.all((a.positions >= 1000) & (a.positions < 1500))
