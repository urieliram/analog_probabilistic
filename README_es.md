# analog_probabilistic — en palabras llanas

Versión de divulgación del método y del resultado. Lo técnico, el protocolo del
experimento y cómo reproducirlo están en [README.md](README.md), en inglés.

El método busca en el pasado los **episodios** en que el precio se comportaba como
hoy, mira qué pasó después en cada uno, y usa esos desenlaces como los futuros
posibles de mañana. Son cuarenta futuros, no un **pronóstico determinista puntual**.
Juntos forman un abanico: por abajo lo barato que podría salir, por arriba lo caro.

Antes de usar cada desenlace pasado hay que ajustarlo al tamaño del presente, y ahí
se esconde una decisión. Puedes dejarle su separación tal cual, o puedes apretar los
desenlaces hacia su promedio. Apretarlos es lo que hace la regresión lineal, y es lo
que la versión anterior de este trabajo presentaba como su aportación.

**Apretar está mal.** Cierra el abanico, y un abanico cerrado promete menos riesgo
del que hay: el precio real se sale por arriba más seguido. Eso se cumplió los seis
años, las 25 zonas de carga, siempre en el mismo sentido, sin una sola excepción.
Entre siete y diez de cada cien veces de más.

Lo que sí daba apretar era un centro un poco mejor: el pronóstico determinista
puntual pegaba más cerca. Por eso parecía un intercambio — pierdes abanico, ganas
puntería. **Resulta que no.** Esa puntería aparece sólo en el año tranquilo donde se
afinó el método. En los cinco años siguientes se invierte, y se invierte más entre
más movido es el año. En 2024, el peor de ellos, apretar fue peor por los dos lados
a la vez.

No hay intercambio. Apretar simplemente es peor.

Esto importa sobre todo a quien usa el pronóstico para cubrirse, no para comparar
modelos. El error caro es quedarse corto en el pico, y un abanico apretado es
exactamente la máquina de quedarse corto en el pico.

Y el punto metodológico es el incómodo: **si afinas con un año y lo das por bueno,
eliges mal.** Nosotros lo hicimos, y los cinco años que no habíamos tocado nos
corrigieron.
