# Reinforcement-Learning

## ¿Qué es RL?

El aprendizaje reforzado (Reinforcement Learning, RL) es un paradigma de la inteligencia artificial en el que un agente aprende a tomar decisiones a través de la interacción directa con su entorno, recibiendo recompensas o penalizaciones según la calidad de sus acciones. En el ámbito de las finanzas y, específicamente, en el trading, esta metodología resulta especialmente útil debido a la naturaleza dinámica y estocástica de los mercados. Mediante la recopilación y el análisis de información procedente de precios históricos, indicadores técnicos y datos macroeconómicos, el agente de RL puede ajustar continuamente su estrategia con el objetivo de maximizar la ganancia o el rendimiento ajustado al riesgo. Este enfoque contrasta con los métodos tradicionales, que suelen basarse en modelos estáticos y no son tan flexibles frente a cambios súbitos en las condiciones de mercado.

## El Deep Q-Learning

Una de las técnicas más prominentes dentro del RL es el Deep Q-Learning (DQN), introducido por el equipo de DeepMind. El DQN fusiona el clásico algoritmo de Q-Learning con redes neuronales profundas, lo que permite lidiar con espacios de estado de gran dimensión o continuos, comunes en el trading financiero. Al emplear estas redes para aproximar la función de valor Q, el agente puede aprender a seleccionar acciones (comprar, vender o mantener) a partir de grandes volúmenes de datos de mercado, mejorando su capacidad de generalización.

## Aplicando el DQL para blackjack

'blackjack.py' entrena un agente de Deep Q-Learning (DQN) en el entorno de Blackjack de Gymnasium. El agente utiliza una red neuronal densa para predecir los valores Q (una salida por cada posible acción: “hit” o “stand”) y toma decisiones según una política epsilon-greedy, la cual va reduciendo el factor 
𝜖
ϵ a lo largo del entrenamiento para pasar gradualmente de la exploración a la explotación.

El código gestiona un replay buffer (almacenamiento de transiciones) de donde se extraen muestras aleatorias para entrenar la red, rompiendo la correlación temporal de los datos. Para mejorar la estabilidad del aprendizaje, se emplea una red objetivo que se sincroniza periódicamente con la red principal. Al final, se guarda la red entrenada en un archivo y se ofrece una función para jugar un episodio usando la política aprendida.


## Resultados

El agente realiza su entrenamiento a lo largo de 2500 episodios. Cada 100 episodios, se registra en los logs la recompensa acumulada junto con el valor de epsilon.

![image](https://github.com/user-attachments/assets/a2240a6a-547e-4bfb-b9fc-b80e64bf4b87)

