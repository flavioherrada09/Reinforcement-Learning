# Reinforcement-Learning

## ¿Qué es RL?

El aprendizaje reforzado (Reinforcement Learning, RL) es un paradigma de la inteligencia artificial en el que un agente aprende a tomar decisiones a través de la interacción directa con su entorno, recibiendo recompensas o penalizaciones según la calidad de sus acciones. En el ámbito de las finanzas y, específicamente, en el trading, esta metodología resulta especialmente útil debido a la naturaleza dinámica y estocástica de los mercados. Mediante la recopilación y el análisis de información procedente de precios históricos, indicadores técnicos y datos macroeconómicos, el agente de RL puede ajustar continuamente su estrategia con el objetivo de maximizar la ganancia o el rendimiento ajustado al riesgo. Este enfoque contrasta con los métodos tradicionales, que suelen basarse en modelos estáticos y no son tan flexibles frente a cambios súbitos en las condiciones de mercado.

## El Deep Q-Learning

Una de las técnicas más prominentes dentro del RL es el Deep Q-Learning (DQN), introducido por el equipo de DeepMind. El DQN fusiona el clásico algoritmo de Q-Learning con redes neuronales profundas, lo que permite lidiar con espacios de estado de gran dimensión o continuos, comunes en el trading financiero. Al emplear estas redes para aproximar la función de valor Q, el agente puede aprender a seleccionar acciones (comprar, vender o mantener) a partir de grandes volúmenes de datos de mercado, mejorando su capacidad de generalización.

## Aplicando el DQL para blackjack

'blackjack.py' entrena un agente de Deep Q-Learning (DQL) para jugar al blackjack usando Gymnasium y PyTorch. En primer lugar, se define una red neuronal (QNetwork) que actúa como aproximador de los valores Q para cada acción (hit o stand). Luego, dentro de la función train(), se crea el entorno de Blackjack y se configura la política epsilon-greedy (con parámetros epsilon, epsilon_min y epsilon_decay para controlar la exploración/explotación). Durante cada episodio, el agente observa el estado, elige una acción con la política epsilon-greedy, y guarda la experiencia (estado, acción, recompensa, siguiente estado) en una memoria de replay.

Una vez que la memoria acumula suficiente información, se toman mini-lotes (batch) aleatorios para actualizar los pesos de la red con la ecuación de Bellman, usando la función de pérdida MSE. De este modo, el modelo aprende a estimar los valores Q de cada acción y, con el tiempo, mejora su desempeño en el juego de Blackjack. Tras completarse los episodios de entrenamiento, se reduce progresivamente la tasa de exploración (epsilon) y se guarda el modelo entrenado en un archivo (blackjack_dql.pth).


## Resultados

El agente realiza su entrenamiento a lo largo de 5000 episodios. Cada 100 episodios, se registra en los logs la recompensa acumulada junto con el valor de epsilon.

![image](https://github.com/user-attachments/assets/9cef7df1-c798-4260-9c72-96c4b61c4b59)
