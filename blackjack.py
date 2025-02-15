import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random

class DQAgentBlackjack:
    def __init__(
        self,
        env_name: str = "Blackjack-v1",
        max_episodes: int = 5000,
        max_steps_per_episode: int = 100,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay_steps: int = 4000,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        buffer_size: int = 10_000
    ):
        """
        Inicializa el agente DQN para el juego de Blackjack.
        - env_name: nombre del entorno de Gymnasium (por defecto, Blackjack-v1).
        - max_episodes: número máximo de episodios de entrenamiento.
        - max_steps_per_episode: pasos máximos por episodio (aunque Blackjack suele terminar antes).
        - gamma: factor de descuento para el cálculo del valor futuro.
        - epsilon, epsilon_min, epsilon_decay_steps: parámetros para la política epsilon-greedy.
        - batch_size: tamaño del minibatch para el entrenamiento.
        - learning_rate: tasa de aprendizaje para el optimizador.
        - buffer_size: capacidad del buffer de replay.
        """
        # Crear entorno
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n  # 2 acciones: "hit" o "stand"

        # Hiperparámetros de entrenamiento
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Buffer de replay (almacenamiento de experiencias)
        self.replay_buffer = deque(maxlen=buffer_size)

        # Construir las redes (Q principal y Q objetivo)
        self.q_network = self._build_network()
        self.q_target_network = self._build_network()
        self._update_target_network()

        # Optimización
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_function = tf.keras.losses.Huber()

        # Variables para decaimiento de epsilon
        self.epsilon_decay_rate = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps
        self.total_steps = 0  # Para controlar el decaimiento de epsilon

    def _build_network(self):
        """
        Construye la red neuronal profunda para aproximar la función Q.
        Dado que el estado de Blackjack es un vector de 3 valores:
            - Suma de cartas del jugador
            - Carta visible del dealer (0-10)
            - Indicador de as usable (0 o 1)
        se usará un MLP sencillo.
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(3,)),  # State = (player_sum, dealer_card, usable_ace)
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        return model

    def _update_target_network(self):
        """ Copia los parámetros de la red principal a la red objetivo. """
        self.q_target_network.set_weights(self.q_network.get_weights())

    def _sample_from_buffer(self):
        """ Obtiene un minibatch aleatorio del buffer de replay. """
        mini_batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def _epsilon_greedy_policy(self, state):
        """
        Selecciona una acción siguiendo la política epsilon-greedy.
        Si un número aleatorio es menor que epsilon, se elige acción aleatoria
        (exploración). De lo contrario, se elige la acción con mayor valor Q.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.numpy()[0])

    def train(self):
        """
        Entrena la red DQN jugando episodios completos de Blackjack.
        Se realiza la recolección de transiciones en el buffer y,
        cuando hay datos suficientes, se entrena la red principal.
        """
        for episode in range(self.max_episodes):
            state, _ = self.env.reset()  # El state suele ser (jugador, carta dealer, usable_ace)
            done = False
            total_reward = 0

            for step in range(self.max_steps_per_episode):
                self.total_steps += 1

                # Escoger acción mediante epsilon-greedy
                action = self._epsilon_greedy_policy(state)

                # Ejecutar acción en el entorno
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Almacenar la transición en el buffer
                self.replay_buffer.append((state, action, reward, next_state, done))

                # Actualizar estado
                state = next_state

                # Entrenar la red principal con un mini-lote, si el buffer es suficientemente grande
                if len(self.replay_buffer) >= self.batch_size:
                    self._train_step()

                # Actualizar la red objetivo periódicamente (opcional, aquí se hace cada 1000 pasos)
                if self.total_steps % 1000 == 0:
                    self._update_target_network()

                # Decaimiento de epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon -= self.epsilon_decay_rate
                    self.epsilon = max(self.epsilon, self.epsilon_min)

                if done:
                    break

            # Mostrar métricas cada cierto número de episodios
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{self.max_episodes} | "
                      f"Total Reward: {total_reward:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")

        print("Entrenamiento finalizado.")

        # Guardar modelo al terminar (opcional)
        self.q_network.save("blackjack_dqn_model.h5")
        print("Modelo guardado en 'blackjack_dqn_model.h5'")

    def _train_step(self):
        """ Un paso de entrenamiento (update) de la red Q principal. """
        states, actions, rewards, next_states, dones = self._sample_from_buffer()

        # Calcular Q-values actuales
        states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
        with tf.GradientTape() as tape:
            q_values = self.q_network(states_tf)
            # Seleccionar solo los Q-values de las acciones ejecutadas
            action_mask = tf.one_hot(actions, self.action_size)
            pred_q = tf.reduce_sum(q_values * action_mask, axis=1)

            # Calcular Q-values futuros (usando la red objetivo para estabilidad)
            next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
            q_next = self.q_target_network(next_states_tf)
            max_q_next = tf.reduce_max(q_next, axis=1)

            # Ecuación de Bellman
            target_q = rewards + (1 - dones) * self.gamma * max_q_next

            # Pérdida (MSE o Huber)
            loss = self.loss_function(target_q, pred_q)

        # Retropropagación
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    def play_one_game(self):
        """
        Permite jugar un episodio completo con la política actual (sin entrenamiento),
        imprimiendo las decisiones y la recompensa final.
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(self.q_network(tf.convert_to_tensor([state], dtype=tf.float32)).numpy()[0])
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
        print(f"Recompensa total en este episodio de prueba: {total_reward}")


if __name__ == "__main__":
    agent = DQAgentBlackjack(
        env_name="Blackjack-v1",
        max_episodes=2500,
        max_steps_per_episode=100,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay_steps=4000,
        batch_size=32,
        learning_rate=0.001,
        buffer_size=10000
    )
    agent.train()
    agent.play_one_game()
