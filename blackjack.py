import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Definir la red neuronal para la aproximación de los Q-values
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Función para elegir una acción basada en la política epsilon-greedy
def epsilon_greedy_policy(state, epsilon, model, action_size):
    if np.random.rand() < epsilon:  # Exploración
        return np.random.choice(action_size)
    else:  # Explotación
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item()

# Entrenamiento del agente con DQL
def train():
    # Crear el entorno de Blackjack
    env = gym.make('Blackjack-v1')
    state_size = 3  # Estado: total jugador, carta dealer, si tiene as usable
    action_size = 2  # Acciones: 0 (stand), 1 (hit)
    
    # Hiperparámetros mejorados
    epsilon = 1.0  
    epsilon_min = 0.01  
    epsilon_decay = 0.9995  # Explora más tiempo antes de explotar
    alpha = 0.0005  # Tasa de aprendizaje más baja para más estabilidad
    gamma = 0.99  # Factor de descuento
    batch_size = 64  # Mini-lotes más grandes
    replay_memory_size = 10_000  # Memoria más grande
    n_episodes = 5000  # Más episodios para mejor aprendizaje
    
    # Crear el modelo y el optimizador
    model = QNetwork(state_size, action_size)
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    
    # Memoria de replay
    memory = []
    
    # Bucle de entrenamiento
    for episode in range(n_episodes):
        state, _ = env.reset()  
        done = False
        total_reward = 0
        
        while not done:
            # Seleccionar acción con epsilon-greedy
            action = epsilon_greedy_policy(state, epsilon, model, action_size)
            
            # Ejecutar la acción en el entorno
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.append((state, action, reward, next_state, done))  

            # Entrenar el modelo cuando haya suficientes experiencias
            if len(memory) >= batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)

                # Obtener valores Q actuales
                q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze()

                # Obtener valores Q futuros
                q_values_next = model(next_states_tensor).max(1)[0].detach()

                # Aplicar la ecuación de Bellman corregida
                target = rewards_tensor + (gamma * q_values_next * (1 - dones_tensor))

                # Calcular la pérdida y actualizar el modelo
                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            total_reward += reward
        
        # Reducir epsilon progresivamente
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Mostrar progreso cada 100 episodios
        if episode % 100 == 0:
            print(f"Episode {episode}/{n_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'blackjack_dql.pth')
    print("Entrenamiento finalizado.")

# Ejecutar el entrenamiento
if __name__ == '__main__':
    train()
