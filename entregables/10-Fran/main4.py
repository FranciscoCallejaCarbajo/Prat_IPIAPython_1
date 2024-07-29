import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

# Generar un grafo dirigido con pesos
def generate_directed_graph(num_nodes, weight_range=(1, 100)):
    G = nx.DiGraph()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                G.add_edge(i, j, weight=random.randint(*weight_range))
    return G

# Algoritmo de Floyd-Warshall con predecesores
def floyd_warshall(G):
    n = G.number_of_nodes()
    dist = np.full((n, n), float('inf'))
    np.fill_diagonal(dist, 0)
    pred = np.full((n, n), None)
    
    for u, v, data in G.edges(data=True):
        dist[u][v] = data['weight']
        pred[u][v] = u
    
    start_time = time.time()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    pred[i][j] = pred[k][j]
    end_time = time.time()
    tiempo_ejecucion = end_time - start_time
    
    return dist, pred, tiempo_ejecucion

# Reconstrucción del camino más corto
def reconstruct_path(pred, start, end):
    if pred[start][end] is None:
        return []
    path = [end]
    while end != start:
        end = pred[start][end]
        path.append(end)
    path.reverse()
    return path

# Visualización del grafo y el camino más corto
def plot_shortest_path(G, path):
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
    
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='green', node_size=500)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('Camino más corto')
    plt.show()

# Visualización del grafo original
def plot_graph(G, title=''):
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

# Función principal para realizar las pruebas y mediciones
def main():
    num_nodes_list = [5, 10, 15, 20, 25, 30, 35, 40]
    tiempos_fw = []
    teoricos_fw = []

    for num_nodes in num_nodes_list:
        G = generate_directed_graph(num_nodes)
        
        # Ejecutar Floyd-Warshall
        dist_matrix, pred_matrix, tiempo_fw = floyd_warshall(G)
        tiempos_fw.append(tiempo_fw)
        
        # Calcular el tiempo teórico Θ(n^3)
        teorico_fw = num_nodes ** 3
        teoricos_fw.append(teorico_fw)
        
        print(f'Número de nodos: {num_nodes}')
        print(f'Tiempo de ejecución de Floyd-Warshall: {tiempo_fw:.4f} segundos')
        print("Matriz de distancias:")
        print(dist_matrix)
        
        start_node = 0
        end_node = num_nodes - 1
        path = reconstruct_path(pred_matrix, start_node, end_node)
        
        print(f'Camino más corto de {start_node} a {end_node}: {path}')
        
        # Mostrar gráfico del grafo y el camino más corto
        plot_shortest_path(G, path)
        
        # Mostrar gráfico del grafo original
        plot_graph(G, title=f'Grafo dirigido con {num_nodes} nodos')
    
    # Mostrar resultados de comparación de tiempos de ejecución
    plt.figure()
    plt.plot(num_nodes_list, tiempos_fw, label='Floyd-Warshall Empírico', marker='o')
    plt.plot(num_nodes_list, [t / max(teoricos_fw) * max(tiempos_fw) for t in teoricos_fw], label='Θ(n^3) Teórico', linestyle='--', marker='x')
    plt.xlabel('Número de nodos')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Comparación de tiempos de ejecución de Floyd-Warshall')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
