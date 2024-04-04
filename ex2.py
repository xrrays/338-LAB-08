# Code partly generated using ChatGPT

# Exercise 2 Part 1)

# A simple array or list can be used, where the node with the smallest distance is found through a linear search, resulting in an (O(n^2)) time complexity for the algorithm.
# This method is straightforward but slow, suitable for small graphs.

# On the other hand, a binary heap allows for faster retrieval of the node with the smallest distance, with an extract-min operation time complexity of (O(\log n)), leading to an overall (O((n + m) \log n)) time complexity, where (m) is the number of edges.
# The binary heap is more complex but significantly faster, making it a better choice for larger graphs with many nodes and edges.

import heapq
import time
import matplotlib.pyplot as plt

class GraphNode:
    def __init__(self, data):
        self.data = data
        self.adjacent = {}  # key: GraphNode, value: weight

class Graph:
    def __init__(self):
        self.nodes = {}  # key: node data, value: GraphNode object

    def addNode(self, data):
        if data not in self.nodes:
            new_node = GraphNode(data)
            self.nodes[data] = new_node
            return new_node
        return self.nodes[data]

    def addEdge(self, n1, n2, weight=1):
        self.nodes[n1.data].adjacent[n2] = weight
        self.nodes[n2.data].adjacent[n1] = weight

    def slowSP(self, start):
        distances = {node: float('infinity') for node in self.nodes.values()}
        distances[start] = 0
        visited = set()

        while len(visited) != len(self.nodes):
            current_node, current_distance = None, float('infinity')
            for node, distance in distances.items():
                if node not in visited and distance < current_distance:
                    current_distance = distance
                    current_node = node

            visited.add(current_node)
            for neighbour, weight in current_node.adjacent.items():
                if neighbour not in visited:
                    new_distance = current_distance + weight
                    if new_distance < distances[neighbour]:
                        distances[neighbour] = new_distance

    def fastSP(self, start):
        distances = {node: float('infinity') for node in self.nodes.values()}
        distances[start] = 0
        priority_queue = [(0, start.data, start)]  # Include node data for comparison

        while priority_queue:
            current_distance, _, current_node = heapq.heappop(priority_queue)
            if current_distance > distances[current_node]:
                continue
            for neighbour, weight in current_node.adjacent.items():
                new_distance = current_distance + weight
                if new_distance < distances[neighbour]:
                    distances[neighbour] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbour.data, neighbour))

        return distances

    def importFromFile(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if '--' in line:
                parts = line.split('--')
                node1, rest = parts[0].strip(), parts[1].strip()
                node2, weight = rest.split('[weight=')
                node2 = node2.strip()
                weight = int(weight.rstrip('];\n'))
                self.addEdge(self.addNode(node1), self.addNode(node2), weight)

graph = Graph()
graph.importFromFile('random.dot')

# Measuring performance
slow_times = []
fast_times = []
for node in graph.nodes.values():
    start_time = time.time()
    graph.slowSP(node)
    slow_times.append(time.time() - start_time)

    start_time = time.time()
    graph.fastSP(node)
    fast_times.append(time.time() - start_time)

# Printing average, max, and min times
print(f"SlowSP - Average: {sum(slow_times) / len(slow_times)}, Max: {max(slow_times)}, Min: {min(slow_times)}")
print(f"FastSP - Average: {sum(fast_times) / len(fast_times)}, Max: {max(fast_times)}, Min: {min(fast_times)}")

# Plotting the histogram
plt.hist(slow_times, alpha=0.5, label='slowSP')
plt.hist(fast_times, alpha=0.5, label='fastSP')
plt.legend(loc='upper right')
plt.xlabel('Execution Time')
plt.ylabel('Frequency')
plt.title('Execution Time Distribution')
plt.show()



# Exercise 2 Part 4)

# The histogram shows us that the fast method is far more efficient at finding the shortest route in the graph. The execution times
# for the fast method are all clustered on the left side, indicating lower time. The use of a heap and priority queue is preferrable
# and leads to a lower complexity and better performance.
