# Code generated using chatgpt

# Question 2 Part 1)

# A simple array or list can be used, where the node with the smallest distance is found through a linear search, resulting in an (O(n^2)) time complexity for the algorithm.
# This method is straightforward but slow, suitable for small graphs.

# On the other hand, a binary heap allows for faster retrieval of the node with the smallest distance, with an extract-min operation time complexity of (O(\log n)), leading to an overall (O((n + m) \log n)) time complexity, where (m) is the number of edges.
# The binary heap is more complex but significantly faster, making it a better choice for larger graphs with many nodes and edges.

class GraphNode:
    def __init__(self, data):
        self.data = data
        self.adjacent = {}  # key: GraphNode, value: weight

class Graph:
    def __init__(self):
        self.nodes = {}  # key: node data, value: GraphNode object
    
    def addNode(self, data):                        # exercise 1/2, implement basic methods
        if data not in self.nodes:
            new_node = GraphNode(data)
            self.nodes[data] = new_node
            return new_node
        return self.nodes[data]
    
    def removeNode(self, node):
        if node.data in self.nodes:
            del self.nodes[node.data]
            for n in self.nodes.values():
                if node in n.adjacent:
                    del n.adjacent[node]
    
    def addEdge(self, n1, n2, weight=1):
        if n1.data in self.nodes and n2.data in self.nodes:
            self.nodes[n1.data].adjacent[n2] = weight
            self.nodes[n2.data].adjacent[n1] = weight
    
    def removeEdge(self, n1, n2):
        if n1.data in self.nodes and n2.data in self.nodes:
            if n2 in self.nodes[n1.data].adjacent:
                del self.nodes[n1.data].adjacent[n2]
            if n1 in self.nodes[n2.data].adjacent:
                del self.nodes[n2.data].adjacent[n1]
    
    def importFromFile(self, file):                 # exercise 1/3, implement import function
        try:                                        # with format specifications
            with open(file, 'r') as f:
                content = f.read().strip()

            if not content.startswith('strict graph'):
                return None

            # Extracting the content inside the curly braces
            content = content[content.find('{') + 1:content.rfind('}')].strip()
            lines = content.split('\n')

            self.nodes.clear()  # Clear existing nodes and edges

            for line in lines:
                parts = line.strip().split('--')
                if len(parts) != 2:
                    return None  # Invalid edge format

                node1, rest = parts[0].strip(), parts[1].strip()
                node2, *weight_part = rest.replace('[', ' ').replace(']', ' ').split()
                weight = 1  # default weight

                if weight_part:
                    try:
                        weight = int(weight_part[1])
                    except ValueError:
                        return None  # Invalid weight value

                n1, n2 = self.addNode(node1), self.addNode(node2)
                if n2 in self.nodes[n1.data].adjacent:
                    return None  # Duplicate edge detected
                self.addEdge(n1, n2, weight)

        except Exception:
            return None

        return self


class ListQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if self.is_empty():
            return None
        return self.items.pop(0)

    def peek(self):
        if self.is_empty():
            return None
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

class BinaryHeapQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)
        self._sift_up(len(self.items) - 1)

    def dequeue(self):
        if self.is_empty():
            return None
        item = self.items[0]
        self.items[0] = self.items[-1]
        self.items.pop()
        self._sift_down(0)
        return item

    def peek(self):
        if self.is_empty():
            return None
        return self.items[0]

    def is_empty(self):
        return len(self.items) == 0

    def _sift_up(self, index):
        parent = (index - 1) // 2
        while index > 0 and self.items[index] < self.items[parent]:
            self.items[index], self.items[parent] = self.items[parent], self.items[index]
            index = parent
            parent = (index - 1) // 2

    def _sift_down(self, index):
        left = 2 * index + 1
        right = 2 * index + 2
        smallest = index
        if left < len(self.items) and self.items[left] < self.items[smallest]:
            smallest = left
        if right < len(self.items) and self.items[right] < self.items[smallest]:
            smallest = right
        if smallest != index:
            self.items[index], self.items[smallest] = self.items[smallest], self.items[index]
            self._sift_down(smallest)


import timeit

def dijkstra_array(graph, source):
  """
  Dijkstra's shortest path algorithm using an array as the priority queue.
  """
  n = len(graph.nodes)
  visited = [False] * n
  distances = [float('inf')] * n
  distances[source] = 0

  for _ in range(n):
    min_index = -1
    for i in range(n):
      if not visited[i] and (min_index == -1 or distances[i] < distances[min_index]):
        min_index = i
    visited[min_index] = True

    for neighbor, weight in graph.nodes[graph.nodes[min_index].data].adjacent.items():
      if not visited[neighbor.data] and distances[min_index] + weight < distances[neighbor.data]:
        distances[neighbor.data] = distances[min_index] + weight

  return distances


def dijkstra_heap(graph, source):
  """
  Dijkstra's shortest path algorithm using a binary heap as the priority queue.
  """
  n = len(graph.nodes)
  visited = [False] * n
  distances = [float('inf')] * n
  distances[source] = 0

  # Use BinaryHeapQueue from your previous code
  pq = BinaryHeapQueue()
  pq.enqueue((0, source))

  while not pq.is_empty():
    dist, node_data = pq.dequeue()
    node = graph.nodes[node_data]

    if visited[node.data]:
      continue

    visited[node.data] = True

    for neighbor, weight in node.adjacent.items():
      if not visited[neighbor.data] and distances[node.data] + weight < distances[neighbor.data]:
        distances[neighbor.data] = distances[node.data] + weight
        pq.enqueue((distances[neighbor.data], neighbor.data))

  return distances


def main():
  # Read graph from file (replace with your function)
  sample_graph = Graph()
  sample_graph.importFromFile(file="random.dot")

  # Choose a source node (any node index)
  source = 0

  # Run Dijkstra's algorithm with array and measure time
  number = 100  # Number of times to run for accurate timing
  time_array = timeit.timeit(lambda: dijkstra_array(sample_graph, source), number=number)
  average_time_array = time_array / number

  # Run Dijkstra's algorithm with binary heap and measure time
  time_heap = timeit.timeit(lambda: dijkstra_heap(sample_graph, source), number=number)
  average_time_heap = time_heap / number

  # Find min and max execution times across all nodes (assuming dijkstra implementations return execution times)
  all_times_array, all_times_heap = [], []
  for node in sample_graph.nodes.values():
    all_times_array.append(dijkstra_array(sample_graph, node.data)[source])
    all_times_heap.append(dijkstra_heap(sample_graph, node.data)[source])

  min_time_array = min(all_times_array)
  max_time_array = max(all_times_array)
  min_time_heap = min(all_times_heap)
  max_time_heap = max(all_times_heap)

  # Print results
  print("Dijkstra with Array:")
  print(f"  Average Time: {average_time_array:.6f} seconds")
  print(f"  Min Time: {min_time_array:.6f} seconds")
  print(f"  Max Time: {max_time_array:.6f} seconds")

  print("\nDijkstra with Binary Heap:")
  print(f"  Average Time: {average_time_heap:.6f} seconds")
  print(f"  Min Time: {min_time_heap:.6f} seconds")
  print(f"  Max Time: {max_time_heap:.6f} seconds")
  
  import matplotlib.pyplot as plt

    # Create the histogram plots
  plt.figure(figsize=(10, 6))

  plt.subplot(121)  # Subplot 1 for array execution times
  plt.hist(all_times_array, bins=20, edgecolor='black', alpha=0.7, label='Array Dijkstra')
  plt.xlabel('Execution Time (seconds)')
  plt.ylabel('Frequency')
  plt.title('Histogram of Dijkstra with Array Execution Times')
  plt.grid(True)

  plt.subplot(122)  # Subplot 2 for heap execution times
  plt.hist(all_times_heap, bins=20, edgecolor='black', alpha=0.7, label='Binary Heap Dijkstra')
  plt.xlabel('Execution Time (seconds)')
  plt.ylabel('Frequency')
  plt.title('Histogram of Dijkstra with Binary Heap Execution Times')
  plt.grid(True)

    # Add legend if labels are used
  plt.legend()

    # Adjust layout and display the plot
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
    main()