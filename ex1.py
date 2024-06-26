#generated by gpt and d2l

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
