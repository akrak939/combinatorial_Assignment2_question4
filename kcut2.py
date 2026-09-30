import hashlib
import random
import time
import matplotlib.pyplot as plt

def random_bit(a, b, p):
    randomseed = int(hashlib.sha256(f"{a},{b}".encode()).hexdigest(), 16) + seed
    rng = random.Random(randomseed)
    return 1 if rng.random() < p else 0

def edge(i, j):
    # undirected graph, no self-loops
    if i == j:
        return 0
    a = min(i, j)
    b = max(i, j)
    return random_bit(a, b, p)

# Use your group number
seed = 15
p = 0.2
n = 1000
k = 2 


# Build the graph once: neighbours[v] = list of earlier vertices u < v adjacent to v
neighbours = [[] for _ in range(n)]
total_edges = 0
for v in range(n):
    for u in range(v):
        if edge(u, v) == 1:
            neighbours[v].append(u)
            total_edges += 1
print(f"Graph has {total_edges} edges")

results = []

while True:
    # partition[v] = set number containing vertex v
    partition = [-1] * n
    cut_edges = 0

    # Greedy assignment of vertices
    for v in range(n):
        # count[s] = number of earlier neighbours of v in set s
        count = [0] * k
        for u in neighbours[v]:
            count[partition[u]] += 1
        # gain of set s = neighbours NOT in s = len(neighbours[v]) - count[s]
        best_set = 0
        best_gain = -1
        for s in range(k):
            gain = len(neighbours[v]) - count[s]
            if gain > best_gain:
                best_gain = gain
                best_set = s
        partition[v] = best_set
        cut_edges += best_gain      # edges from v to earlier vertices that are cut

    results.append((k, cut_edges))
    print(f"k = {k}, ALGk = {cut_edges}")

    # Stop if all edges are in the cut
    if cut_edges == total_edges:
        print(f"All edges are in the cut for k = {k}")
        break
    k += 1

# Plot k against ALGk
x = [r[0] for r in results]
y = [r[1] for r in results]
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker="o")
plt.axhline(total_edges, linestyle="--", color="grey", label="total number of edges")
plt.xlabel("k")
plt.ylabel("ALGk")
plt.title("Greedy Max k-Cut")
plt.legend()
plt.grid(True)
plt.savefig("kcut_results.png", dpi=200)
plt.show()