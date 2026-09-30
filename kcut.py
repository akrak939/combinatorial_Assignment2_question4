import hashlib
import random
import matplotlib.pyplot as plt

def random_bit(a, b, p):
    randomseed = int(hashlib.sha256(f"{a},{b}".encode()).hexdigest(), 16) + seed
    rng = random.Random(randomseed)
    return 1 if rng.random() < p else 0

def edge(i, j):
    if i==j:
        return 0
    a, b = min(i, j), max(i, j)
    return random_bit(a, b, p)

seed = 15
p = 0.2
n = 1000
k = 2
alg = 0

neighbours = []
tot_edges = 0
for i in range(n):
    neighbours.append([])
    for u in range(i):
        if edge(u, i) == 1:
            neighbours[i].append(u)
            tot_edges = tot_edges + 1

def greedy_kcut(k):
    assigned_set = [0] * n
    z = 0
    for i in range(n):
        neighbours_in_set = [0] * k
        for u in neighbours[i]:
            neighbours_in_set[assigned_set[u]] = neighbours_in_set[assigned_set[u]] + 1

        chosen_set = 0
        max_edges_added = -1
        for s in range(k):
            edges_added = len(neighbours[i]) - neighbours_in_set[s]
            if edges_added > max_edges_added:
                max_edges_added = edges_added
                chosen_set = s

        assigned_set[i] = chosen_set
        z = z + max_edges_added
    return z

k_set = []
alg_set = []
while alg < tot_edges:
    alg = greedy_kcut(k)
    k_set.append(k)
    alg_set.append(alg)
    k = k + 1

plt.plot(k_set, alg_set, "o-")
plt.xlabel("k")
plt.ylabel("ALG_k")
plt.title("Greedy k-cut")
plt.savefig("kcut_plot.png")
plt.show()
