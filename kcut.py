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

# for every vertex v, store its neighbours that come before v
nb = []
m = 0   # total number of edges
for v in range(n):
    nb.append([])
    for u in range(v):
        if edge(u, v) == 1:
            nb[v].append(u)
            m = m + 1

def pick(v, k, grp):
    # sets of the neighbours of v that are already placed
    ng = []
    for u in nb[v]:
        ng.append(grp[u])

    # try every set and pick the one that cuts the most edges
    best = 0
    bgain = -1
    for s in range(k):
        gain = len(ng) - ng.count(s)
        if gain > bgain:
            bgain = gain
            best = s
    return best, bgain

def greedy(k):
    grp = [0] * n   # grp[v] = the set that vertex v is put in
    cut = 0         # number of edges in the cut
    for v in range(n):
        best, bgain = pick(v, k, grp)
        grp[v] = best
        cut = cut + bgain
    return cut

ks = []
algs = []
k = 2
while True:
    alg = greedy(k)
    ks.append(k)
    algs.append(alg)
    print("k =", k, " ALG_k =", alg)
    if alg == m:
        break
    k = k + 1

plt.plot(ks, algs)
plt.xlabel("k")
plt.ylabel("ALG_k")
plt.savefig("greedy_kcut_plot.png")
plt.show()
