"""
VERIFIED REFERENCE for Layer 5 (CVaR-QAOA).  Owner: M3 adapts this into quantum_layer.py.
Tested with PennyLane 0.45.1 (default.qubit), scipy 1.17.  Demonstrates:
  1. QUBO  ->  Ising expansion (asserted numerically against the true cost on every bitstring)
  2. p-layer QAOA circuit with RZ / IsingZZ cost layers and RX mixer (no deprecated qml.Hamiltonian)
  3. CVaR objective over the best alpha-fraction of sampled energies, optimised with COBYLA
  4. Elite-bitstring selection + comparison against brute force
Bit convention: bitstring[i] <-> wire i <-> critical_nodes[i];  x_i = 1  =>  MAX_GREEN,  x_i = 0 => ADAPTIVE_SHORT.
Adapt to TriageResult (weights/couplings dicts keyed by node id). Cache results; seed everything.
"""
import itertools, numpy as np, pennylane as qml
from scipy.optimize import minimize
rng = np.random.default_rng(42)
n = 5
w = rng.uniform(0.3, 1.0, n)
edges = {(0,1):0.4,(1,2):0.3,(2,4):0.5,(3,4):0.2}
K = 2; lam = 0.6

# QUBO: x^T Q x form  -> a_i (linear), Qij (i<j)
a = -w + lam*(1-2*K)
Q = {(i,j): lam*2 + edges.get((i,j),0.0) for i in range(n) for j in range(i+1,n)}
def qubo_energy(x):
    return sum(a[i]*x[i] for i in range(n)) + sum(q*x[i]*x[j] for (i,j),q in Q.items()) + lam*K*K
def true_cost(x):
    return -sum(w[i]*x[i] for i in range(n)) + sum(j*x[a_]*x[b_] for (a_,b_),j in edges.items()) + lam*(sum(x)-K)**2
# Ising
h = np.array([-a[i]/2 - sum(Q[tuple(sorted((i,j)))] for j in range(n) if j!=i)/4 for i in range(n)])
J = {k: q/4 for k,q in Q.items()}
const = lam*K*K + sum(a)/2 + sum(Q.values())/4
def ising_energy(x):
    z = [1-2*xi for xi in x]
    return sum(h[i]*z[i] for i in range(n)) + sum(J[k]*z[k[0]]*z[k[1]] for k in J) + const
states = list(itertools.product([0,1], repeat=n))
assert all(abs(true_cost(x)-qubo_energy(x))<1e-9 for x in states), "QUBO expansion wrong"
assert all(abs(true_cost(x)-ising_energy(x))<1e-9 for x in states), "Ising mapping wrong"
E = np.array([ising_energy(x) for x in states])  # index order == bitstring order, wire0 = MSB
print("QUBO and Ising expansions match true cost on all", len(states), "states")

dev = qml.device("default.qubit", wires=n)
p = 2
@qml.qnode(dev)
def circ(theta):
    g, b = theta[:p], theta[p:]
    for i in range(n): qml.Hadamard(i)
    for l in range(p):
        for i in range(n): qml.RZ(2*g[l]*h[i], wires=i)
        for (i,j),jv in J.items(): qml.IsingZZ(2*g[l]*jv, wires=[i,j])
        for i in range(n): qml.RX(2*b[l], wires=i)
    return qml.probs(wires=range(n))

def cvar(theta, alpha=0.2, shots=1024, seed=0):
    pr = circ(theta); r = np.random.default_rng(seed)
    idx = r.choice(len(pr), size=shots, p=pr/pr.sum())
    e = np.sort(E[idx]); k = max(1, int(np.ceil(alpha*shots)))
    return e[:k].mean()

best=None
for s in range(3):
    r = np.random.default_rng(s)
    x0 = np.concatenate([r.uniform(0,1,p), r.uniform(0,1,p)])
    res = minimize(lambda t: cvar(t, seed=s), x0, method="COBYLA", options={"maxiter":80})
    if best is None or res.fun<best.fun: best=res
pr = circ(best.x); r = np.random.default_rng(123)
samp = r.choice(len(pr), size=2048, p=pr/pr.sum())
elite = samp[np.argmin(E[samp])]
opt = int(np.argmin(E))
ratio = (E.max()-E[elite])/(E.max()-E.min())
print("brute-force optimum:", format(opt,f'0{n}b'), round(E[opt],3))
print("QAOA elite        :", format(elite,f'0{n}b'), round(E[elite],3), "| approx ratio", round(ratio,3))
print("P(optimum) =", round(pr[opt],3), "| uniform baseline =", round(1/2**n,3))
