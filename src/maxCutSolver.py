import networkx as nx
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.translators import to_ising
from qiskit.circuit.library import RealAmplitudes
import matplotlib.pyplot as plt
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit.visualization import plot_distribution

graph = nx.Graph()

graph.add_weighted_edges_from([
    (0,1,1),
    (1,2,3),
    (2,3,2),
    (3,4,1),
    (4,0,3),
    (1,4,4)])

maxcut = Maxcut(graph)
quadraticProgram = maxcut.to_quadratic_program()
hamiltonian, offset = to_ising(quadraticProgram)
#print(hamiltonian)

ansatz = RealAmplitudes(num_qubits=4)
# fig = ansatz.decompose().draw("mpl")
# fig.savefig("pics/ansatz_circuit.png")


optimizer = COBYLA()
estimator = Estimator()
vqe = VQE(estimator,ansatz,optimizer)

result = vqe.compute_minimum_eigenvalue(hamiltonian)
optimalState = ansatz.assign_parameters(result.optimal_parameters)
optimalState.measure_all()

sampler = Sampler(options={"shots": 1024})
distribution = sampler.run([optimalState]).result().quasi_dists[0]



solution = max(distribution.binary_probabilities().items(), key=lambda x: x[1])
print(solution)
fig = plot_distribution(distribution.binary_probabilities())


fig.savefig("pics/distribution.png")