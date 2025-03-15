import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from mpl_toolkits.mplot3d import Axes3D

class QuantumState:
    def __init__(self, n):
        self.n = n
        self.state_vector = np.zeros((2**n,), dtype=complex)
        self.state_vector[0] = 1

    def apply_gate(self, gate):
        self.state_vector = np.dot(gate, self.state_vector)

    def measure(self):
        probabilities = np.abs(self.state_vector) ** 2
        return np.random.choice(range(2**self.n), p=probabilities)

    def get_state_vector(self):
        return self.state_vector

class QuantumCircuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []
        self.state = QuantumState(n_qubits)

    def add_gate(self, gate):
        self.gates.append(gate)

    def execute(self):
        for gate in self.gates:
            self.state.apply_gate(gate)

    def measure(self):
        return self.state.measure()

    def get_state_vector(self):
        return self.state.get_state_vector()

def create_hadamard_gate(n):
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    gate = np.eye(1)
    for _ in range(n):
        gate = np.kron(gate, H)
    return gate

def create_cnot_gate(control_qbit, target_qbit, n):
    CNOT = np.array([[1, 0, 0, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 0, 1], 
                     [0, 0, 1, 0]])
    gate = np.eye(1)
    for i in range(n):
        if i == control_qbit or i == target_qbit:
            gate = np.kron(gate, CNOT)
        else:
            gate = np.kron(gate, np.eye(2))
    return gate

def create_rotation_gate(theta, n):
    R = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                  [np.sin(theta/2), np.cos(theta/2)]])
    gate = np.eye(1)
    for _ in range(n):
        gate = np.kron(gate, R)
    return gate

def visualize_state_vector(state_vector):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(len(state_vector))
    y = np.abs(state_vector)
    z = np.angle(state_vector)
    
    ax.bar3d(x, np.zeros_like(x), np.zeros_like(x), 1, 1, y, shade=True)
    ax.set_xlabel('States')
    ax.set_ylabel('Probability Amplitude')
    ax.set_zlabel('Phase')
    
    plt.show()

def run_bernstein_vazirani(n):
    circuit = QuantumCircuit(n)
    circuit.add_gate(create_hadamard_gate(n))
    circuit.add_gate(create_cnot_gate(0, 1, n))  # Example: Add CNOT gates
    circuit.execute()
    result = circuit.measure()
    return result

def main():
    n = 3  # Number of qubits
    print("Number of qubits:", n)
    
    print("Running Bernstein-Vazirani algorithm...")
    bv_result = run_bernstein_vazirani(n)
    print(f"B-V Result: {bv_result}")
    
    state = QuantumState(n)
    hadamard = create_hadamard_gate(n)
    state.apply_gate(hadamard)
    
    print("State vector after applying Hadamard gate:")
    print(state.get_state_vector())
    
    visualize_state_vector(state.get_state_vector())

if __name__ == "__main__":
    main()