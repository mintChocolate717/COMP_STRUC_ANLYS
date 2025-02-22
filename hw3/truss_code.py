import numpy as np
import pandas as pd
from pprint import pprint
from typing import List, Dict, Tuple

def read_nodes_file(filepath: str = 'inputs/nodes') -> Tuple[np.ndarray, int, int]:
   # Read the first two lines manually
    with open(filepath, 'r') as file:
        num_dims = int(file.readline().strip())
        num_nodes = int(file.readline().strip())
    # Read the rest of the file (starting from the third line)
    node_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=2)
    # Convert to numpy array
    nodes = node_file.to_numpy()
    # Add an extra row to accommodate 1-based indexing
    padded_nodes = np.pad(nodes, ((1,0), (0,0)), 'constant', constant_values=np.nan)

    return padded_nodes, num_nodes, num_dims

def build_global_connectivity(node_mx: np.ndarray, dof_per_node) -> np.ndarray:
    global_connectivity = np.zeros_like(node_mx, dtype=int)
    # form gcon data structure
    for node_num in range(1, len(node_mx)): # iterate thru all nodes
        for local_dof in range(1, dof_per_node + 1): # iterate thru all dof components
            # assign global DOF to each element:
            global_connectivity[node_num, local_dof] = dof_per_node * (node_num - 1) + local_dof
    
    return global_connectivity

def read_elements_file(filepath: str = 'inputs/elements') -> Tuple[np.ndarray, int]:
   # Read the first linee manually
    with open(filepath, 'r') as file:
        num_elements = int(file.readline().strip())
    # Read the rest of the file (starting from the third line)
    element_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=1)
    # Convert to numpy array
    elements = element_file.to_numpy()
    # Add an extra row to accommodate 1-based indexing
    padded_elements = np.pad(elements, ((1,0),(0,0)), 'constant', constant_values=np.nan)

    return padded_elements, num_elements

def compute_lengths_and_cosines(elements: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # get each element's 1st nodes
    node_1 = elements[1:, 1].astype(int)
    # get each element's 2nd nodes
    node_2 = elements[1:, 2].astype(int)

    # get coordinates of 1st nodes
    node_1_coordinates = nodes[node_1, 1:]
    # get coordinates of 2nd nodes
    node_2_coordinates = nodes[node_2, 1:]
    
    # compute change in coordinates for each positions.
    coordinate_displacements = node_2_coordinates - node_1_coordinates

    # for each element, compute length:
    element_lengths = np.linalg.norm(coordinate_displacements, axis=1)

    # for each element, compute element cosines
    element_cosines = coordinate_displacements / element_lengths.reshape(len(elements)-1, 1)

    # again, add an extra row to both lengths and cosine matrix to accomodate 1-based indexing:
    element_lengths = np.pad(element_lengths, (1,0), 'constant', constant_values=np.nan)
    element_cosines = np.pad(element_cosines, ((1,0), (1,0)), 'constant', constant_values=np.nan)

    return element_lengths, element_cosines

def read_forces_file(filepath: str = 'inputs/forces') -> Tuple[np.ndarray, int]:
   # Read the first linee manually
    with open(filepath, 'r') as file:
        num_force_BCs = int(file.readline().strip())
    # Read the rest of the file (starting from the third line)
    force_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=1)
    # Convert to numpy array
    forces = force_file.to_numpy()

    return forces, num_force_BCs

def read_displacements_file(filepath: str = 'inputs/displacements') -> Tuple[np.ndarray, int]:
   # Read the first linee manually
    with open(filepath, 'r') as file:
        num_displacement_BCs = int(file.readline().strip())
    # Read the rest of the file (starting from the third line)
    displacement_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=1)
    # Convert to numpy array
    displacements = displacement_file.to_numpy()

    return displacements, num_displacement_BCs


def main():
    # Call the function to get node mx, num_nodes, and num_dims
    node, num_nodes, num_dims = read_nodes_file(filepath='test_cases_3d/nodes')
    # compute dof per node
    dof_per_node = num_dims # for Truss only
    # total number of DOFs in a structure
    total_num_dofs = (len(node) - 1) * dof_per_node
    print(f"Number of Nodes: {num_nodes}\nNumber of dims: {num_dims}\ndof per node:{dof_per_node}\ntotal dofs:{total_num_dofs}")
    print("\nnode matrix:")
    pprint(node)


    global_connectivity = build_global_connectivity(node, dof_per_node)
    print("\ngcon matrix:")
    pprint(global_connectivity)


    element, num_elements = read_elements_file('test_cases_3d/elements')
    print(f"\nnumber of elements: {num_elements}")
    print("elements matrix:")
    pprint(element)


    lengths, cosines = compute_lengths_and_cosines(element, node)
    print("\nelement lenghts:")
    pprint(lengths)
    print("element cosines")
    pprint(cosines)


    force_BCs, num_force_BCs = read_forces_file('test_cases_3d/forces')
    print(f"\nNum of Force BCs: {num_force_BCs}")
    print("Node #, DOF #, Value:")
    pprint(force_BCs)

    
    displacement_BCs, num_displacement_BCs = read_forces_file('test_cases_3d/displacements')
    print(f"\nNum of Displacement BCs: {num_displacement_BCs}")
    print("Node #, DOF #, Value:")
    pprint(displacement_BCs)


if __name__ == "__main__":
    main()