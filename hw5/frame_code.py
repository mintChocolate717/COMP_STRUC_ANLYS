import numpy as np
# Set the line width to use more of the screen width
np.set_printoptions(linewidth=300)
import pandas as pd
from typing import Tuple
from tabulate import tabulate


def read_nodes_file(filepath: str) -> Tuple[np.ndarray, int, int]:
    """
    Reads a node file and returns a padded numpy array of node coordinates along with the number of nodes and dimensions.

    The node file is expected to have the following format:
      - The first line contains an integer representing the number of dimensions (num_dims).
      - The second line contains an integer representing the number of nodes (num_nodes).
      - Each subsequent line contains whitespace-separated numeric values representing a node's coordinates.
    - The number of values per line should match the number of dimensions.
    
    Parameters:
      filepath (str): The path to the node file.

    Returns:
      Tuple[np.ndarray, int, int]: A tuple containing:
        - padded_nodes (np.ndarray): A numpy array of node coordinates with an extra padded row at the beginning.
        - num_nodes (int): The number of nodes as read from the file.
        - num_dims (int): The number of dimensions as read from the file.

    Raises:
      FileNotFoundError: If the file specified by `filepath` does not exist.
      ValueError: If the first two lines cannot be converted to integers.
    """
   # Read the first two lines manually
    try:
        with open(filepath, 'r') as file:
            num_dims = int(file.readline().strip())
            num_nodes = int(file.readline().strip())
    except FileNotFoundError:
        exit(f"ERROR! File not found at {filepath}")
    except ValueError:
        exit(f"ERROR! First two lines of file at {filepath} should be integers. For frames, you have to specify it's 2D in the file.")    
    
    # Read the rest of the file (starting from the third line)
    node_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=2)
    # Convert to numpy array
    nodes = node_file.to_numpy()
    # Add an extra row to accommodate 1-based indexing
    padded_nodes = np.pad(nodes, ((1,0), (0,0)), 'constant', constant_values=np.nan)

    return padded_nodes, num_nodes, num_dims

def build_global_connectivity(num_nodes: int, dof_per_node: int) -> np.ndarray:
    """
    Builds a global connectivity matrix that assigns a unique global degree of freedom (DOF)
    to each local DOF of every node.

    This function takes a node matrix and the number of DOFs per node, then calculates the global
    DOF for each node's local DOF using 1-based indexing. The resulting matrix has the same shape
    as the input node matrix, with each entry filled with the computed global DOF number or NaN where
    not applicable.

    The global DOF is computed as:
        global_DOF = dof_per_node * (node_num - 1) + local_dof
    where:
      - node_num: the node index (starting from 1, assuming the first row is a padding row)
      - local_dof: the local degree of freedom (ranging from 1 to dof_per_node)

    Parameters:
      node_mx (np.ndarray): A numpy array representing nodes. Its shape is used to define the 
                            global connectivity structure.
      dof_per_node (int): The number of degrees of freedom per node.

    Returns:
      np.ndarray: A numpy array of the same shape as `node_mx`, where each valid entry contains the
                  corresponding global DOF number. Cells with no assignment are filled with NaN.

    Notes:
      - The function assumes that `node_mx` uses 1-based indexing for nodes (i.e., index 0 is a dummy
        or padded row). Adjust if using 0-based indexing.
    """
    # Create an array with the same shape as node_mx, filled with NaN,
    # which will hold the global connectivity information.
    global_connectivity = np.full((num_nodes + 1, dof_per_node + 1), np.nan, dtype=float)
    # Iterate through each node, starting at 1 to skip the padded row for 1-based indexing.
    for node_num in range(1, num_nodes+1): # iterate thru all nodes
        for local_dof in range(1, dof_per_node + 1): # iterate thru all dof components (x, y, ...)
            # Calculate and assign the global DOF for the current NODE and LOCAL DOF.
            global_connectivity[node_num, local_dof] = dof_per_node * (node_num - 1) + local_dof
    
    return global_connectivity

def read_elements_file(filepath: str = 'inputs/elements') -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Reads an elements file and returns the padded elements array, a subset of element nodes, and the total number of elements.

    The expected file format is:
      - The first line contains an integer representing the number of elements.
      - The following lines contain element data with at least three columns. The first three columns are assumed
        to represent the element nodes (e.g., connectivity information).
        - For FRAMES, columns are: [node1, node2, E, A, I]
      
    Parameters:
      filepath (str): Path to the elements file. Defaults to 'inputs/elements'.

    Returns:
      Tuple[np.ndarray, np.ndarray, int]:
        - padded_elements (np.ndarray): The complete elements array with a padded row for 1-based indexing.
        - element_nodes (np.ndarray): A subarray containing the first three columns of the padded elements array.
        - num_elements (int): The number of elements, as specified in the first line of the file.
    """
    try:
        # Read the first line manually to get the number of elements
        with open(filepath, 'r') as file:
            num_elements = int(file.readline().strip())
    except FileNotFoundError:
        exit(f"ERROR! File not found at {filepath}")
    except ValueError:
        exit(f"ERROR! First line of file at {filepath} should be an integer.")
    # Read the rest of the file (starting from the second line)
    element_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=1)
    # Convert the DataFrame to a numpy array
    elements = element_file.to_numpy()
    # Pad the array with an extra row at the top (filled with NaN) to accommodate 1-based indexing
    padded_elements = np.pad(elements, ((1, 0), (0, 0)), 'constant', constant_values=np.nan)
    # Extract the first three columns as element_nodes (assumes that elements have at least 3 columns)
    element_nodes = padded_elements[:, :3]

    return padded_elements, element_nodes, num_elements

def compute_lengths_and_cosines(elements: np.ndarray, nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the lengths and direction cosines for each element based on node connectivity and coordinates.

    Each element is defined by two node indices stored in columns 1 and 2 of the 'elements' array (ignoring the padded row).
    The corresponding coordinates for these nodes are retrieved from the 'nodes' array, where the first column is also assumed to be a dummy 
    index and actual coordinate data starts from the second column onward.

    The steps performed are:
      1. Extract the first and second node indices for each element from the 'elements' array (skipping the padded row).
      2. Retrieve the coordinates for these nodes from the 'nodes' array (ignoring the dummy index column).
      3. Calculate the displacement (difference) between the coordinates of the two nodes for each element.
      4. Compute the Euclidean norm (length) of each element based on the displacement.
      5. Calculate the direction cosines for each element by dividing the displacement vector by the element length.
      6. Pad the resulting lengths and cosines arrays with an extra row at the top to maintain 1-based indexing.

    Parameters:
      elements (np.ndarray): A 2D numpy array of element connectivity data, where:
                             - Row 0 is a padded dummy row.
                             - Columns 1 and 2 (starting from row 1) contain the first and second node indices respectively.
      nodes (np.ndarray): A 2D numpy array of node coordinates, where:
                          - Row 0 is a padded dummy row.
                          - Columns starting from index 1 contain the coordinate values for each node.

    Returns:
      Tuple[np.ndarray, np.ndarray]:
        - element_lengths (np.ndarray): A 1D numpy array containing the length of each element, padded with a dummy 
          first entry (NaN) for 1-based indexing.
        - element_cosines (np.ndarray): A 2D numpy array containing the direction cosines for each element, padded 
          with an extra first row and column (NaN) for 1-based indexing.
    """
    # Get each element's 1st node indices (skipping the padded row)
    node_1 = elements[1:, 1].astype(int)
    # Get each element's 2nd node indices (skipping the padded row)
    node_2 = elements[1:, 2].astype(int)

    # Retrieve 2D or 3D coordinates for the first nodes (ignoring the dummy first column in nodes)
    node_1_coordinates = nodes[node_1, 1:] # yes this is fancy indexing from NumPy
    # Retrieve 2D or 3D coordinates for the second nodes (ignoring the dummy first column in nodes)
    node_2_coordinates = nodes[node_2, 1:] # again, fancy indexing
    
    # Compute the displacement vector (difference in coordinates) for each element for each dimension (x, y, ..)
    coordinate_displacements = node_2_coordinates - node_1_coordinates

    # Compute the Euclidean norm (length) of each element based on the displacement vector
    element_lengths = np.linalg.norm(coordinate_displacements, axis=1)

    try:
        # Compute the direction cosines for each element
        element_cosines = coordinate_displacements / element_lengths.reshape(len(elements) - 1, 1)
    except ZeroDivisionError:
        exit("ERROR! Division by zero encountered while computing direction cosines.")

    # Pad the lengths and cosines arrays with an extra row to maintain 1-based indexing
    element_lengths = np.pad(element_lengths, (1, 0), 'constant', constant_values=np.nan)
    element_cosines = np.pad(element_cosines, ((1, 0), (1, 0)), 'constant', constant_values=np.nan)

    return element_lengths, element_cosines

def read_forces_file(filepath: str = 'inputs/forces') -> Tuple[np.ndarray, int]:
    """
    Reads a forces file and returns a padded forces array along with the number of force boundary conditions.

    The expected file format is:
      - The first line contains an integer specifying the number of force boundary conditions.
      - The following lines contain force data, where each row corresponds to a force vector or boundary condition.

    Parameters:
      filepath (str): The path to the forces file. Defaults to 'inputs/forces'.

    Returns:
      Tuple[np.ndarray, int]:
        - padded_forces (np.ndarray): A numpy array containing the force data, padded with an extra row for 1-based indexing.
        - num_force_BCs (int): The number of force boundary conditions read from the first line of the file.
    """
    try:
        # Read the first line to get the number of force boundary conditions
        with open(filepath, 'r') as file:
            num_force_BCs = int(file.readline().strip())
    except FileNotFoundError:
        exit(f"ERROR! File not found at {filepath}")
    except ValueError:
        exit(f"ERROR! First line of file at {filepath} should be an integer.")
    # Read the rest of the file (starting from the second line) using pandas.read_csv with whitespace as the delimiter
    force_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=1)
    # Convert the DataFrame to a numpy array
    forces = force_file.to_numpy()
    # Pad the forces array with an extra row at the beginning (filled with NaN) for 1-based indexing
    padded_forces = np.pad(forces, ((1, 0), (0, 0)), 'constant', constant_values=np.nan)

    return padded_forces, num_force_BCs

def read_displacements_file(filepath: str = 'inputs/displacements') -> Tuple[np.ndarray, int]:
    """
    Reads a displacements file and returns a padded numpy array of displacement data along with the number of displacement boundary conditions.

    The expected file format is:
      - The first line contains an integer indicating the number of displacement boundary conditions.
      - The subsequent lines contain displacement data, where each row corresponds to a displacement entry.

    The function performs the following steps:
      1. Opens the file and reads the first line to extract the number of displacement boundary conditions.
      2. Uses pandas.read_csv to read the rest of the file (skipping the first line) with whitespace as the delimiter.
      3. Converts the read data into a numpy array.
      4. Pads the array by adding an extra row at the beginning (filled with NaN) to support 1-based indexing.

    Parameters:
      filepath (str): The path to the displacements file. Defaults to 'inputs/displacements'.

    Returns:
      Tuple[np.ndarray, int]:
        - padded_displacements (np.ndarray): A numpy array containing the displacement data, padded with an extra row for 1-based indexing.
        - num_displacement_BCs (int): The number of displacement boundary conditions as read from the first line of the file.
    """
    try:
        # Read the first line to get the number of displacement boundary conditions
        with open(filepath, 'r') as file:
            num_displacement_BCs = int(file.readline().strip())
    except FileNotFoundError:
        exit(f"ERROR! File not found at {filepath}")
    except ValueError:
        exit(f"ERROR! First line of file at {filepath} should be an integer.") 
    
    # Read the rest of the file (starting from the second line) using pandas.read_csv with whitespace as the delimiter
    displacement_file = pd.read_csv(filepath, sep=r'\s+', header=None, skiprows=1)
    # Convert the DataFrame to a numpy array
    displacements = displacement_file.to_numpy()
    # Pad the displacements array with an extra row at the beginning (filled with NaN) for 1-based indexing
    padded_displacements = np.pad(displacements, ((1, 0), (0, 0)), 'constant', constant_values=np.nan)

    return padded_displacements, num_displacement_BCs

def reorder_global_connectivity(disp_BCs: np.ndarray, gcon: np.ndarray, num_total_dofs: int) -> np.ndarray:
    """
    Reorders the global connectivity matrix by moving degrees of freedom (DOFs) with prescribed displacement
    boundary conditions to the end of the numbering. This ensures that free (unknown) DOFs are numbered first,
    which is often required for numerical solution procedures.

    The function assumes that:
      - The displacement boundary conditions (disp_BCs) array is padded for 1-based indexing.
      - Each row in disp_BCs (except the first padded row) is in the format: [Node Number, Local DOF, Displacement Value],
        where Node Number and Local DOF are used for reordering.
      - The global connectivity matrix (gcon) is also padded for 1-based indexing, and it maps each node and its local DOFs
        to a unique global DOF number.
      - The total number of DOFs is provided in num_total_dofs.

    The reordering is performed by:
      1. Iterating over each displacement boundary condition (starting from the second row to skip the padding).
      2. Extracting the corresponding node number and local DOF.
      3. Retrieving the current global DOF for the given node and local DOF.
      4. Decreasing the global DOF numbers of all entries greater than the current prescribed DOF by 1.
         This effectively "shifts" free DOFs upward.
      5. Assigning the highest global DOF number (num_total_dofs) to the current prescribed DOF position, thereby moving it to the end.

    Parameters:
      disp_BCs (np.ndarray): A 2D array of displacement boundary conditions, padded for 1-based indexing.
                             Each row (after the first dummy row) should contain [Node Number, Local DOF, Displacement Value].
      gcon (np.ndarray): A 2D global connectivity matrix, padded for 1-based indexing, where each entry corresponds to a global DOF number.
      num_total_dofs (int): The total number of degrees of freedom in the system, NOT equal to number of active equations.

    Returns:
      np.ndarray: The reordered global connectivity matrix with prescribed DOFs moved to the end.
    """
    # Create a copy of the global connectivity matrix to avoid modifying the original
    gcon = gcon.copy()

    # Process each displacement boundary condition (skipping the padded first row)
    for i in range(1, len(disp_BCs)):
        # Each line in disp_BCs is expected to be [Node Number, Local DOF, Displacement Value]
        # Convert Node Number and Local DOF to integers (displacement value is ignored here)
        node_num, local_dof, _ = disp_BCs[i].astype(int)
        # Retrieve the current global DOF for the specified node and local DOF
        curr_BC_global_dof = gcon[node_num, local_dof]
        # Adjust global DOF numbers:
        # For all entries with a DOF number greater than the current prescribed DOF, decrement by 1 to shift free DOFs upward.
        gcon[gcon > curr_BC_global_dof] -= 1
        # Set the prescribed DOF's new global number to the last position (num_total_dofs)
        gcon[node_num, local_dof] = num_total_dofs

    return gcon

def init_forces(force_BCs: np.ndarray, gcon: np.ndarray, num_total_dofs: int) -> np.ndarray:
    """
    Initializes the global force vector based on the specified force boundary conditions.

    This function creates a force vector of size (num_total_dofs + 1) to support 1-based indexing.
    The first element (index 0) is padded with NaN since it is not used. Then, for each force boundary
    condition provided in the force_BCs array, the function:
      1. Extracts the node number, local DOF, and force value.
      2. Casts the node number and local DOF to integers, as they are used as indices.
      3. Uses the global connectivity matrix (gcon) to determine the corresponding global DOF.
      4. Adds the force value to the corresponding entry in the force vector.

    Parameters:
      force_BCs (np.ndarray): A 2D array of force boundary conditions (padded for 1-based indexing), where each row
                              (after the padded row) has the format [Node Number, Local DOF, Force Value].
      gcon (np.ndarray): The global connectivity matrix (padded for 1-based indexing) that maps each node's local DOF to a global DOF.
      num_total_dofs (int): The total number of degrees of freedom in the system.

    Returns:
      np.ndarray: A global force vector of shape (num_total_dofs + 1, 1) with force contributions applied to prescribed DOFs.
                  The first element is NaN as it is used solely for 1-based indexing.
    """
    # Create a force vector initialized to zeros with an extra element for 1-based indexing.
    forces = np.zeros(shape=(num_total_dofs + 1, 1), dtype=float)
    forces[0] = np.nan  # The 0th element is a placeholder for 1-based indexing.
    
    # Process each force boundary condition (skip the padded first row).
    for i in range(1, len(force_BCs)):
        # Each row is expected to be [Node Number, Local DOF, Force Value].
        node_num, local_dof, force_val = force_BCs[i]
        # Cast node number and local DOF to integers (indices) since they may be provided as floats.
        node_num, local_dof = int(node_num), int(local_dof)
        # Retrieve the corresponding global DOF using the global connectivity matrix.
        curr_BC_global_dof = int(gcon[node_num, local_dof])
        # Add the force boundary condition value to the force vector at the global DOF index.
        forces[curr_BC_global_dof] += force_val

    return forces

def create_known_displacements_matrix(displacements: np.ndarray, num_disp_BCs: int, num_nodes: int, dof_per_node: int) -> np.ndarray:
    """
    Creates a matrix of known (prescribed) displacements from the displacement boundary conditions.

    This function constructs a matrix where each entry corresponds to a prescribed displacement value
    for a given node and its local degree of freedom (DOF). The matrix dimensions are (num_nodes + 1) x (dof_per_node + 1),
    with an extra row and column to facilitate 1-based indexing (the 0th row and 0th column are placeholders).

    Parameters:
      displacements (np.ndarray): A 2D array of displacement boundary conditions, padded for 1-based indexing.
                                  Each row (after the dummy first row) is expected to have the format:
                                  [node number, local DOF, prescribed displacement value].
      num_disp_BCs (int): The number of displacement boundary conditions (excluding the padded row).
      num_nodes (int): The total number of nodes in the system (excluding the padded row).
      dof_per_node (int): The number of degrees of freedom per node.

    Returns:
      np.ndarray: A matrix of shape (num_nodes + 1, dof_per_node + 1) where each entry represents
                  the prescribed displacement for the corresponding node and DOF. Entries with no prescribed
                  displacement remain 0.
    """
    # Initialize the known displacements matrix with zeros, including an extra row and column for 1-based indexing.
    known_displacements = np.zeros((num_nodes + 1, dof_per_node + 1))
    # Populate the known displacements matrix using the displacement boundary conditions.
    for i in range(1, num_disp_BCs + 1):
        node = int(displacements[i, 0])   # Node number from displacement BC
        dof  = int(displacements[i, 1])   # Local DOF number from displacement BC
        value = displacements[i, 2]       # Prescribed displacement value
        known_displacements[node, dof] = value
        # So, for node i, the row looks like: 
        # [NaN, prescribed value for DOF 1, prescribed value for DOF 2, ...]
    return known_displacements

def assemble_element_by_element(elenodes: np.ndarray, cosines: np.ndarray,gcon: np.ndarray, displacements: np.ndarray, forces: np.ndarray, 
                                Es: np.ndarray, As: np.ndarray, Is: np.ndarray, Ls: np.ndarray, dof_per_node: int, num_active_dofs: int, num_disp_BCs: int, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
      Assembles the global stiffness matrix and modifies the force vector using an element-by-element approach 
      for a **2D frame structure** with axial and bending effects.

      This function constructs the **reduced global stiffness matrix** and **reduced force vector** 
      by iterating over each element and incorporating local stiffness contributions while handling prescribed displacements.

      **Process:**
        1. **Initialize**:
          - The reduced force vector (for free DOFs).
          - The reduced global stiffness matrix.
          - The known displacement matrix for boundary conditions.

        2. **Loop through elements**:
          - Compute the **local stiffness matrix** (before rotation).
          - Compute the **transformation matrix** for global-to-local conversion.
          - Rotate the local stiffness matrix.
          - Assemble the element stiffness contributions into the global system.

        3. **Handle boundary conditions**:
          - If a DOF corresponds to a prescribed displacement, adjust the force vector by moving contributions 
            from the stiffness matrix to the RHS.

      **Parameters:**
        elenodes : np.ndarray
            (num_elements+1, 3) array mapping elements to their nodes. Columns: [nan, node_1, node_2].
        cosines : np.ndarray
            (num_elements+1, 3) array of directional cosines. Columns: [nan, cos(theta), sin(theta)].
        gcon : np.ndarray
            (num_nodes+1, dof_per_node+1) global connectivity matrix mapping local DOFs to global DOFs.
        displacements : np.ndarray
            (num_disp_BCs+1, 3) prescribed displacement conditions. Columns: [node, local DOF, prescribed value].
        forces : np.ndarray
            (num_active_dofs+1, ) force vector (external loads).
        Es : np.ndarray
            (num_elements+1, ) array of Young's modulus values (1-based indexing).
        As : np.ndarray
            (num_elements+1, ) array of cross-sectional areas (1-based indexing).
        Is : np.ndarray
            (num_elements+1, ) array of second moments of area (1-based indexing).
        Ls : np.ndarray
            (num_elements+1, ) array of element lengths (1-based indexing).
        dof_per_node : int
            Number of degrees of freedom per node (typically **3** for a **2D frame**: [u, v, theta]).
        num_active_dofs : int
            Number of free DOFs after applying displacement boundary conditions.
        num_disp_BCs : int
            Number of prescribed displacement boundary conditions.
        num_nodes : int
            Total number of nodes in the system.

      **Returns:**
        Tuple[np.ndarray, np.ndarray]:
            - `reduced_global_stiff` (num_active_dofs+1, num_active_dofs+1) → Reduced global stiffness matrix.
            - `reduced_forces` (num_active_dofs+1, ) → Adjusted force vector after handling prescribed displacements.
    """

    # Create a reduced force vector for the free DOFs (preserving 1-based indexing; index 0 is a dummy).
    reduced_forces = forces.copy()[0:num_active_dofs+1]
    # Initialize the reduced global stiffness matrix for free DOFs.
    stiff_size = num_active_dofs + 1  # extra "+1" for 1-based indexing
    reduced_global_stiff = np.zeros((stiff_size, stiff_size), dtype=float)
    # Build a matrix for known (prescribed) displacements from the displacement BCs.
    known_displacements = create_known_displacements_matrix(displacements, num_disp_BCs, num_nodes, dof_per_node)

    # Loop over each element (starting at index 1 due to 1-based indexing).
    for element_i in range(1, len(elenodes)):
        ##################################
        # COMPUTE K_LOCAL FOR A FRAME ELEMENT:
        
        # The size is (2*dof_per_node)x(2*dof_per_node) for 2 nodes per element with an extra row and column for 1-based indexing.
        local_stiffness = np.zeros((2 * dof_per_node + 1, 2 * dof_per_node + 1), dtype=float)
        
        # EXTRACT NECESSARY quantities
        E, I, A, L = Es[element_i], Is[element_i], As[element_i], Ls[element_i]
        
        # first, assemble K_local that represetns stiffness for a horiztonal beam/frame-element (before rotation):
        local_stiffness[1:, 1:] =  np.array([
            [E * A / L, 0, 0, -E * A / L, 0, 0],
            [0, 12 * E * I / L**3, 6 * E * I / L**2, 0, -12 * E * I / L**3, 6 * E * I / L**2],
            [0, 6 * E * I / L**2,  4 * E * I / L, 0, -6 * E * I / L**2,  2 * E * I / L],
            [-E * A / L, 0, 0, E * A / L, 0, 0],
            [0, -12 * E * I / L**3, -6 * E * I / L**2, 0, 12 * E * I / L**3, -6 * E * I / L**2],
            [0, 6 * E * I / L**2, 2 * E * I / L, 0, -6 * E * I / L**2, 4 * E * I / L]
        ], dtype=float)
  
        # second, assemble Rotational matrix for the entire frame element
        local_rotational = np.zeros((2 * dof_per_node + 1, 2 * dof_per_node + 1), dtype=float)
        
        # compute cosine and sines angles for each element, recall, element consine matrix has [nan, c1, c2]
        cos_angle = cosines[element_i, 1]
        sin_angle = cosines[element_i, 2]
        
        # form a GENERAL rotational matrix:
        rotational_matrix = np.array([[cos_angle, sin_angle, 0], [-sin_angle, cos_angle, 0], [0, 0, 1]])
        
        # the full element rotational matrix, is rotational_matrix on diagonal and zeros else where:
        local_rotational[1:dof_per_node+1, 1:dof_per_node+1] = rotational_matrix
        local_rotational[dof_per_node+1:, dof_per_node+1:] = rotational_matrix
        
        # now perform the matrix multiplications of: [R_ele]^-1 [K_local] [R_ele] --> [K_ele]
        local_stiffness = np.linalg.multi_dot([local_rotational.T, local_stiffness, local_rotational])
        ##################################
        # Assemble the LOCAL stiffness contributions into the GLOBAL stiffness matrix.
        # NOTE: K_ele's row/col architecture:
                  # 1st half is for node1 and the 2nd half is for node2,
                  # and each node has dof_per_node (2 or 3) degrees of freedom.

        # NOTE: We go Rows first
        for node_i in [1, 2]: # Loop over the two NODES of the element.
          for node_dof_i in range(1, dof_per_node + 1): # Loop over each LOCAL dof for the current node.
                # get the LOCAL ROW index in the element's stiffness matrix.
                # we are accessing each ROW of K_ele based on node# and local dof#
                dof_i_local = (node_i - 1) * dof_per_node + node_dof_i # essentially the area of a rectangle
                
                # get the node number local node (1 or 2) for current element
                local_node_num_i = int(elenodes[element_i, node_i])
                # Map the LOCAL node and DOF to the GLOBAL DOF using the global connectivity matrix.
                dof_i_global = int(gcon[local_node_num_i, node_dof_i])
                
                # Assemble only if the global DOF is free / active equation.
                # we don't care about equatiosn that are associated with prescribed displacements. 
                if dof_i_global <= num_active_dofs:
                    # NOTE: now we do the same thing for the COLUMNS, but note that removing columns requires MOVING known values to the RHS
                    for node_j in [1, 2]: # Loop over the two nodes (columns) of the element.
                        for node_dof_j in range(1, dof_per_node + 1):
                            # Calculate the local column index in the element's stiffness matrix.
                            dof_j_local = (node_j - 1) * dof_per_node + node_dof_j

                            # get the node number local node (1 or 2) for current element
                            local_node_num_j = int(elenodes[element_i, node_j])
                            # Map the local DOF to the corresponding global DOF.
                            dof_j_global = int(gcon[local_node_num_j, node_dof_j])
                            
                            # again, check if the column is for corresponding free row.
                            if dof_j_global <= num_active_dofs:
                                # For FREE/ACTIVE DOFs, add the local stiffness contribution to the global stiffness matrix.
                                reduced_global_stiff[dof_i_global, dof_j_global] += local_stiffness[dof_i_local, dof_j_local]
                            # if NOT, move columns that are beyond active number of equations, to the RHS
                            else:
                                # For prescribed DOFs, adjust the force vector by subtracting the contribution 
                                u = known_displacements[local_node_num_j, node_dof_j]
                                reduced_forces[dof_i_global] -= local_stiffness[dof_i_local, dof_j_local] * u   
    return reduced_global_stiff, reduced_forces

def solve_reduced_nodal_displacements(reduced_stiffness: np.ndarray, RHS_reduced: np.ndarray):
    """
    Solves for the unknown nodal displacements in a reduced finite element system.

    This function takes the reduced global stiffness matrix (after removing prescribed DOFs)
    and the corresponding reduced force vector (RHS) to solve for the unknown nodal displacements.

    Parameters:
      reduced_stiffness : np.ndarray
          The reduced global stiffness matrix after eliminating constrained DOFs.
      RHS_reduced : np.ndarray
          The reduced force vector after applying prescribed displacement boundary conditions.

    Returns:
      np.ndarray
          A 1D array containing the computed unknown nodal displacements for free DOFs.
    """
    # Extract the actual stiffness matrix and RHS vector by removing the first row and column.
    # These first entries are dummy values for 1-based indexing and do not affect the solution.
    return np.linalg.solve(reduced_stiffness[1:, 1:], RHS_reduced[1:]) 

def insert_solved_displacements(full_disp: np.ndarray, solved_disp: np.ndarray, gcon: np.ndarray, num_active_dofs: int) -> np.ndarray:
    """
    Inserts the computed displacements of free (active) DOFs into the full nodal displacement matrix.

    This function updates the full nodal displacement matrix by filling in the solved displacement values 
    for the active DOFs while preserving the prescribed (fixed) displacement values.

    Parameters:
      full_disp : np.ndarray
          The full nodal displacement array where all DOFs (free and prescribed) are stored.
      solved_disp : np.ndarray
          The computed displacements of free DOFs obtained from solving the reduced system.
      gcon : np.ndarray
          The global connectivity matrix mapping local node DOFs to global DOF numbers.
      num_active_dofs : int
          The total number of active (free) DOFs in the system.
      num_nodes : int
          The total number of nodes in the structure.
      dof_per_node : int
          The number of degrees of freedom per node.

    Returns:
      np.ndarray
          The updated full displacement array (matrix) with solved free DOFs filled in.

    Notes:
      - Only active DOFs (those that were not prescribed) are updated with solved values.
      - Prescribed DOFs (displacement boundary conditions) remain unchanged.
    """
    # Flatten the global connectivity matrix to get a 1D array of global DOF indices
    flat_gcon = gcon.flatten()
    # Create a boolean mask that identifies free (active) DOFs:
    # because we only need to fill in solved displacements, we already have prescribed displacements in our matrix.
    # A DOF is free if its global index is <= num_active_dofs.
    free_dofs_mask = flat_gcon <= num_active_dofs
    # Flatten the full displacement matrix so we can update it using 1D indexing
    flat_full_disp = full_disp.flatten()
    # For free DOFs, update the displacement values with the solved values.
    flat_full_disp[free_dofs_mask] = solved_disp[flat_gcon[free_dofs_mask].astype(int)]

    # Reshape the updated displacement array back to its original matrix form
    return flat_full_disp.reshape(full_disp.shape)

def compute_axial_transverse_displacements(nodal_disps: np.ndarray, elenodes: np.ndarray, cosines: np.ndarray, num_elements: int, dof_per_node: int) -> np.ndarray:
    """
    Computes the axial and transverse displacements for each element in the local coordinate system.

    This function transforms global nodal displacements into element-local axial and transverse components
    using directional cosines. It also retains rotational displacements.

    **Transformation Process:**
      1. Extract nodal displacement values for both nodes of each element.
      2. Construct the transformation matrix using directional cosines.
      3. Convert global displacements into local axial and transverse components.
      4. Append rotational DOFs to the output matrix.

    **Parameters:**
      nodal_disps : np.ndarray
          (num_nodes+1, 4) matrix containing nodal displacements with 1-based indexing.
          Columns: [nan, u, v, theta].
      elenodes : np.ndarray
          (num_elements+1, 3) matrix mapping elements to their node numbers.
          Columns: [nan, node_1, node_2].
      cosines : np.ndarray
          (num_elements+1, 3) matrix containing directional cosines.
          Columns: [nan, cos(theta), sin(theta)].
      num_elements : int
          Number of elements in the system.
      dof_per_node : int
          Degrees of freedom per node (typically **3** for a **2D frame**: [u, v, theta]).

    **Returns:**
      np.ndarray
          (num_elements+1, 2 * dof_per_node + 1) matrix where each row contains: [nan, U1^a, U1^T, U2^a, U2^T, θ1, θ2].
    """
    # Initialize matrix to store transformed displacements (1-based indexing).
    axial_trans_disps = np.full(shape=(num_elements+1, 2 * dof_per_node + 1), fill_value=np.nan) # 4 axial & trans components and 2 more for thetas
    # Iterate over elements (1-based indexing)
    for element_i in range(1, num_elements + 1): # 1 based indexing
        # Extract node numbers
        node_1_num, node_2_num = int(elenodes[element_i, 1]), int(elenodes[element_i, 2])
        # get cosines and sines of each element:
        cosine, sine = cosines[element_i, 1], cosines[element_i, 2]
        # form transformation matrix:
        disp_transformations = np.array([
                    [cosine, sine, 0, 0],
                    [-sine, cosine, 0, 0],
                    [0, 0, cosine, sine],
                    [0, 0, -sine, cosine]])
        # Extract global nodal displacements
        u1, v1 = nodal_disps[node_1_num, 1:3]
        u2, v2 = nodal_disps[node_2_num, 1:3]
        theta1, theta2 = nodal_disps[node_1_num, 3], nodal_disps[node_2_num, 3]
        # apply transformations to nodal displacements
        axial_trans_disps[element_i, 1: -2] = disp_transformations @ np.array([u1, v1, u2, v2])
        # Store rotational DOFs
        axial_trans_disps[element_i, -2] = theta1
        axial_trans_disps[element_i, -1] = theta2
    
    return axial_trans_disps

def compute_element_forces_moments(axial_trans_disps: np.ndarray, Es: np.ndarray, Is: np.ndarray, As: np.ndarray, Ls: np.ndarray) -> np.ndarray:
    """
      Computes axial force, shear force, and bending moments for each element in a 2D frame structure.

      The element forces and moments are computed as follows:
        - Axial force (N)  = (E * A / L) * (u2^a - u1^a)
        - Shear force (V)  = (12 * E * I / L^3) * (u1^T - u2^T) + (6 * E * I / L^2) * (theta1 + theta2)
        - Moment 1 (M1)    = (6 * E * I / L^2) * (u2^T - u1^T) - (2 * E * I / L) * (2 * theta1 + theta2)
        - Moment 2 (M2)    = (6 * E * I / L^2) * (u1^T - u2^T) + (2 * E * I / L) * (theta1 + 2 * theta2)

      Parameters:
        axial_trans_disps : np.ndarray
            (num_elements+1, 7) matrix containing local displacements.
            Columns: [nan, u1^a, u1^T, u2^a, u2^T, theta1, theta2].
        Es : np.ndarray
            (num_elements+1,) array of Young’s modulus values.
        Is : np.ndarray
            (num_elements+1,) array of second moments of area.
        As : np.ndarray
            (num_elements+1,) array of cross-sectional areas.
        Ls : np.ndarray
            (num_elements+1,) array of element lengths.

      Returns:
        np.ndarray
            (num_elements+1, 5) matrix with:
            [nan, Axial Force (N), Shear Force (V), Moment 1 (M1), Moment 2 (M2)].

      Notes:
      - Shear and bending moments include rotational DOFs in their formulation.
    """
    # Initialize matrix for forces and moments (1-based indexing)
    forces_moments = np.full(shape=(len(axial_trans_disps), 4 + 1), fill_value=np.nan)
    # Compute forces and moments for each element
    for ele_i in range(1, len(axial_trans_disps)):
        E, A, I, L = Es[ele_i], As[ele_i], Is[ele_i], Ls[ele_i]
        u1A, u1T, u2A, u2T, theta1, theta2 = axial_trans_disps[ele_i, 1:]
        # compute beam force N
        forces_moments[ele_i, 1] = E * A / L * (u2A - u1A)
        # compute beam shear force V
        forces_moments[ele_i, 2] = ((12 * E * I / (L**3)) * (u1T - u2T)) + ((6 * E * I / (L**2)) * (theta1 + theta2))
        # compute moment 1 M1
        forces_moments[ele_i, 3] = ((6 * E * I / (L**2)) * (u2T - u1T)) - ((2 * E * I / L) * (2 * theta1 + theta2))
        # compute moment 2 M2
        forces_moments[ele_i, 4] = ((6 * E * I / (L**2)) * (u1T - u2T)) + ((2 * E * I / L) * (theta1 + 2 * theta2))
    
    return forces_moments
    
def compute_external_forces_moments(internal_forces_moments: np.ndarray, cosines: np.ndarray, elenodes: np.ndarray, num_nodes: int, dof_per_node: int) -> np.ndarray:
    """
    Computes the external forces and moments at each node based on internal element forces.

    Parameters:
    -----------
    internal_forces_moments : np.ndarray
        (num_elements+1, 5) matrix with element-level forces and moments.
        Columns: [nan, N (axial), V (shear), M1, M2].
    force_BCs : np.ndarray
        (num_force_BCs+1, 3) matrix with prescribed forces.
        Columns: [node#, dof#, force_value].
    cosines : np.ndarray
        (num_elements+1, 3) matrix of element directional cosines.
        Columns: [nan, cos(theta), sin(theta)].
    elenodes : np.ndarray
        (num_elements+1, 3) matrix mapping elements to their node numbers.
        Columns: [nan, node_1, node_2].
    num_nodes : int
        Total number of nodes in the system.
    num_force_BCs : int
        Number of prescribed force boundary conditions.
    dof_per_node : int
        Degrees of freedom per node (typically 3: [Fx, Fy, M] for 2D frames).

    Returns:
    --------
    np.ndarray
        (num_nodes+1, dof_per_node+1) matrix with external forces and moments at each node.
        Columns: [nan, Fx, Fy, M].
    """
    # Initialize the global external forces matrix with zeros. We add one extra row and column for 1-based indexing.
    external_forces_moments = np.zeros((num_nodes + 1, dof_per_node + 1))
    external_forces_moments[0, :] = external_forces_moments[:, 0] = np.nan
    
    # compute forces and moments for all nodes:
    for ele_i in range(1, len(elenodes)):
        # extract local node numbers
        node_1, node_2 = elenodes[ele_i, 1:3].astype(int)
        # extract element's axial, shear, and moments:
        N, V, M1, M2= internal_forces_moments[ele_i, 1:]
        # get cosines and sines for this element:
        cosine, sine = cosines[ele_i, 1:]
        # sum up external forces contributions
        # Fx:
        external_forces_moments[node_1, 1] += (-N * cosine) - (V * sine)
        external_forces_moments[node_2, 1] += -((-N * cosine) - (V * sine)) # equal and opposite
        # Fy:
        external_forces_moments[node_1, 2] += (-N * sine) + (V * cosine)
        external_forces_moments[node_2, 2] += -((-N * sine) + (V * cosine))
        # M:
        external_forces_moments[node_1, 3] += -M1
        external_forces_moments[node_2, 3] += M2

    return external_forces_moments

def write_nodal_displacements_file(nodes:np.ndarray, nodal_disp: np.ndarray, num_nodes:int, dof_per_node: int) -> pd.DataFrame:
    # pull out x and y coordinates of each node to form full data table
    data = np.hstack([nodes[1:, 1:], nodal_disp[1:, 1:]]) # Skip the 0th row/col for 1-based indexing

    # Construct the DataFrame
    df = pd.DataFrame(
        data=data,
        columns=['x', 'y', 'u', 'v', 'theta'],
        index=np.arange(1, num_nodes+1)
    )
    # Set the index name explicitly
    df.index.name = 'Node #'

    return df.round(6)

def write_external_forces_file(nodes:np.ndarray, ext_forces_moments: np.ndarray, num_nodes:int) -> pd.DataFrame:
    # pull out x and y coordinates of each node to form full data table
    data = np.hstack([nodes[1:, 1:], ext_forces_moments[1:, 1:]]) # Skip the 0th row/col for 1-based indexing

    # Construct the DataFrame
    df = pd.DataFrame(
        data=data,
        columns=['x', 'y', 'F_x', 'F_y', 'M'],
        index=np.arange(1, num_nodes+1)
    )
    # Set the index name explicitly
    df.index.name = 'Node #'

    return df.round(6)

def write_element_forces_file(element_forces_moments: np.ndarray, elenodes: np.ndarray, num_elements: int) -> pd.DataFrame:
    # Construct the DataFrame
    df = pd.DataFrame(
        data=element_forces_moments[1:, 1:],
        columns=['N', 'V', 'M1', 'M2'],
        index=np.arange(1, num_elements+1)
    )
    # Set the index name explicitly
    df.index.name = 'Element #'

    return df.round(6)

def main():

    # Define the directory containing the input files.
    directory_path = "/Users/kis/Desktop/COE321K/hw5/discussion_test_inputs/"
    # Alternative directories can be set if needed.
    print(f"ACCESSING FOLDER: {directory_path}")

    # -------------------------------
    # STEP 1: READ NODE INFORMATION
    # -------------------------------
    # Read node data: coordinates, total nodes, and dimensions (2D/3D).
    nodes, num_nodes, num_dims = read_nodes_file(filepath = directory_path + 'nodes')

    # For a 2D frame, DOFs per node equals 3, x, y, theta
    dof_per_node = 3 

    # Compute the total number of DOFs in the structure (1-based indexing).
    num_total_dofs = (len(nodes) - 1) * dof_per_node

    # Build the global connectivity matrix mapping each node's local DOFs to global DOF numbers.
    global_connectivity = build_global_connectivity(num_nodes, dof_per_node)
    # -------------------------------
    # STEP 2: READ ELEMENT CONNECTIVITY
    # -------------------------------
    # Read element data: connectivity, element nodes, and total number of elements.
    elements, elenodes, num_elements = read_elements_file(filepath=directory_path + 'elements')

    # Compute element lengths and directional cosines used in transformation matrices.
    Ls, cosines = compute_lengths_and_cosines(elements, nodes)

    # -------------------------------
    # STEP 3: READ FORCE AND DISPLACEMENT BOUNDARY CONDITIONS
    # -------------------------------
    # Read external force boundary conditions.
    force_BCs, num_force_BCs = read_forces_file(filepath=directory_path + 'forces')

    # Read prescribed displacement (fixed DOFs) boundary conditions.
    disp_BCs, num_disp_BCs = read_displacements_file(filepath=directory_path + 'displacements')

    # Determine the number of active (free) DOFs by subtracting prescribed DOFs.
    num_active_dofs = num_total_dofs - num_disp_BCs

    # Reorder the global connectivity matrix so that free DOFs come first.
    global_connectivity = reorder_global_connectivity(disp_BCs, global_connectivity, num_total_dofs)

    # Initialize the global force vector using external loads and the global connectivity.
    forces = init_forces(force_BCs, global_connectivity, num_total_dofs)

    # -------------------------------
    # STEP 4: ASSEMBLE GLOBAL STIFFNESS MATRIX
    # -------------------------------
    # Extract material properties: Young's modulus (E) and cross-sectional area (A), and 2nd moment of area, for each element.
    Es = elements[:, -3]  # Column for Young's modulus.
    As = elements[:, -2]  # Column for cross-sectional area. (A)
    Is = elements[:, -1]  # column for Second moment of area (I)
 

    # Assemble the global stiffness matrix and adjust the force vector for prescribed displacements.
    K_red, F_red = assemble_element_by_element(elenodes, cosines, global_connectivity, disp_BCs, forces, Es, As, Is, Ls, dof_per_node, num_active_dofs, num_disp_BCs, num_nodes)

    # -------------------------------
    # STEP 5: SOLVE FOR NODAL DISPLACEMENTS
    # -------------------------------
    # Solve the reduced system: K_red * u_free = F_red for the unknown (free) nodal displacements.
    solved_U_red = solve_reduced_nodal_displacements(K_red, F_red)

    # Reconstruct the full nodal displacement matrix by combining solved displacements with prescribed ones.
    nodal_disps = create_known_displacements_matrix(disp_BCs, num_disp_BCs, num_nodes, dof_per_node)

    # Pad the solved free displacement vector to maintain 1-based indexing.
    solved_U_red = np.pad(solved_U_red, ((1,0), (0,0)), 'constant', constant_values=np.nan).reshape((len(solved_U_red)+1,))

    # Insert the solved free displacements into the full displacement matrix.
    nodal_disps = insert_solved_displacements(nodal_disps, solved_U_red, global_connectivity, num_active_dofs)

    # -------------------------------
    # STEP 6: COMPUTE ELEMENT STRETCH, STRAIN, AND FORCE
    # -------------------------------
    # compute axial and transverse components of nodal displacements:
    nodal_axial_trans_disps = compute_axial_transverse_displacements(nodal_disps,elenodes, cosines, num_elements, dof_per_node)
    # Compute the beam forces and moments
    element_forces_moments = compute_element_forces_moments(nodal_axial_trans_disps, Es, Is, As, Ls)
    # -------------------------------
    # STEP 7: COMPUTE GLOBAL EXTERNAL FORCES
    # -------------------------------
    # Accumulate the force contributions from all elements into the global external forces array.
    external_forces_moments = compute_external_forces_moments(element_forces_moments, cosines, elenodes, num_nodes, dof_per_node)
    # -------------------------------
    # STEP 8: OUTPUT RESULTS
    # -------------------------------
    
    ## DEBUGGING COMMENTS:
    """
    print(f"\nNum of Nodes: {num_nodes}, Num of Dims: {num_dims}") # DEBUG
    print(f"\nNodes Matrix:\n{nodes}")
    print(f"\nNum of Total Dofs: {num_total_dofs}")
    print(f"\nGCON Matrix:\n{global_connectivity}")
    print(f"\nNum of Elements: {num_elements}") # DEBUG
    print(f"\n Elements Matrix:\n{elements}")
    print(f"\n elenodes Matrix:\n{elenodes}")
    print(f"\nelement cosines: {cosines}")
    print(f"\n Forces Matrix:\n{force_BCs}")
    print(f"\n disp_BCs Matrix:\n{disp_BCs}")
    print(f"num of disp BCs: {num_disp_BCs}")
    print(f"num of active DOFs: {num_active_dofs}")
    print(f"\nGCON Matrix after reordering:\n{global_connectivity}")    
    print(f"\ninitialized global force vector:\n{forces}")
    print(f"Es: {Es}")
    print(f"As: {As}")
    print(f"Is: {Is}")
    print(f"Ls: {Ls}")
    print(f"\nK_reduced:\n{K_red}")
    print(f"\nF_reduced:\n{F_red}")
    print(f"\nsolved_U_red:\n{solved_U_red}")
    print(f"\n full nodal displacements:\n{np.round(nodal_disps[1:, 1:], 2)}")
    print(f"\n axial and transverse components of nodal displacements and two thetas:\n{np.round(nodal_axial_trans_disps,2)}")
    print(f"\n element_forces_moments:\n{np.round(element_forces_moments,2)}")
    print(f"\n external_forces_moments:\n{np.round(external_forces_moments,2)}")
    """
    print()
    # Write the computed nodal displacements to a DataFrame for display.
    nodal_displacements_df = write_nodal_displacements_file(nodes, nodal_disps, num_nodes, dof_per_node)
    print("<<< NODAL DISPLACEMENTS DATA TABLE >>>")
    print(tabulate(nodal_displacements_df, headers="keys", tablefmt="fancy_grid", showindex=True))

    # Write the computed external forces to a DataFrame for display.
    external_forces_df = write_external_forces_file(nodes, external_forces_moments, num_nodes)
    print("\n<<< EXTERNAL FORCES AND MOMENTS DATA TABLE >>>")
    print(tabulate(external_forces_df, headers="keys", tablefmt="fancy_grid", showindex=True))

    # Write the computed element strains and forces to a DataFrame for display.
    element_properties_df = write_element_forces_file(element_forces_moments, nodes, num_elements)
    print("\n<<< ELEMENT FORCES AND MOMENTS DATA TABLE >>>")
    print(tabulate(element_properties_df, headers="keys", tablefmt="fancy_grid", showindex=True))


if __name__ == "__main__":
    main()