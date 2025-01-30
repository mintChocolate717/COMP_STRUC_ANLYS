import numpy as np
from sympy import *
import math

# Enable global pretty printing and keep fractions in fractional form
init_printing(use_unicode=True, pretty_print=True)

def question_1():
    """
    Equilibrium - Define M matrix, and maybe (F_ext vector, N_bars vector)
    """
    P = symbols('P')
    # the global M matrix
    M = Matrix([
        [-1,  0,   0],
        [ 0,  0,   0],
        [ 0, -3 / 5, 0],
        [ 0,  4 / 5, 0],
        [ 0,  0,   0],
        [ 0,  0,   1],
        [ 1,  3 / 5, 0],
        [ 0, -4 / 5, -1]
    ])
    """
    Bar-Deformation - define C_δ matrix
    """
    # Define the C_delta matrix
    E, A, L = symbols('E A L')
    C_δ = E * A / L * Matrix([
        [E * A / (3*L),    0,           0],
        [0,        E * A / (5 * L),  0],
        [0,        0,        E * A / (4 * L)]
    ])
    """
    Kinematics - define N_δ matrix
    """
    # note that N_delta is just transpose of global M mx:
    N_δ = M.T

    """
    Combine above steps to formulate global stiffness matrix K and reduce it using boundar conditions
    """
    # compute global K, stiffness matrix
    K = M @ C_δ @ N_δ
    # reduced K matrix is just lower left 2 by 2 mx:
    K_reduced = K[6:, 6:]
    # F_ext_reduced
    F_ext_reduced = Matrix([0, -P])

    # solved U_reduced
    U_reduced = K_reduced.solve(F_ext_reduced).applyfunc(simplify)

    """
    Solving Rest, plugging back in to find N_bars
    """
    # plug in solved U_reduced back to global U_nodes
    U_nodes = zeros(8,1) # rest are zeros
    U_nodes[6] = U_reduced[0] # plug in U_Dx
    U_nodes[7] = U_reduced[1] # plug in U_Dy
    
    """
    Now, solve for N_bars unreduced and unreduced F_ext
    """
    # Find the internal forces:
    N_bars = C_δ @ N_δ @ U_nodes
    # then solve for external Force vector
    F_ext = M @ N_bars

    """
    Now, format the solutions in a aesthetically pleasing format
    """
    print(f"\n{"-"*20} Solutions for Problem 1 {"-"*20}") # a title for a problem
    print("\n Internal Forces AD, BD, CD: ")
    pprint(N_bars) # round to 3 decimal places
    print()
    print("\n External Forces Ax, ..., Dy: ")
    pprint(F_ext) # round to 3 decimal places
    print()

def question_2():
    """
    Equilibrium - Define M matrix, and maybe (F_ext vector, N_bars vector)
    """
    # internal force coeffs matrix M:
    M = Matrix([
    [0,-4/5, 0, 0, 0, 0, 0],
    [1, 3/5, 0, 0, 0, 0, 0],
    [0,  0,-4/5, -1, 0, 0, 0],
    [-1, 0, -3/5, 0, 0, 0, 0],
    [0, 4/5, 4/5, 0, 0, -4/5, 0],
    [0, -3/5, 3/5, 0, 1, 3/5, 0],
    [0, 0, 0, 1, 0, 0, -1],
    [0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 4/5, 1],
    [0, 0, 0, 0, 0, -3/5, 0]])

    """
    Bar-Deformation - define C_δ matrix
    """
    E, A, L = symbols('E A L') # EA
    C_δ = E * A / L * Matrix([
    [1/3, 0,   0,   0,   0,   0,   0],
    [0,   2/5, 0,   0,   0,   0,   0],
    [0,   0,   2/5, 0,   0,   0,   0],
    [0,   0,   0,   1/2, 0,   0,   0],
    [0,   0,   0,   0,   2/3, 0,   0],
    [0,   0,   0,   0,   0,   2/5, 0],
    [0,   0,   0,   0,   0,   0,   1/2]])

    """
    Kinematics - define N_δ matrix
    """
    N_δ = M.T # recall that N_δ is just the transpose of M matrix.

    """
    Combine above steps to formulate global stiffness matrix K and reduce it using boundar conditions
    """
    # define a force variable P
    P = symbols('P')
    # form a global stiffness matrix K:
    K = M @ C_δ @ N_δ

    # now, get F_ext_reduced where we only leave out known external forces.
    # in this case, this exlcudes, Ax, Bx, By, and Ey.
    F_ext_reduced = Matrix([0, 0,0,0,-P,0])

    # based on this, create a 6 x 6 K_reduced matrix, dropping rows and cols at 1,3,4,10:
    K_reduced = K.copy()
    # pick out indexes to remove
    index_to_delete = [0, 2, 3, 9]
    # now, remove these rows and columns in reverse order to ensure we don't disrupt the original indexing:
    for index in index_to_delete[::-1]:
        K_reduced.row_del(index) # delete the row
        K_reduced.col_del(index) # delete the column

    # Now, solve for U_reduced
    U_reduced = K_reduced.solve(F_ext_reduced)

    # with U_reduced, create unreduced U_nodes
    U_nodes = zeros(10,1) # initialize them to zeros

    # indexing variable for U_reduced
    index_reduced = 0
    # now, skip [0, 2, 3, 9] and fill in actual U_reduced to U_nodes
    for index in range(len(U_nodes)):
        # only substitute in U values from U_reduced (hence rows that weren't deleted previously)
        if index not in index_to_delete:
            # remeber that index-system are differrent between two vectors due to length difference.
            U_nodes[index] = U_reduced[index_reduced]
            # increment the index_reduced accordinlgy
            index_reduced+=1


    """
    Now, solve for N_bars unreduced and unreduced F_ext
    """
    # slve for N-bars
    N_bars = C_δ @ N_δ @ U_nodes
    # then solve for external Force vector
    F_ext = M @ N_bars

    print(f"\n{"-"*20} Solutions for Problem 2 {"-"*20}") # a title for a problem
    print("\n Internal Forces AB, ..., DE: ")
    pprint(N_bars) # round to 3 decimal places
    print()
    print("\n External Forces Ax, ..., Ey:")
    pprint(F_ext) # round to 3 decimal places
    print()





def main():
    question_1()
    question_2()

if __name__ == "__main__":
    main()



