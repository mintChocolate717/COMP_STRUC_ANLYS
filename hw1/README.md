# COE 321K - Homework 1

This repository contains solutions for **Homework 1** of **COE 321**, which involves structural analysis of pin-ended strut structures and matrix computations.

## **Overview**
The code solves for:
1. **Support reactions and internal forces** in a system of pin-ended, weightless struts using equilibrium equations and stiffness matrices.
2. **Deformations and displacements** using bar deformation principles and global stiffness matrix reduction.
3. **Matrix operations** such as solving for unknown forces, displacements, and applying numerical simplifications.

---

## **Files**
- `hw1.py` → Python script solving the homework problems.
- `Homework 1.pdf` → Homework questions.
- `COE321K-HW1.pdf` → Handwritten Solutions

---

## **Problem Descriptions & Solutions**
### **Problem 1: Structural Analysis**
#### **Objective**
- Determine support reactions and all internal forces in the given truss system.
- Use **equilibrium equations**, **bar deformation equations**, and **global stiffness matrix reduction**.

#### **Approach**
1. **Define Equilibrium Equations**:
   - Construct the **M matrix**, representing force coefficients.
   - Define **external force vector** \( F_{ext} \).
   - Define **internal force vector** \( N_{bars} \).

2. **Bar Deformation Equations**:
   - Construct the **C_δ matrix**, representing deformation relationships based on Young’s modulus \( E \), cross-sectional area \( A \), and length \( L \).

3. **Global Stiffness Matrix Computation**:
   - Compute the **global stiffness matrix** \( K \) using:
     \[
     K = M C_δ N_δ
     \]

4. **Reduction Using Boundary Conditions**:
   - Reduce the **stiffness matrix** \( K \) to a **2×2** matrix.
   - Solve for unknown displacements \( U_{Dx} \) and \( U_{Dy} \).

5. **Compute Internal & External Forces**:
   - Solve for internal forces \( N_{bars} \) and external reaction forces \( F_{ext} \).

---

### **Problem 2: Extended Structural Analysis**
#### **Objective**
- Solve a **larger truss system** using the same equilibrium and deformation principles.

#### **Approach**
1. **Define Force and Stiffness Matrices**:
   - Construct an **M matrix** (10×7 system).
   - Define the **C_δ matrix** (7×7 deformation coefficient matrix).

2. **Global Stiffness Computation**:
   - Compute:
     \[
     K = M C_δ N_δ
     \]
   - Reduce **K** by removing known reaction force indices \( \{1,3,4,10\} \).

3. **Solve for Displacements**:
   - Compute **U_reduced** and map back into **U_nodes**.

4. **Compute Forces**:
   - Solve for **internal forces** \( N_{bars} \) and **external forces** \( F_{ext} \).

---

## **How to Run the Code**
Ensure you have **Python** and **SymPy** installed. Run the script with:

```bash
python3 hw1.py
