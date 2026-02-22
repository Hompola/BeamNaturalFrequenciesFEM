import math
import numpy as np
from scipy.linalg import eig  # For solving natural frequencies

# This task was created to solve several more and more complex dynamics problems related to finding the natural
# frequencies of a beam at different levels of abstraction

##########################################################################
# Variables according to FigureOfBeamForFEM.png:

a = 2.1  # [m]
b = 7    # [m]
d = 0.045  # [m]

m0 = 25  # [kg]
rho = 7000  # [kg/m^3] (steel density)
E1 = 210_000_000_000  # [Pa] (Young's modulus of steel)

pi = math.pi

A = (d ** 2) * pi / 4
Iz = (d ** 4) * pi / 64

print("Cross-sectional area of the beam: A = " + str(A) + " m^2")
print("Second moment of area: Iz = " + str(Iz) + " m^4")

##########################################################################
# Determining the first three bending natural frequencies of the beam
# using the Finite Element Method (FEM), neglecting the concentrated mass m0.
# Using 1 element on both the AB and BC sections.
print("Task 1")
elem1 = [0, 1, 2, 3]
elem2 = [2, 3, 4, 5]

# Unconstrained degrees of freedom
free_dofs = [1, 2, 3]

def Me(Le):
    return (rho * A * Le) / 420 * np.array(
        [
            [156, 22 * Le, 54, -13 * Le],
            [22 * Le, 4 * Le ** 2, 13 * Le, -3 * Le ** 2],
            [54, 13 * Le, 156, -22 * Le],
            [-13 * Le, -3 * Le ** 2, -22 * Le, 4 * Le ** 2]
        ]
    )

def K0(Le):
    return (Iz * E1) / (Le ** 3) * np.array(
        [
            [12, 6 * Le, -12, 6 * Le],
            [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
            [-12, -6 * Le, 12, -6 * Le],
            [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2]
        ]
    )

Me_1 = Me(b)
Me_2 = Me(a)

MGLOB = np.zeros((6, 6))
for i in range(4):
    for j in range(4):
        MGLOB[elem1[i], elem1[j]] += Me_1[i, j]
        MGLOB[elem2[i], elem2[j]] += Me_2[i, j]

Mk = MGLOB[np.ix_(free_dofs, free_dofs)]

print("\nGlobal Mass Matrix:")
print(np.array_str(MGLOB, precision=3))
print("Condensed Mass Matrix:")
print(np.array_str(Mk, precision=3))

K0_1 = K0(b)
K0_2 = K0(a)

KGLOB = np.zeros((6, 6))
for i in range(4):
    for j in range(4):
        KGLOB[elem1[i], elem1[j]] += K0_1[i, j]
        KGLOB[elem2[i], elem2[j]] += K0_2[i, j]

Kk = KGLOB[np.ix_(free_dofs, free_dofs)]

print("\nGlobal Stiffness Matrix:")
print(np.array_str(KGLOB, precision=3))
print("Condensed Stiffness Matrix:")
print(np.array_str(Kk, precision=3))

eigenvalues = eig(Kk, Mk)[0]
real_eigenvalues = np.sort(eigenvalues[np.isreal(eigenvalues)].real)

alpha = np.sqrt(real_eigenvalues)
f = alpha / (2 * pi)

print("\nNatural frequencies [rad/s]:")
print(np.array_str(alpha, precision=5))
print("Natural frequencies [Hz]:")
print(np.array_str(f, precision=5))

##########################################################################
# Determining the first three bending natural frequencies of the beam
# using FEM, neglecting the concentrated mass m0.
# Using two elements of equal length on section AB
# and 1 element on section BC.
print("\nTask 2")

elem1 = [0, 1, 2, 3]
elem2 = [2, 3, 4, 5]
elem3 = [4, 5, 6, 7]

free_dofs = [1, 2, 3, 4, 5]

Me_1 = Me(b / 2)
Me_2 = Me(b / 2)
Me_3 = Me(a)

MGLOB = np.zeros((8, 8))
for i in range(4):
    for j in range(4):
        MGLOB[elem1[i], elem1[j]] += Me_1[i, j]
        MGLOB[elem2[i], elem2[j]] += Me_2[i, j]
        MGLOB[elem3[i], elem3[j]] += Me_3[i, j]

Mk = MGLOB[np.ix_(free_dofs, free_dofs)]

print("\nGlobal Mass Matrix:")
print(np.array_str(MGLOB, precision=3))
print("Condensed Mass Matrix:")
print(np.array_str(Mk, precision=3))

K0_1 = K0(b / 2)
K0_2 = K0(b / 2)
K0_3 = K0(a)

KGLOB = np.zeros((8, 8))
for i in range(4):
    for j in range(4):
        KGLOB[elem1[i], elem1[j]] += K0_1[i, j]
        KGLOB[elem2[i], elem2[j]] += K0_2[i, j]
        KGLOB[elem3[i], elem3[j]] += K0_3[i, j]

Kk = KGLOB[np.ix_(free_dofs, free_dofs)]

print("\nGlobal Stiffness Matrix:")
print(np.array_str(KGLOB, precision=3))
print("Condensed Stiffness Matrix:")
print(np.array_str(Kk, precision=3))

eigenvalues = eig(Kk, Mk)[0]
real_eigenvalues = np.sort(eigenvalues[np.isreal(eigenvalues)].real)

alpha = np.sqrt(real_eigenvalues)
f = alpha / (2 * pi)

print("\nNatural frequencies [rad/s]:")
print(np.array_str(alpha, precision=5))
print("Natural frequencies [Hz]:")
print(np.array_str(f, precision=5))

##########################################################################
# Determining the first three bending natural frequencies of the beam
# using FEM, taking the concentrated mass m0 into account
print("\nTask 3")

Me_1 = Me(b / 2)
Me_2 = Me(b / 2)
Me_3 = Me(a)

M_concentrated = np.zeros((4, 4))
M_concentrated[2, 2] = m0
Me_2 += M_concentrated

MGLOB = np.zeros((8, 8))
for i in range(4):
    for j in range(4):
        MGLOB[elem1[i], elem1[j]] += Me_1[i, j]
        MGLOB[elem2[i], elem2[j]] += Me_2[i, j]
        MGLOB[elem3[i], elem3[j]] += Me_3[i, j]

Mk = MGLOB[np.ix_(free_dofs, free_dofs)]

K0_1 = K0(b / 2)
K0_2 = K0(b / 2)
K0_3 = K0(a)

KGLOB = np.zeros((8, 8))
for i in range(4):
    for j in range(4):
        KGLOB[elem1[i], elem1[j]] += K0_1[i, j]
        KGLOB[elem2[i], elem2[j]] += K0_2[i, j]
        KGLOB[elem3[i], elem3[j]] += K0_3[i, j]

Kk = KGLOB[np.ix_(free_dofs, free_dofs)]

eigenvalues = eig(Kk, Mk)[0]
real_eigenvalues = np.sort(eigenvalues[np.isreal(eigenvalues)].real)

alpha = np.sqrt(real_eigenvalues)
f = alpha / (2 * pi)

print("\nNatural frequencies with concentrated mass [rad/s]:")
print(np.array_str(alpha, precision=5))
print("Natural frequencies with concentrated mass [Hz]:")
print(np.array_str(f, precision=5))

##########################################################################
# Instead of using the consistent mass matrix,
# the element mass matrices are computed using the lumped mass matrix,
# taking the moments of inertia into consideration.
print("\nTask 4")

def MeConcentratedMx(Le):
    return (rho * A * Le) / 420 * np.array(
        [
            [1, 0, 0, 0],
            [0, (Le ** 2) / 12, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, (Le ** 2) / 12]
        ]
    )

Me_1 += MeConcentratedMx(b / 2)
Me_2 += MeConcentratedMx(b / 2)
Me_3 += MeConcentratedMx(a)

MGLOB = np.zeros((8, 8))
for i in range(4):
    for j in range(4):
        MGLOB[elem1[i], elem1[j]] += Me_1[i, j]
        MGLOB[elem2[i], elem2[j]] += Me_2[i, j]
        MGLOB[elem3[i], elem3[j]] += Me_3[i, j]

Mk = MGLOB[np.ix_(free_dofs, free_dofs)]

eigenvalues = eig(Kk, Mk)[0]
real_eigenvalues = np.sort(eigenvalues[np.isreal(eigenvalues)].real)

alpha = np.sqrt(real_eigenvalues)
f = alpha / (2 * pi)

print("\nNatural frequencies with additional lumped mass matrix [rad/s]:")
print(np.array_str(alpha, precision=5))
print("Natural frequencies with additional lumped mass matrix [Hz]:")
print(np.array_str(f, precision=5))

print("\nConcentrated mass matrix used:")
print(M_concentrated)