import numpy.linalg as LA
import numpy as np

class Linear:
	A = np.array([[1, 2], [3, 4]])

	B = np.array([[5, 6], [7, 8]])
	
	print("A:", A)
	print("B:", B)
	
	print("matrix mult or dot product:", np.dot(A, B))
	print("OR YOU CAN DO THIS:", A @ B)

	print("transpose A:", A.T)
	print("transpose B:", B.T)
	
	I = np.eye(3)
	print("Identity matrix:", I)
	I2 = np.eye(10)
	print("Identity matrix 2:", I2)
	
	inv_A = np.linalg.inv(A)
	print("Inverse of A:", inv_A)
	print("dot of A and inv_A:", A @ inv_A)
	
	det_A = np.linalg.det(A)
	print("determinant of A:", det_A)
	
	# x = np.linalg.solve(A, 3)
	# print("What is this:?", x)
	
	vals, vecs = np.linalg.eig(A)
	
	print("I have no idea what these are of A:")
	print("vals:", vals)
	print("vecs:", vecs)


	print("---------\nPRACTICE1")
	A = np.array([[2, 1], [5, 3]])
	print("A:", A)
	
	A_inv = np.linalg.inv(A)
	print("Inverse of A:", A_inv)
	# Verify that A * A_inv is the identity matrix:
	print("A * A_inv = \n", np.round(A @ A_inv, 2))
	
	det = np.linalg.det(A)
	print("Determinant of A:", det)
	
	Sys_A = np.array([[3, 2], [1, -1]])
	print("Sys_A:", Sys_A)
	Sys_B = np.array([5, 1])
	print("Sys_B:", Sys_B)
	sol = np.linalg.solve(Sys_A, Sys_B)
	print("Solving System of equations:", sol)

	Eigen_A = np.array([[4, -2], [1, 1]])
	
	# Compute eigenvalues and eigenvectors
	vals, vecs = np.linalg.eig(Eigen_A)
	print("Vals:", vals)
	print("Vecs:", vecs)
	
	v = vecs[: 0]
	scale = vals[0]

	print("av:\n", Eigen_A * v)
	print("scalev:\n", scale * v)	
