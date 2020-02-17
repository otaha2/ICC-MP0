import numpy as np

if __name__ == "__main__":
	# zero array size 9x6
	a = np.zeros((9,6))

	# Create block I
	a[0][1:5] = 1
	a[1][1:5] = 1
	a[2][2:4] = 1
	a[3][2:4] = 1
	a[4][2:4] = 1
	a[5][2:4] = 1
	a[6][2:4] = 1
	a[7][1:5] = 1
	a[8][1:5] = 1

	# Create b
	tmp = np.zeros(6)

	b = np.insert(a, 1, tmp)
	b = np.insert(b, len(b), tmp)
	b = b.reshape((11,6))

	# Create c
	c = np.arange(1, 67, 1).reshape(11, 6)

	# Element wise multiplication
	d = np.multiply(b, c)

	# Grab non zero elements
	e = d[np.nonzero(d)]

	# Normalize
	max, min = e.max(), e.min()

	f = (e - min) / (max - min)

	index = np.abs(f - 0.25).argmin()
	closest_val = f[index]

	print("A: \n", a)
	print("B: \n", b)
	print("C: \n", c)
	print("D: \n", d)
	print("E: \n", e)
	print("F: \n", f)
	print("Result: ", closest_val)