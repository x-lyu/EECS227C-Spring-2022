import numpy as np
import math
import matplotlib.pyplot as plt

A = np.empty([5, 10, 10])
B = np.empty([5, 10])

for k in range(1, 6) :
	for i in range(1, 11) :
		B[k - 1][i - 1] = math.exp(i / k) * math.sin(i * k)
	for i in range(1, 11):
		for j in range(i + 1, 11) :
			A[k - 1][i - 1][j - 1] = math.exp(i / j) * math.cos(i * j) * math.sin(k)
	for i in range(1, 11) :
		for j in range(1, i) :
			A[k - 1, i - 1, j - 1] = A[k - 1, j - 1, i - 1]
	for i in range(1, 11):
		A[k - 1, i - 1, i - 1] = math.fabs(math.sin(k)) * i / 10
		for j in range(1, 11):
			if (i != j) :
				A[k - 1, i - 1, i - 1] += math.fabs(A[k - 1, i - 1, j - 1])

def eval(x):
	ans = -0
	for k in range(1, 6):
		tmp = x.T.dot(A[k - 1]).dot(x) - B[k - 1].dot(x);
		if (k == 1 or tmp > ans) :
			ans = tmp
	return ans

def gradient(x):
	ans = 0
	d = 0
	for k in range(1, 6) :
		tmp = x.T.dot(A[k - 1]).dot(x) - B[k - 1].dot(x)
		if (k == 1 or tmp > ans) :
			ans = tmp
			d = 2 * A[k - 1].dot(x) - B[k - 1]

	ell2 = 0
	for i in range(0, 10) :
		ell2 = ell2 + d[i] * d[i]

	#return d / math.sqrt(ell2)					FOR QUESTION (b)
	return d / ell2;

x = np.ones(10)

print(eval(x))

T = int(1e5)
C = 0.05

moni = []
cur = 2

while True :
	moni.append(cur)
	if (cur > T) : break
	cur = max(cur + 1, int(cur * 1.1))

best = eval(x)
cur = 0
X = []
Y = []

empirical_min = -0.842

for t in range(1, T + 1) :
	if (t == moni[cur]) :
		X.append(t)
		Y.append(best)
		cur = cur + 1
	x = x - gradient(x) * (eval(x) - empirical_min)
	# x = x - gradient(x) * C / math.sqrt(t) 		FOR QUESTION (b)
	best = min(best, eval(x))

print(best)

for i in range(0, cur) :
	Y[i] = Y[i] - best

plt.loglog(X, Y)
plt.savefig("./out.jpg")
plt.show()