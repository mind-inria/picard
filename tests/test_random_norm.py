import numpy as np
import numbers
from picard._tools import check_random_state

random_state = check_random_state(5489)
n_components = 11
w_init = np.asarray(random_state.normal(size=(n_components,
                    n_components)), dtype='float64')
# print full matrix with 10 decimals, all rows and columns
np.set_printoptions(precision=10)

# print all rows and columns of w_init as a MATLAB matrix
print('[')
for i in range(n_components):
    for j in range(n_components):
        print(w_init[i, j], end=' ')
    print(';')
print(']')


