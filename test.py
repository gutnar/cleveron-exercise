#%%
import numpy as np

order = 2
size = 5

coeffs = []
coeffs.append([1, 1, 2])
coeffs.append([1.5, 0.5, 2.2])

np.mean(coeffs, axis=0)

#%%
np.array([1, 2, 3])[np.array([1])]


#%%
import numpy as np

a = np.array([1, 1, 2, 3, 4, 4, 5, 9])
a

#%%
hist = np.histogram(a, 10, (0, 10))[0]
hist

#%%
hist[a]

#%%
