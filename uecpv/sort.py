import numpy as np
import pandas as pd

#der_zi_kl = pd.read_csv("der_zi_kl.csv")
#der_zi_ce = pd.read_csv("der_zi_ce.csv")
der_zi_kl = pd.read_csv("der_zi_kl_sorted.csv")
der_zi_ce = pd.read_csv("der_zi_ce_sorted.csv")

der_zi_kl=np.array(der_zi_kl)
der_zi_ce=np.array(der_zi_ce)
print(np.shape(der_zi_kl))

x=der_zi_kl.shape[1]*0.97
z=der_zi_ce.shape[1]*0.039
print(x)
print(z)
x=der_zi_kl[1, 9030]
z=der_zi_ce[1, 330]
print(x)
print(z)
"""der_zi_kl=np.sort(der_zi_kl)
der_zi_ce=np.sort(der_zi_ce)

np.savetxt('der_zi_kl_sorted.csv', der_zi_kl, delimiter=',', fmt='%f')
np.savetxt('der_zi_ce_sorted.csv', der_zi_ce, delimiter=',', fmt='%f')"""