import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
a = [10,15,12,8,12,14,9,8,11,12,10,15,12,8,12,14,9,8,11,12]
b = [10,12,15,13,12,14,15,16,17,16,15,17,18,16,19,20,18,19,20,21,22,20,21,23,24,22,23,25,24,26,27,25,28,29]
ll = {'index':[i for i in range(len(b))],'shuzi':b}
lll = pd.DataFrame(ll)
yici = lll.diff(1)
erci= yici.diff(1)
fig1 = plt.figure()
plt.plot(b)
plt.title('Original data')
fig2 = plt.figure()
plt.plot(yici)
plt.title('Once difference')
fig3 = plt.figure()
plt.plot(erci)
plt.title('Twice difference')
plt.show()