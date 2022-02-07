import networkx as nx
import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def plot_scatterplot(x_data, y_data, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data)
    ax.set_ylabel(y_label, fontsize=15)
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_title(title)
    plt.show()


G = nx.read_gml('models/SNetT.gml')

"""NO START 2"""

L = nx.laplacian_matrix(G).toarray()

eigenvalues = linalg.eigvals(L)  # TODO: change to .eigenvalsh because we know the matrix is symmetric
eigenvalues.sort()
enumerator = np.array(range(1, len(eigenvalues) + 1))
df_eig = pd.DataFrame(list(zip(enumerator, eigenvalues)))

# write eigenvalue table
df_eig30 = df_eig[:30]
df_eig30.columns = ['k', 'lambda_k']
df_eig30 = df_eig30.astype({'k': 'int32', 'lambda_k': 'float'})
print(df_eig30)

plot_scatterplot(enumerator, eigenvalues, r'$k$', r'$\lambda_k$', 'Ceo spektar graf laplasijana')

"""NO START 3"""

df_eig_30 = df_eig[:10]

plot_scatterplot(df_eig_30.iloc[:, 0], df_eig_30.iloc[:, 1], r'$k$', r'$\lambda_k$',
                 'prvih 10 sopstvenih vrednosti graf laplasijana')

df_eig_30 = df_eig[10:20]

plot_scatterplot(df_eig_30.iloc[:, 0], df_eig_30.iloc[:, 1], r'$k$', r'$\lambda_k$',
                 '10-20 sopstvenih vrednosti graf laplasijana')

df_eig_30 = df_eig[20:30]

plot_scatterplot(df_eig_30.iloc[:, 0], df_eig_30.iloc[:, 1], r'$k$', r'$\lambda_k$',
                 '20-30 sopstvenih vrednosti graf laplasijana')

df_eig_30 = df_eig[30:40]

plot_scatterplot(df_eig_30.iloc[:, 0], df_eig_30.iloc[:, 1], r'$k$', r'$\lambda_k$',
                 '30-40 sopstvenih vrednosti graf laplasijana')
