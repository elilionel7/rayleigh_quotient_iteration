# SEEM_Chebyshev_master/print_results.py

import matplotlib.pyplot as plt

def make_graph_qr(results, filename):
    p = results['p']
    pts = results['pts']
    l2_norms = results['L2']

    for l_idx, l in enumerate(p):
        plt.figure(figsize=(8, 6))
        for eigen_idx in range(l2_norms.shape[2]):
            plt.plot(
                pts,
                l2_norms[l_idx, :, eigen_idx],
                marker='o',
                label=f'Eigenvalue {eigen_idx + 1}'
            )
        plt.xlabel('Number of Points')
        plt.ylabel('L² Norm')
        plt.title(f'L² Norm vs. Number of Points for p={l}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'L2_norm_p{l}.pdf')
        plt.close()