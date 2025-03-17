# SEEM_Chebyshev_master/print_results.py

def make_chart_qr(results):
    p = results['p']
    pts = results['pts']
    eigenvalues = results['eigenvalues']
    iterations = results['iterations']
    l2_norms = results['L2']

    for l_idx, l in enumerate(p):
        print(f"\nResults for p = {l}:")
        print("Number of Points | Eigenvalue 1 | Iterations 1 | L2 Norm 1 | Eigenvalue 2 | Iterations 2 | L2 Norm 2 | Eigenvalue 3 | Iterations 3 | L2 Norm 3 |")
        for k_idx, pt in enumerate(pts):
            print(f"{pt} | ", end='')
            for eigen_idx in range(eigenvalues.shape[2]):
                ev = eigenvalues[l_idx, k_idx, eigen_idx]
                it = iterations[l_idx, k_idx, eigen_idx]
                l2 = l2_norms[l_idx, k_idx, eigen_idx]
                print(f"{ev:.6f} | {it} | {l2:.6e} | ", end='')
            print()