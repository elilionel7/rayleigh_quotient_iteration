import matplotlib.pyplot as plt

def plot_eigenfunction(u, gdata):
    u_reshaped = u.reshape(gdata.x1.shape)
    plt.contourf(gdata.x1, gdata.x2, u_reshaped, levels=50)
    plt.colorbar()
    plt.title("Computed Eigenfunction")
    plt.show()

