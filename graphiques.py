import numpy as np
import matplotlib.pyplot as plt

def load_file(degre=19):
    """return matlab, python"""
    return np.loadtxt(f"Matlab_{degre}_degre.csv", skiprows=1, delimiter=",", dtype=float), np.loadtxt(f"Python_{degre}_degre.csv", skiprows=1, delimiter=",", dtype=float)


for degre in [19,25,30]:
    matlab, python = load_file(degre)
    python[:,2:] -= 273.15
    
    ax1 = plt.subplot(111)
    plt.plot(matlab[:,0], matlab[:,2], "-", color="blue", label="Thermistance 1")
    plt.plot(matlab[:,0], matlab[:,3], "-", color="green", label="Thermistance 2")
    plt.plot(matlab[:,0], matlab[:,1], "-", color="red", label="Thermistance 3")
    plt.plot(python[:,0], python[:,4], "--", color="red")
    plt.plot(python[:,0], python[:,3], "--", color="green")
    plt.plot(python[:,0], python[:,2], "--", color="blue")
    plt.legend(fontsize=15)
    plt.xlabel(r"Temps [s]", fontsize=16)
    plt.ylabel(r"Température [$^\circ$C]", fontsize=16)
    ax1.xaxis.set_tick_params(labelsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.set_xlim((-50,1100))
    plt.tight_layout()
    plt.savefig(f"graph_temperature_{degre}.pdf")
    plt.show()