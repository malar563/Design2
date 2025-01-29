import os
os.system("git pull")
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
#changement

#double changement

class Plaque:
    def __init__(self, dimensions=(0.117, 0.061), epaisseur=0.001, resolution_x=0.0001, resolution_t=1, T_ambiante=25, densite=2699, cap_calorifique=900, conduc_thermique=237, coef_convection=20):
        self.dim = dimensions
        self.e = epaisseur
        self.dx = resolution_x
        # self.dt = resolution_t VA FAIRE DIVERGERR
        self.T_amb = T_ambiante
        self.rho = densite
        self.cp = cap_calorifique
        self.k = conduc_thermique
        self.h = coef_convection
        self.grille = self.T_amb*np.ones((int(self.dim[0]/self.dx), int(self.dim[1]/self.dx)))
        self.points_chauffants = []
        self.alpha = self.k/(self.rho*self.cp)
        self.dt = (self.dx**2)/(self.alpha*8)


    def show(self):
        plt.imshow(self.grille, cmap="gnuplot", origin = "lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))
        plt.colorbar()
        plt.show()
    
    def deposer_T(self, T_desiree=30, endroit=(0,0)):
        self.points_chauffants.append((T_desiree, endroit))
        for point in self.points_chauffants:
            self.grille[int(point[1][0]/self.dx), int(point[1][1]/self.dx)] = point[0]
        # Rajouter la possibilité d'un groupe de points chauffants?

    def iteration(self):
        """
        Trouver le potentiel dans chaque case de la matrice pour l'itération suivante.
        Le potentiel de la case à l'itération suivante correspond à la moyenne des 
        4 cases autour de la case d'intérêt.

        Args:
        chambre (numpy.ndarray): Chambre pour laquelle on cherche le potentiel à l'itération suivante.

        Returns:
        chambre_nouvelle_petite (numpy.ndarray) : Chambre contenant le potentiel de l'itération suivante.
        """

        big_grille = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        big_grille[1:-1, 1:-1] = copy.copy(self.grille)
        diffusion = (self.alpha*self.dt*(big_grille[2:, 1:-1] + big_grille[:-2, 1:-1] + big_grille[1:-1, 2:] +  big_grille[1:-1, :-2] - 4*big_grille[1:-1, 1:-1])/(self.dx**2)) 
        
        # CF
        self.grille[0, :] = 50    # Bord haut
        self.grille[-1, :] = 50   # Bord bas
        self.grille[:, 0] = 50    # Bord gauche
        self.grille[:, -1] = 50

        # # Convection (mettre *2 car rentre de 2 côtés?)
        # convection = self.h * self.dx**2 * (self.T_amb - self.grille) / (self.rho * self.cp * self.dx**2 * self.e)

        new_grille = self.grille + diffusion #+ convection
        self.grille = new_grille
        return self.grille
    




Ma_plaque = Plaque(T_ambiante=26)

Ma_plaque.deposer_T(40, (0.10, 0.04))
Ma_plaque.deposer_T(12, (0.02, 0.02))
Ma_plaque.iteration()
Ma_plaque.show()
Ma_plaque.iteration()
Ma_plaque.show()
for n in tqdm(range(2500)):
    Ma_plaque.iteration()
Ma_plaque.show()
