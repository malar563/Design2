import os
os.system("git pull")
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
#changement

#double changement

class Plaque:
    def __init__(self, dimensions=(0.117, 0.061), epaisseur=0.001, resolution_x=0.0001, resolution_t=1, T_ambiante=25, densite=2699, cap_calorifique=900, conduc_thermique=237, coef_convection=0):
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
        self.dt = (0.5*self.dx**2)/self.alpha


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
        # # Matrice non décalée
        # T_centre = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # T_centre[1:-1, 1:-1] = self.grille

        # # Matrice décalée vers le haut
        # T_haut = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # T_haut[0:-2, 1:-1] = self.grille

        # # Matrice décalée vers le bas
        # T_bas = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # T_bas[2:, 1:-1] = self.grille

        # # Matrice décalée vers la gauche
        # T_gauche = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # T_gauche[1:-1, 0:-2] = self.grille

        # # Matrice décalée vers la gauche
        # T_droite = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # T_droite[1:-1, 2:] = self.grille
 
        # # Le potentiel à l'itération suivante est calculé selon l'expression trouvée à la question 1a)
        # nouvelle_grille = (self.alpha*self.dt*(T_droite + T_gauche + T_bas + T_haut - 4*T_centre)/(self.dx**2)) + T_centre

        # # Restreindre la chambre à sa grandeur d'origine
        # self.grille = nouvelle_grille[1:-1, 1:-1]

        "# CODE OPTIMISÉ"
        big_grille = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        big_grille[1:-1, 1:-1] = copy.copy(self.grille)
        new_grille = (self.alpha*self.dt*(big_grille[2:, 1:-1] + big_grille[:-2, 1:-1] + big_grille[1:-1, 2:] +  big_grille[1:-1, :-2] - 4*big_grille[1:-1, 1:-1])/(self.dx**2)) + big_grille[1:-1, 1:-1]
        self.grille = new_grille
        return self.grille



Ma_plaque = Plaque(resolution_t=0.01,T_ambiante=297)

Ma_plaque.deposer_T(307, (0.10, 0.04))
Ma_plaque.deposer_T(293, (0.02, 0.02))
Ma_plaque.iteration()
Ma_plaque.show()
Ma_plaque.iteration()
Ma_plaque.show()

