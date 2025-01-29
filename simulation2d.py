import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Class definition for Plaque
class Plaque:
    def __init__(self, dimensions=(100, 1), epaisseur=0.001, resolution_x=0.001, resolution_t=1, T_ambiante=25, densite=2699, cap_calorifique=900, conduc_thermique=237, coef_convection=1):
        self.dim = dimensions  # dimensions devrait être un tuple (largeur, hauteur)
        self.e = epaisseur
        self.dx = resolution_x
        self.dt = resolution_t
        self.T_amb = T_ambiante
        self.rho = densite
        self.cp = cap_calorifique
        self.k = conduc_thermique
        self.h = coef_convection
        # Grille 100x1
        self.grille = self.T_amb * np.ones(self.dim)  # dimensions est maintenant un tuple (100, 1)
        self.points_chauffants = []

    def show(self, nb_iterations=5):
        plt.ion()
        for i in range(nb_iterations):
            self.iteration()

            # Affichage de la grille avec une color map
            plt.imshow(self.grille, cmap="gnuplot", origin="lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))
            plt.colorbar()
            plt.title(f"Temps: {i * self.dt:.2f} secondes")
            plt.pause(0.01)
            plt.clf()

        plt.show()

    def deposer_T(self, T_desiree=30, endroit=(0, 0)):
        self.points_chauffants.append((T_desiree, endroit))
        # Mise à jour de la température aux endroits spécifiés
        for point in self.points_chauffants:
            x_index = int(point[1][0] / self.dx)
            if x_index < len(self.grille):  # S'assurer que l'index est dans les limites
                self.grille[x_index] = point[0]

    def iteration(self):
        """
        Calcule la température dans chaque case de la grille pour l'itération suivante.
        """

        # matriceN
        mat_N = 2*np.ones(self.dim)
        mat_trois = 3*np.ones(self.dim)
        mat = np.vstack([mat_trois[1], mat_N[2:], mat_trois[-1]])
        mat[:, 0] = mat_trois[:, 0]
        mat[:, -1] = mat_trois[:, -1]
        print(mat)

        # Création d'une matrice batard
        T_betex = np.zeros(self.grille.shape[1])
        
        # Création de la matrice de base
        T_centre = np.zeros(self.grille.shape)
        T_centre = self.grille

        # Création de la matrice décalée à gauche
        T_gauche = np.vstack([T_betex, self.grille[0:-1]])

        # Création de la matrice décalée à droite
        T_droite = np.vstack([self.grille[1:], T_betex])

        # Calcul de la température pour la prochaine itération
        nouvelle_grille = (self.k * self.dt * (T_droite + T_gauche - 2 * T_centre) / (self.rho * self.cp * self.dx ** 2)) + T_centre

        # Calcul de la température pour la prochaine itération sur les côtés
        cotes_grille_gauche = T_centre[0] + ((self.dt)/(self.rho * self.cp) * self.h * (self.T_amb - T_centre[0]) * 2 / self.e)
        cotes_grille_gauche = cotes_grille_gauche + (self.k * self.dt * (T_droite[0] - 2 * T_centre[0]) / (self.rho * self.cp * self.dx ** 2)) + T_centre[0]
        cotes_grille_droite = T_centre[-1] + ((self.dt)/(self.rho * self.cp) * self.h * (self.T_amb - T_centre[0]) * 2 / self.e)
        cotes_grille_droite = cotes_grille_droite + (self.k * self.dt * (T_gauche[-1] - 2 * T_centre[-1]) / (self.rho * self.cp * self.dx ** 2)) + T_centre[-1]
        
        # Remplacement pour les côtés
        remplacement_grille = nouvelle_grille[1:-1]
        nouvelle_grille = np.vstack([cotes_grille_gauche, remplacement_grille, cotes_grille_droite])

        # Mise à jour de la grille
        self.grille = nouvelle_grille

        return self.grille


# Instanciation et exécution
Ma_plaque = Plaque(dimensions=(10, 5), resolution_t=0.01, T_ambiante=20)
Ma_plaque.iteration()
Ma_plaque.show()
