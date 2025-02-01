import os
os.system("git pull")
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm


class Plaque:
    def __init__(self, dimensions=(0.117, 0.061), epaisseur=0.001, resolution_x=0.001, resolution_y=0.001, resolution_t=None, T_plaque=25, T_ambiante=23, densite=2699, cap_calorifique=900, conduc_thermique=237, coef_convection=20, puissance_actuateur = 1.5):
        self.dim = dimensions # tuple (y, x)
        self.e = epaisseur
        self.dx = resolution_x
        self.dy = resolution_y
        self.T_amb = T_ambiante + 273.15
        self.T_plaque = T_plaque + 273.15
        self.rho = densite
        self.cp = cap_calorifique
        self.k = conduc_thermique
        self.h = coef_convection
        self.grille = self.T_plaque*np.ones((int(self.dim[0]/self.dy), int(self.dim[1]/self.dx))) #Mettre un dy à qqpart ici
        self.points_chauffants = []
        self.alpha = self.k/(self.rho*self.cp)
        self.dt = min(self.dx**2/(4*self.alpha), self.dy**2/(4*self.alpha)) if resolution_t == None else resolution_t
        self.P = puissance_actuateur # En [W]
        # C'est légal?
        self.actuateur = np.ones((int(0.015/self.dy), int(0.015/self.dx))) # Grosseur de l'actuateur de 15x15 mm^2 #Mettre un dy à qqpart ici
        T_actuateur = (self.dt/(self.rho * self.cp)) * (self.P/self.actuateur.size)/(self.dx*self.dy*self.e) # Diviser le 1.5W sur tous les éléments de la matrice ou mettre direct 1.5 partout?
        self.actuateur_pos, self.T_actuateur = self.place_actuateur(T_actuateur)
    

    def place_actuateur(self, T_actuateur):
        # position_actuateur = (0.03/self.dx, 0.015/self.dy) # format (x, y) ici
        Ly, Lx = self.grille.shape
        act_dim_y, act_dim_x = self.actuateur.shape

        # Trouver le centre cible
        ix_centre = int(Lx / 2)  # Centre en x
        iy_centre = int(Ly * 1 / 8)  # Centre en y

        # Déterminer les indices de début et de fin en soustrayant la moitié de la taille de T_actuateur
        ix_debut = ix_centre - act_dim_x // 2
        ix_fin = ix_centre + act_dim_x // 2 + 1  # +1 pour inclure le dernier indice
        iy_debut = iy_centre - act_dim_y // 2
        iy_fin = iy_centre + act_dim_y // 2 + 1  # +1 pour inclure le dernier indice

        # self.grille[iy_debut:iy_fin, ix_debut:ix_fin] += T_actuateur # + self.grille[ix_debut:ix_fin, iy_debut:iy_debut]
        return (iy_debut,iy_fin,ix_debut,ix_fin), T_actuateur


    def show(self):
        plt.imshow(self.grille, cmap="gnuplot", origin = "lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0])) # LEQUEL EST CORRECT??????
        # plt.imshow(self.grille, cmap="gnuplot", origin = "lower", extent=(0, 100*self.dim[0], 0, 100*self.dim[1]))
        plt.colorbar()
        plt.show()
    
    # def deposer_T(self, T_desiree=30, endroit=(0,0)):
    #     self.points_chauffants.append((T_desiree, endroit))
    #     for point in self.points_chauffants:
    #         self.grille[int(point[1][0]/self.dx), int(point[1][1]/self.dx)] = point[0]
    #     # Rajouter la possibilité d'un groupe de points chauffants?

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
        # big_grille = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # big_grille[1:-1, 1:-1] = copy.copy(self.grille)
        # diffusion = (self.alpha*self.dt*(big_grille[2:, 1:-1] + big_grille[:-2, 1:-1] + big_grille[1:-1, 2:] +  big_grille[1:-1, :-2] - 4*big_grille[1:-1, 1:-1])/(self.dx**2)) 
        
        # big_grille = np.zeros((self.grille.shape[0]+2, self.grille.shape[1]+2))
        # big_grille[1:-1, 1:-1] = copy.copy(self.grille) EST-CE QUE CE SERAIT DES 1 À LA PLACE DES 2?
        
        "Section conduction"
        # conduction cas général
        conduction = (self.alpha * self.dt) * (
            ((np.roll(self.grille, shift=1, axis=0) + np.roll(self.grille, shift=-1, axis=0) - 2* self.grille)/self.dy**2) +  # Haut - Bas
            ((np.roll(self.grille, shift=1, axis=1) + np.roll(self.grille, shift=-1, axis=1) - 2 * self.grille)/self.dx**2)) # Gauche - Droite   
        # conduction rangée du haut
        conduction[0,:] = (self.alpha * self.dt) * (
            ((self.grille[1,:] -  self.grille[0,:])/self.dy**2) +  # Bas
            ((np.roll(self.grille[0,:], shift=1) + np.roll(self.grille[0,:], shift=-1) - 2 * self.grille[0,:])/self.dx**2)) # Gauche - Droite
        # conduction rangée du bas
        conduction[-1,:] = (self.alpha * self.dt) * (
            ((self.grille[-2,:] -  self.grille[-1,:])/self.dy**2) +  # Bas
            ((np.roll(self.grille[0,:], shift=1) + np.roll(self.grille[0,:], shift=-1) - 2 * self.grille[0,:])/self.dx**2)) # Gauche - Droite
        # conduction côté gauche
        conduction[:,0] = (self.alpha * self.dt) * (
            ((np.roll(self.grille[:,0], shift=1) + np.roll(self.grille[:,0], shift=-1) - 2 * self.grille[:,0])/self.dy**2) + # Haut - Bas
            ((self.grille[:,1] -  self.grille[:,0])/self.dx**2))  # Gauche
        # conduction côté droit
        conduction[:,-1] = (self.alpha * self.dt) * (
            ((np.roll(self.grille[:,-1], shift=1) + np.roll(self.grille[:,-1], shift=-1) - 2 * self.grille[:,-1])/self.dy**2) + # Haut - Bas
            ((self.grille[:,-2] -  self.grille[:,-1])/self.dx**2))  # Droit
        # conduction coin supérieur gauche
        conduction[0,0] = (self.alpha * self.dt) * (
            ((self.grille[1,0] - self.grille[0,0])/self.dy**2) + # Bas
            ((self.grille[0,1] -  self.grille[0,0])/self.dx**2))  # Droit
        # conduction coin supérieur droit
        conduction[0,-1] = (self.alpha * self.dt) * (
            ((self.grille[1,-1] - self.grille[0,-1])/self.dy**2) + # Bas
            ((self.grille[0,-2] -  self.grille[0,-1])/self.dx**2))  # Gauche
        # conduction coin inférieur gauche
        conduction[-1,0] = (self.alpha * self.dt) * (
            ((self.grille[-2,0] - self.grille[-1,0])/self.dy**2) + # Haut
            ((self.grille[-1,1] -  self.grille[-1,0])/self.dx**2))  # Droit
        # conduction coin inférieur droit
        conduction[-1,-1] = (self.alpha * self.dt) * (
            ((self.grille[-2,-1] - self.grille[-1,-1])/self.dy**2) + # Haut
            ((self.grille[-1,-2] -  self.grille[-1,-1])/self.dx**2))  # Gauche
        
        "Section convection"
        # convection cas général (2 surfaces exposées)
        convection = 2 * self.dt * self.h * (self.T_amb - self.grille) / (self.rho * self.cp * self.e) # aire_top/volume : dx*dy s'annule laissant e
        # convection haut et bas (cas général + 1 surface)
        convection[0,:] =  convection[0,:] + (self.dt * self.h * (self.T_amb - self.grille[0,:]) / (self.rho * self.cp * self.dy)) # aire_side/volume : dx*e s'annule laissant dy
        convection[-1,:] =  convection[-1,:] + (self.dt * self.h * (self.T_amb - self.grille[-1,:]) / (self.rho * self.cp * self.dy)) # aire_side/volume : dx*e s'annule laissant dy
        # convection gauche et droite (cas général + 1 surface)
        convection[:,0] =  convection[:,0] + (self.dt * self.h * (self.T_amb - self.grille[:,0]) / (self.rho * self.cp * self.dx)) # aire_side/volume : dy*e s'annule laissant dx
        convection[:,-1] =  convection[:,-1] + (self.dt * self.h * (self.T_amb - self.grille[:,-1]) / (self.rho * self.cp * self.dx)) # aire_side/volume : dy*e s'annule laissant dx
        # Si je ne me trompe pas, la convection sur les coins a déjà été prise en compte...

        "Section total"
        new_grille = self.grille + conduction + convection

        "Section puissance "
        
        # print(T_actuateur.shape)
        self.grille = new_grille
        self.grille[self.actuateur_pos[0]:self.actuateur_pos[1], self.actuateur_pos[2]:self.actuateur_pos[3]] += self.T_actuateur
        # reste à placer cette matrice de température au centre de la grille

        # # CF
        # self.grille[0, :] = self.T_plaque    # Bord haut LE BORD DU HAUT SAFFICHE EN BAS ET VICE VERSA
        # self.grille[-1, :] = self.T_plaque  # Bord bas
        # self.grille[:, 0] = self.T_plaque   # Bord gauche
        # self.grille[:, -1] = self.T_plaque
        
        return self.grille
    


Ma_plaque = Plaque(T_plaque=64, T_ambiante=5, resolution_t=None, puissance_actuateur=1.5)

# Ma_plaque.deposer_T(40, (0.10, 0.04))
# Ma_plaque.deposer_T(12, (0.02, 0.02))
# Ma_plaque.iteration()
# Ma_plaque.show()
# Ma_plaque.iteration()
# Ma_plaque.show()


"ICII"
start = time.time()
for n in tqdm(range(2000)):
    for k in range(20): # Vérifie que cette boucle tourne aussi
        Ma_plaque.iteration()
end = time.time()
print(end-start)
Ma_plaque.show()
print(Ma_plaque.grille.size)
print(Ma_plaque.grille.shape)



