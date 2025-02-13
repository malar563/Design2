import os
os.system("git pull")
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
#changement


class Plaque:
    def __init__(self, dimensions=(0.116, 0.06), epaisseur=0.001, resolution_x=0.0015, resolution_y=0.001, resolution_t=None, T_plaque=25, T_ambiante=23, densite=2699, cap_calorifique=900, conduc_thermique=237, coef_convection=20, puissance_actuateur = 1.5, perturbations = []):
        self.dim = dimensions # tuple (y, x)
        # TEMPS TOTAL
        # NOMBRE DITÉRATION TEMPORELLE
        self.e = epaisseur
        self.dx = resolution_x
        self.dy = resolution_y
        self.T_amb = T_ambiante + 273.15
        self.T_plaque = T_plaque + 273.15
        self.rho = densite
        self.cp = cap_calorifique
        self.k = conduc_thermique
        self.h = coef_convection
        self.grille = self.T_plaque*np.ones((int(self.dim[0]/self.dy), int(self.dim[1]/self.dx))) 
        self.alpha = self.k/(self.rho*self.cp)
        self.dt = min(self.dx**2/(4*self.alpha), self.dy**2/(4*self.alpha)) if resolution_t == None else resolution_t # 8 ALPHA PLUTÔT QUE 4 ALPHA
        self.P_act = puissance_actuateur # En [W]
        self.actuateur = np.ones((int(0.015/self.dy), int(0.015/self.dx))) # Grosseur de l'actuateur de 15x15 mm^2 #Mettre un dy à qqpart ici
        T_actuateur = (self.dt/(self.rho * self.cp)) * (self.P_act/self.actuateur.size)/(self.dx*self.dy*self.e) # Diviser le 1.5W sur tous les éléments de la matrice ou mettre direct 1.5 partout?
        self.actuateur_pos, self.T_actuateur = self.place_actuateur(T_actuateur)
        self.perturbations = perturbations
        self.convertir_perturbations()
        self.rep_echelon = [[0],[0],[self.T_plaque], [self.T_plaque], [self.T_plaque]]

    def convertir_perturbations(self):
        for i in range(len(self.perturbations)):
            self.perturbations[i] = (int(self.perturbations[i][0][0]/self.dy), int(self.perturbations[i][0][1]/self.dx)), (self.dt/(self.rho * self.cp)) * self.perturbations[i][1]/(self.dx*self.dy*self.e)


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



    # def show(self):
    #     
    #     if not hasattr(self, 'fig') or self.fig is None:
    #         # Graphique 3D
    #         self.temp = []
    #         self.fig = plt.figure()
    #         self.ax = self.fig.add_subplot(121, projection='3d')
    #         self.ax2 = self.fig.add_subplot(122)
    #         self.x = np.linspace(0, 100 * self.dim[0], self.grille.shape[0])  
    #         self.y = np.linspace(0, 100 * self.dim[1], self.grille.shape[1])  
    #         self.x, self.y = np.meshgrid(self.y, self.x) 
    #         self.surface = self.ax.plot_surface(self.x, self.y, self.grille, cmap="plasma", edgecolor='k')  
    #         self.ax.set_xlabel('x (cm)')
    #         self.ax.set_ylabel('y (cm)')
    #         self.ax.set_zlabel('Température (K)')
    #         self.ax.set_title("Température de la plaque après simulation")
    #         # self.fig.colorbar(self.surface, ax=self.ax)
            
    #         # Graphique 2D
    #         self.t = [0] 
    #         self.temp1 = [self.grille[int(50 * self.dim[1]) , int(10 * self.dim[0])]]
    #         self.temp2 = [self.grille[int(50 * self.dim[1]) , int(50 * self.dim[0])]]
    #         self.temp3 = [self.grille[int(50 * self.dim[1]) , int(90 * self.dim[0])]]
    #         self.ax2.plot(self.t, self.temp1, color='b')
    #         self.ax2.plot(self.t, self.temp2, color='g')
    #         self.ax2.plot(self.t, self.temp3, color='r')
    #         self.ax2.set_xlabel('t (s)')
    #         self.ax2.set_ylabel('T (K)')
    #         self.ax2.set_title("Température des thermistances en fonction du temps ")

    #     else:
    #         # Graphique 3D
    #         self.surface.remove()  
    #         self.surface = self.ax.plot_surface(self.x, self.y, self.grille, cmap="plasma", edgecolor='k') 

    #         # Graphique 2D
    #         self.t.append(self.t[-1] + 85*self.dt)
    #         self.temp1.append(self.grille[int(50 * self.dim[1]) , int(10 * self.dim[0])])
    #         self.temp2.append(self.grille[int(50 * self.dim[1]), int(50 * self.dim[0])])
    #         self.temp3.append(self.grille[int(50 * self.dim[1]) , int(90 * self.dim[0])])
    #         self.ax2.clear() 
    #         self.ax2.plot(self.t, self.temp1, color='b')
    #         self.ax2.plot(self.t, self.temp2, color='g')
    #         self.ax2.plot(self.t, self.temp3, color='r')
    #         self.ax2.set_xlabel('t (s)')
    #         self.ax2.set_ylabel('T (K)')
    #         self.ax2.set_title("Température des thermistances en fonction du temps ")

    #     self.fig.canvas.flush_events()
    #     plt.pause(0.001)
    def show(self):
        plt.imshow(self.grille, cmap="inferno", origin = "lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))#plt.cm.jet
        plt.colorbar()
        plt.show()


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
            ((np.roll(self.grille[-1,:], shift=1) + np.roll(self.grille[-1,:], shift=-1) - 2 * self.grille[-1,:])/self.dx**2)) # Gauche - Droite
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



        "Section total"
        new_grille = self.grille + conduction + convection

        "Section puissance "
        self.grille = new_grille
        self.grille[self.actuateur_pos[0]:self.actuateur_pos[1], self.actuateur_pos[2]:self.actuateur_pos[3]] += self.T_actuateur

        # Perturbations
        for perturb in self.perturbations:
            self.grille[perturb[0][0], perturb[0][1]] += perturb[1]
        
        # Position d'enregistrement donc des thermistances
        pos_thermi1 = (int(0.015/self.dy), int(0.03/self.dx)) # En (x=3, y=1,5) cm
        pos_thermi2 = (int(0.06/self.dy), int(0.03/self.dx)) # En (x=3, y=6) cm
        pos_thermi3 = (int(0.104/self.dy), int(0.03/self.dx)) # En (x=3, y=(11,6-1,2)) cm
        self.rep_echelon[0].append(self.rep_echelon[0][-1]+self.dt)
        self.rep_echelon[1].append(self.P_act)
        self.rep_echelon[2].append(self.grille[pos_thermi1[0], pos_thermi1[1]])
        self.rep_echelon[3].append(self.grille[pos_thermi2[0], pos_thermi2[1]])
        self.rep_echelon[4].append(self.grille[pos_thermi3[0], pos_thermi2[1]])

        return self.grille
    

    
    def enregistre_rep_echelon(self):
        df = pd.DataFrame(np.array(self.rep_echelon).T)
        df.to_csv("output.csv", index=False) # temps, entrée, T1, T2, T3


#Ma_plaque = Plaque(T_plaque=35, T_ambiante=21, resolution_t=None, puissance_actuateur=1) # TUPLE (Y, X)

# Ma_plaque.deposer_T(40, (0.10, 0.04))
# Ma_plaque.deposer_T(12, (0.02, 0.02))
# Ma_plaque.iteration()
# Ma_plaque.show()
# Ma_plaque.iteration()
# Ma_plaque.show()


"ICII"
# start = time.time()
# for n in tqdm(range(10000)):
#     for k in range(50): 
#         Ma_plaque.iteration()
#         # Ma_plaque.show()
# end = time.time()
# print(end-start)
# Ma_plaque.enregistre_rep_echelon()
# Ma_plaque.show()
# print(Ma_plaque.grille.size)
# print(Ma_plaque.grille.shape)



