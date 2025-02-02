import os
os.system("git pull")
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm
import json

# interface
import tkinter as tk # module de base
from tkinter import ttk # mettre ça beau

class Plaque:
    def __init__(self, dimensions=(0.117, 0.061), epaisseur=0.001, resolution_x=0.001, resolution_y=0.001, resolution_t=None, T_plaque=25, T_ambiante=23, densite=2699, cap_calorifique=900, conduc_thermique=237, coef_convection=20, puissance = 1.5):
        self.dim = dimensions
        self.e = epaisseur
        self.dx = resolution_x
        self.dy = resolution_y
        self.T_amb = T_ambiante + 273.15
        self.T_plaque = T_plaque + 273.15
        self.rho = densite
        self.cp = cap_calorifique
        self.k = conduc_thermique
        self.h = coef_convection
        self.grille = self.T_plaque*np.ones((int(self.dim[0]/self.dx), int(self.dim[1]/self.dx))) #Mettre un dy à qqpart ici
        self.points_chauffants = []
        self.alpha = self.k/(self.rho*self.cp)
        self.dt = min(self.dx**2/(4*self.alpha), self.dy**2/(4*self.alpha)) if resolution_t == None else resolution_t
        self.P = puissance # En [W]
        self.actuateur = np.ones((int(0.015/self.dx), int(0.015/self.dx))) # Grosseur de l'actuateur de 15x15 mm^2
        # Diviser le 1.5W sur tous les éléments de la matrice ou mettre direct 1.5 partout?
        # self.position_actuateur = (self.dim[])


    def show(self):
        plt.imshow(self.grille, cmap="gnuplot", origin = "lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))
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
        "Section puissance "


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
        self.grille = new_grille
        # # CF
        # self.grille[0, :] = self.T_plaque    # Bord haut LE BORD DU HAUT SAFFICHE EN BAS ET VICE VERSA
        # self.grille[-1, :] = self.T_plaque  # Bord bas
        # self.grille[:, 0] = self.T_plaque   # Bord gauche
        # self.grille[:, -1] = self.T_plaque

        return self.grille
    
class Interface:
    def __init__(
            self,
            dimensions=(0.117, 0.061),
            epaisseur=0.001,
            resolution_x=0.001,
            resolution_y=0.001,
            resolution_t=None,
            T_plaque=25,
            T_ambiante=23,
            densite=2699,
            cap_calorifique=900,
            conduc_thermique=237,
            coef_convection=20,
            puissance = 1.5
            ):
        self.inter = tk.Tk()
        self.frame = ttk.Frame(self.inter, padding=10)
        self.frame.grid()
        self.inter.title('Contrôle de la simulation Python')

        # Initier toutes les entrées 
        self.dimx_var=tk.IntVar()
        self.dimy_var=tk.IntVar()
        self.e_var=tk.IntVar()
        self.dx_var=tk.IntVar()
        self.dy_var=tk.IntVar()
        self.dt_var=tk.IntVar()
        self.rho_var=tk.IntVar()
        self.cp_var=tk.IntVar()
        self.k_var=tk.IntVar()
        self.h_var=tk.IntVar()
        self.T_plaque_var=tk.IntVar()
        self.T_amb_var=tk.IntVar()
        self.T_dep_var=tk.IntVar()
        self.T_posx_var=tk.IntVar()
        self.T_posy_var=tk.IntVar()
        self.P_var=tk.IntVar()

        # Initier toutes les variables
        self.dim = dimensions
        self.e = epaisseur
        self.dx = resolution_x
        self.dy = resolution_y
        self.dt = resolution_t ## pas utilisé dans class Plaque
        self.rho = densite
        self.cp = cap_calorifique
        self.k = conduc_thermique
        self.h = coef_convection
        self.T_plaque = T_plaque + 273.15
        self.T_amb = T_ambiante + 273.15
        self.T_depo = 0 # points chauffants??
        self.T_pos = 0 # points chauffants??
        self.P = puissance

        # Go go main interface
        self.main()
        
    def main(self):
        # Température initiale de la plaque
        ttk.Label(self.frame, text="Température initiale de la plaque [K]").grid(column=0, row=0)
        T_plaque_entry = ttk.Entry(self.frame, textvariable = self.T_plaque_var)
        T_plaque_entry.grid(column=1, row=0)
        T_plaque_entry.insert(0, self.T_plaque)

        # Température ambiante
        ttk.Label(self.frame, text="Température ambiante [K]").grid(column=0, row=1)
        T_amb_entry = ttk.Entry(self.frame, textvariable = self.T_amb_var)
        T_amb_entry.grid(column=1, row=1)
        T_amb_entry.insert(0, self.T_amb)

        # Coefficient de convection
        ttk.Label(self.frame, text="Coefficient de convection [??]").grid(column=0, row=2)
        h_entry = ttk.Entry(self.frame, textvariable = self.h_var)
        h_entry.grid(column=1, row=2)
        h_entry.insert(0, self.h)

        # Puissance appliquée
        ttk.Label(self.frame, text="Puissance appliquée [W]").grid(column=0, row=3)
        P_entry = ttk.Entry(self.frame, textvariable = self.P_var)
        P_entry.grid(column=1, row=3)
        P_entry.insert(0, self.P)

        # Boutons pour autres fenêtres
        ttk.Button(self.inter,text = 'Variables de la plaque', command = self.plaque).grid(column=0, row=4)
        ttk.Button(self.inter,text = 'Résolutions de la simulation', command = self.resolution).grid(column=0, row=5)
        ttk.Button(self.inter,text = 'Température déposée', command = self.T_dep).grid(column=0, row=6)

        # finish
        ttk.Button(self.inter,text = 'OK', command = self.submit).grid(column=2, row=0)
        self.inter.mainloop()
        
    def plaque(self):
        self.plaque_frame = tk.Toplevel()
        self.plaque_frame.title('Variables de la plaque')
        ttk.Button(self.plaque_frame,text = 'OK', command = self.submit_plaque).grid(column=2, row=0)

        # Dimensions de la plaque
        ttk.Label(self.plaque_frame, text="Longueur en x de la plaque").grid(column=0, row=0)
        dimx_entry = ttk.Entry(self.plaque_frame, textvariable = self.dimx_var)
        dimx_entry.grid(column=1, row=0)
        dimx_entry.insert(0, self.dim[0])
        ttk.Label(self.plaque_frame, text="Longueur en y de la plaque").grid(column=0, row=1)
        dimy_entry = ttk.Entry(self.plaque_frame, textvariable = self.dimy_var)
        dimy_entry.grid(column=1, row=1)
        dimy_entry.insert(0, self.dim[1])

        # Épaisseur de la plaque
        ttk.Label(self.plaque_frame, text="Épaisseur de la plaque").grid(column=0, row=2)
        e_entry = ttk.Entry(self.plaque_frame, textvariable = self.e_var)
        e_entry.grid(column=1, row=2)
        e_entry.insert(0, self.e)

        # Bouton pour autre fenêtre
        ttk.Button(self.plaque_frame,text = 'Paramètres du matériau de la plaque', command = self.mat).grid(column=0, row=3)

    def mat(self):
        self.mat_frame = tk.Toplevel()
        self.mat_frame.title('Paramètres du matériau de la plaque')
        ttk.Button(self.mat_frame,text = 'OK', command = self.submit_mat).grid(column=2, row=0)

        # Densité du matériau
        ttk.Label(self.mat_frame, text="Densité du matériau [kg / m^3]").grid(column=0, row=0)
        rho_entry = ttk.Entry(self.mat_frame, textvariable = self.rho_var)
        rho_entry.grid(column=1, row=0)
        rho_entry.insert(0, self.rho)

        # Capacité calorifique du matériau
        ttk.Label(self.mat_frame, text="Capacité calorifique du matériau [J / kg.K]").grid(column=0, row=1)
        cp_entry = ttk.Entry(self.mat_frame, textvariable = self.cp_var)
        cp_entry.grid(column=1, row=1)
        cp_entry.insert(0, self.cp)

        # Conductivité thermique du matériau
        ttk.Label(self.mat_frame, text="Conductivité thermique du matériau [W / m.K]").grid(column=0, row=2)
        k_entry = ttk.Entry(self.mat_frame, textvariable = self.k_var)
        k_entry.grid(column=1, row=2)
        k_entry.insert(0, self.k)

    def resolution(self):
        self.reso_frame = tk.Toplevel()
        self.reso_frame.title('Résolutions de la simulation de la plaque')
        ttk.Button(self.reso_frame,text = 'OK', command = self.submit_reso).grid(column=2, row=0)

        # Résolutions de longueur
        ttk.Label(self.reso_frame, text="En x").grid(column=0, row=0)
        ttk.Entry(self.reso_frame, textvariable = self.dx_var).grid(column=1, row=0)
        ttk.Label(self.reso_frame, text="En y").grid(column=0, row=1)
        ttk.Entry(self.reso_frame, textvariable = self.dy_var).grid(column=1, row=1)

        # Résolution de temps
        ttk.Label(self.reso_frame, text="Temps").grid(column=0, row=2)
        ttk.Entry(self.reso_frame, textvariable = self.dt_var).grid(column=1, row=2)

    def T_dep(self):
        self.T_dep_frame = tk.Toplevel()
        self.T_dep_frame.title('Contrôle de la température déposée')
        ttk.Button(self.T_dep_frame,text = 'OK', command = self.submit_T_dep).grid(column=2, row=0)

        # Température déposée
        ttk.Label(self.T_dep_frame, text="Température déposée [K]").grid(column=0, row=0)
        ttk.Entry(self.T_dep_frame, textvariable = self.T_dep_var).grid(column=1, row=0)

        # Position de la temp
        ttk.Label(self.T_dep_frame, text="Position en x de la température déposée").grid(column=0, row=1)
        ttk.Entry(self.T_dep_frame, textvariable = self.T_posx_var).grid(column=1, row=1)
        ttk.Label(self.T_dep_frame, text="Position en y de la température déposée").grid(column=0, row=2)
        ttk.Entry(self.T_dep_frame, textvariable = self.T_posy_var).grid(column=1, row=2)

    def submit(self):
        self.T_plaque = self.T_plaque_var.get()
        self.T_amb=self.T_amb_var.get()
        self.h=self.h_var.get()
        self.P=self.P_var.get()

        # Go go simulation plaque
        Ma_plaque = Plaque(
            dimensions=(self.dim),
            epaisseur=self.e,
            resolution_x=self.dx,
            resolution_y=self.dy,
            resolution_t=self.dt,
            T_plaque=self.T_plaque,
            T_ambiante=self.T_amb,
            densite=self.rho,
            cap_calorifique=self.cp,
            conduc_thermique=self.k,
            coef_convection=self.h,
            puissance=self.P
            )
        #Ma_plaque.show()
        self.inter.destroy() #### À FAIRE!!

    def submit_plaque(self):
        self.dim=(self.dimx_var.get(), self.dimy_var.get())
        self.e=self.e_var.get()
        self.plaque_frame.destroy()

    def submit_mat(self):
        self.rho=self.rho_var.get()
        self.cp=self.cp_var.get()
        self.k=self.k_var.get()
        self.mat_frame.destroy()
    
    def submit_reso(self):
        self.dx=self.dx_var.get()
        self.dy=self.dy_var.get()
        self.dt=self.dt_var.get()
        self.reso_frame.destroy()

    def submit_T_dep(self):
        self.T_depo=self.T_dep_var.get()
        self.T_pos=(self.T_posx_var.get(), self.T_posy_var.get())
        self.T_dep_frame.destroy()

#Ma_plaque = Plaque(T_plaque=64, T_ambiante=5, resolution_t=None)

Inter = Interface()

# Ma_plaque.deposer_T(40, (0.10, 0.04))
# Ma_plaque.deposer_T(12, (0.02, 0.02))
# Ma_plaque.iteration()
# Ma_plaque.show()
# Ma_plaque.iteration()
# Ma_plaque.show()

# for n in tqdm(range(2000)):
#     for k in range(20): # Vérifie que cette boucle tourne aussi
#         Ma_plaque.iteration()
#Ma_plaque.show()




