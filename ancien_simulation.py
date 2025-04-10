import os
# Met à jour le dépôt local avec les dernières modifications depuis le dépôt distant.
os.system("git pull")

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import pandas as pd



class Plaque:
    """
        Classe représentant une plaque chauffante utilisée pour modéliser l'évolution de la température.
    
        Attributs principaux :
        - dimensions : Tuple représentant la longueur et largeur (y, x) de la plaque (mètres)
        - epaisseur : Épaisseur de la plaque (mètres)
        - resolution_x, resolution_y : Résolution spatiale de la grille de simulation (mètres)
        - resolution_t : Résolution temporelle (secondes), calculée si non spécifiée pour éviter la divergence de la solution
        - T_plaque : Température initiale de la plaque (Celsius)
        - T_ambiante : Température ambiante (Celsius)
        - densite : Masse volumique de la plaque (kg/m³).
        - cap_calorifique : Capacité calorifique massique de la plaque (J/kg·K).
        - conduc_thermique : Conductivité thermique de la plaque (W/m·K).
        - coef_convection : Coefficient de convection (W/m²K)
        - puissance_actuateur : Puissance fournie par l'élément chauffant (Watts)
        - perturbations : Liste des perturbations thermiques appliquées sous forme (position_y, position_x), puissance, (grosseur_y, grosseur_x)
    """

    def __init__(self, dimensions=(0.116, 0.0615), epaisseur=0.00156, resolution_x=0.001, resolution_y=0.001, resolution_t=None, T_plaque=25, T_ambiante=23, densite=2700, cap_calorifique=897, conduc_thermique=167, coef_convection=12, puissance_actuateur = 1.5, perturbations = []):
    
        # Dimensions et propriétés physiques
        self.dim = dimensions  # (longueur y, largeur x)
        self.e = epaisseur  # Épaisseur de la plaque
        self.dx = resolution_x  # Résolution spatiale en x
        self.dy = resolution_y  # Résolution spatiale en y
        self.T_amb = T_ambiante + 273.15  # Conversion en Kelvin
        self.T_plaque = T_plaque + 273.15  # Conversion en Kelvin
        self.rho = densite  # Masse volumique (kg/m³)
        self.cp = cap_calorifique  # Capacité thermique massique (J/kg.K)
        self.k = conduc_thermique  # Conductivité thermique (W/m.K)
        self.h = coef_convection  # Coefficient de convection (W/m².K)
        
        # Initialisation de la grille de température (matrice remplie avec la température initiale)
        self.grille = self.T_plaque * np.ones((int(self.dim[0] / self.dy), int(self.dim[1] / self.dx))) 
        
        # Calcul du coefficient de diffusivité thermique
        self.alpha = self.k / (self.rho * self.cp)
        
        # Calcul du pas temporel optimal pour assurer la stabilité numérique
        self.dt = min(self.dx**2 / (4 * self.alpha), self.dy**2 / (4 * self.alpha)) if resolution_t is None else resolution_t
        
        # Paramètres de l'actuateur (chauffage)
        self.P_act = puissance_actuateur  # Puissance fournie par l'actuateur (W)
        self.actuateur = np.ones((int(0.015 / self.dy), int(0.015 / self.dx)))  # Taille de l'actuateur discret
        T_actuateur = (self.dt / (self.rho * self.cp)) * (self.P_act / self.actuateur.size) / (self.dx * self.dy * self.e) # Conversion de la puissance en température sur chaque élément
        self.actuateur_pos, self.T_actuateur = self.place_actuateur(T_actuateur)
        
        # Initialisation des perturbations thermiques
        self.perturbations = perturbations
        self.convertir_perturbations()
        
        # Enregistrement du temps, de la puissance déposée et des températures aux points d'intérêt (thermistances) dans une liste de listes
        self.rep_echelon = [[0], [0], [self.T_plaque], [self.T_plaque], [self.T_plaque]]

        

    def convertir_perturbations(self):
        """
        convertir_perturbations(self) -> None

        Description :
        Convertir les perturbations définies en termes de position et puissance en indices et température appliquées.

        Arguments : Aucun 
        Utilise self.perturbations 

        Retourne : None 
        Modifie self.perturbations en mettant à jour les tuples de la liste des perturbations thermiques. 
        Chaque perturbation est décrite par un tuple d'indices de position et un objet de type np.ndarray contenant la température de chaque élément.
        """
        nouvelles_perturbations = []
        for perturb in self.perturbations:
            (pos_y, pos_x), puissance, (longueur, largeur) = perturb
            iy, ix = int(pos_y/self.dy), int(pos_x/self.dx)
            longueur, largeur = int(longueur/self.dy), int(largeur/self.dx)

            if largeur < self.dx or longueur < self.dy:
                raise ValueError(
                    f"Erreur : La perturbation est trop petite pour être représentée ")

            # Répartir la puissance sur toute la zone
            T_applique = (self.dt / (self.rho * self.cp)) * (puissance / (longueur * largeur)) / (self.dx * self.dy * self.e) * np.ones((longueur, largeur))
        
            # La position spécifiée de la perturbation correspond à la position de son coin bas gauche sur la plaque
            nouvelles_perturbations.append(((iy, iy+longueur, ix, ix+largeur), T_applique))
            # nouvelles_perturbations.append(((iy-longueur//2, iy+longueur//2+1, ix-longueur//2, ix+largeur//2+1), T_applique))
    
        self.perturbations = nouvelles_perturbations

    def place_actuateur(self, T_actuateur):
        """
        place_actuateur(self, T_actuateur: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]

        Description :
        Positionne l'actuateur au centre inférieur de la plaque.

        Arguments :
        T_actuateur (np.ndarray) : Température appliquée par l'actuateur (K) sur chaque élément.

        Retourne :
        (tuple[tuple[int, int, int, int], np.ndarray]) :
        Une tuple de 4 entiers représentant la position de l’actuateur (iy_debut, iy_fin, ix_debut, ix_fin).
        La température appliquée par l’actuateur (np.ndarray).
        """
        # Le centre de l'actuateur est en y = 0.015m et x = 0.03m
        Ly, Lx = self.grille.shape
        # L'actuateur mesure y = 0.015m et x = 0.015m 
        act_dim_y, act_dim_x = self.actuateur.shape

        # Trouver le centre cible
        ix_centre = int(Lx / 2)  # Centre en x
        iy_centre = int(Ly * 1 / 8)  # Centre en y

        # Déterminer les indices de début et de fin en soustrayant la moitié de la taille de T_actuateur
        ix_debut = ix_centre - act_dim_x // 2
        ix_fin = ix_centre + act_dim_x // 2 + 1  # +1 pour inclure le dernier indice
        iy_debut = iy_centre - act_dim_y // 2
        iy_fin = iy_centre + act_dim_y // 2 + 1  # +1 pour inclure le dernier indice

        # Retourne les indices de positionnement de l'actuateur et la température  
        return (iy_debut,iy_fin,ix_debut,ix_fin), T_actuateur


    def show(self):
        """
        Affiche la répartition de la température sur la plaque.
        """
        T_celsius = self.grille - 273.15
        plt.imshow(T_celsius, cmap="inferno", origin = "lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))#plt.cm.jet
        plt.colorbar()
        plt.xlabel("Position en x (cm)")
        plt.ylabel("Position en y (cm)")
        plt.show()


    def iteration(self):
        """
        Description :
        Effectue une itération de mise à jour de la température sur la plaque en tenant compte de la conduction, de la convection et des sources de chaleur (actuateur, perturbations).

        Arguments :
        Aucun.

        Retourne :
        (np.ndarray) : Nouvelle grille de température mise à jour.
        """
        
        "Section conduction"
        # Décalage et addition de matrices pour éviter d'itérer sur chaque élément. Il s'agit d'une sorte de moyennage
        # conduction cas général
        conduction = (self.alpha * self.dt) * (
            ((np.roll(self.grille, shift=1, axis=0) + np.roll(self.grille, shift=-1, axis=0) - 2* self.grille)/self.dy**2) +  # Haut - Bas
            ((np.roll(self.grille, shift=1, axis=1) + np.roll(self.grille, shift=-1, axis=1) - 2 * self.grille)/self.dx**2)) # Gauche - Droite   
        # conduction rangée du haut
        conduction[0,:] = (self.alpha * self.dt) * (
            ((self.grille[1,:] -  self.grille[0,:])/self.dy**2) +  # Haut
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
        # Avec cette approche linéaire, la convection sur les 4 coins est prise en compte (4 surfaces exposées)

        "Section total"
        # Additionne de la température actuelle de la plaque aux contributions thermique de la conduction et de la convection
        new_grille = self.grille + conduction + convection

        "Section puissance déposée"
        self.grille = new_grille

        # Contribution thermique de l'actuateur positionné au bon endroit 
        self.grille[self.actuateur_pos[0]:self.actuateur_pos[1], self.actuateur_pos[2]:self.actuateur_pos[3]] += self.T_actuateur

        # Contribution thermique des perturbations positionné au bon endroit
        for perturb in self.perturbations:
            self.grille[perturb[0][0]:perturb[0][1], perturb[0][2]:perturb[0][3]] += perturb[1]


        "Enregistrement de la température"
        # Enregistrement de la température aux position des thermistances dans une liste de listes
        pos_thermi1 = (int(0.015/self.dy),  int(self.dim[1]/2 / self.dx)) # En y=1.5cm, x=3cm
        pos_thermi2 = (int(0.06/self.dy),  int(self.dim[1]/2 / self.dx)) # En y=6cm, x=3cm
        pos_thermi3 = (int(0.104/self.dy),  int(self.dim[1]/2 / self.dx)) # En y=(11.6-1.2)cm, x=3cm
        self.rep_echelon[0].append(self.rep_echelon[0][-1]+self.dt) # Temps d'échantillonage
        self.rep_echelon[1].append(self.P_act) # Puissance appliquée à l'actuateur
        self.rep_echelon[2].append(self.grille[pos_thermi1[0], pos_thermi1[1]]) # Température à la thermistance 1
        self.rep_echelon[3].append(self.grille[pos_thermi2[0], pos_thermi2[1]]) # Température à la thermistance 2
        self.rep_echelon[4].append(self.grille[pos_thermi3[0], pos_thermi3[1]]) # Température à la thermistance 3

        return self.grille
    

    
    def enregistre_rep_echelon(self):
        """
        Description :
        Enregistre la réponse du système sous forme de fichier CSV (output.csv). 
        Les données enregistrées incluent le temps, l'entrée du système, et les températures aux trois points d'intérêt (T1, T2, T3).

        Arguments : Aucun 
        Utilise self.rep_echelon

        Retourne : None
        Crée un fichier "output.csv" contenant les données sous forme de tableau.
        """
        df = pd.DataFrame(np.array(self.rep_echelon).T)
        df.to_csv("output.csv", index=False) # temps, entrée, T1, T2, T3


    def affiche_initial(self):
        """
        Affiche la répartition des composants sur la plaque.
    
        - La plaque est en gris
        - L'actuateur est en rouge
        - Les thermistances sont en vert
        - Les perturbations sont en bleu
        """
        size = self.grille.shape
        plaque = np.ones((*size, 3)) * 0.5  # Fond gris

        # Positions des éléments 
        # actuateur
        iy1, iy2, ix1, ix2 = self.actuateur_pos
        # thermistances
        pos_thermi1 = (int(0.015/self.dy), int(self.dim[1]/2 / self.dx)) # En y=1.5cm, x=3cm
        pos_thermi2 = (int(0.06/self.dy), int(self.dim[1]/2 / self.dx)) # En y=6cm, x=3cm
        pos_thermi3 = (int(0.104/self.dy), int(self.dim[1]/2 / self.dx)) # En y=(11.6-1.2)cm, x=3cm
        thermistances = [pos_thermi1, pos_thermi2, pos_thermi3]

        # Affectation des couleurs
        plaque[iy1:iy2,ix1:ix2] = [0, 1, 0]  # Vert pour l'actuateur
        for t in thermistances:
            plaque[t] = [0, 0, 1]  # Bleu pour les thermistances
        for p in self.perturbations:
            (iy1,iy2,ix1,ix2), T = p
            plaque[iy1:iy2,ix1:ix2] = [1, 0, 0]  # Rouge pour les perturbation 

        # Affichage avec imshow
        fig, ax = plt.subplots()
        ax.imshow(plaque, origin = "lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))

        # Ajout de la légende (patches)
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=[0, 0, 1], label='Thermistances'),
            Patch(facecolor=[0, 1, 0], label='Actuateur'),
            Patch(facecolor=[1, 0, 0], label='Perturbation(s)'),
            Patch(facecolor='gray', label='Plaque')
        ]

        ax.legend(handles=legend_elements, bbox_to_anchor=(1.85, 1))
        ax.set_xlabel("Position en x (cm)")
        ax.set_ylabel("Position en y (cm)")

        plt.show()


Ma_plaque = Plaque(T_plaque=24, T_ambiante=24, resolution_t=None, puissance_actuateur=3.6) # TUPLE (Y, X) perturbations=[((0.01,0.01),2), ((0.05,0.03),4)]
# Ma_plaque.perturbations = [((0.015+0.021-0.003, (Ma_plaque.dim[1]/2)-0.0015), 0.75, (0.006,0.003))]#, ((0.01,0.01),3, (0.01,0.01)), ((0.05,0.03),4,(0.001,0.001))
#((0.015+0.021-0.0015, 0.03-0.003), 1, (0.006,0.003)) résistance de perturbation en y = T1y+2.1cm et y3cm-0.3
Ma_plaque.convertir_perturbations()
Ma_plaque.affiche_initial()


# Ma_plaque.iteration()
# Ma_plaque.show()


"ICII"
start = time.time()
for n in tqdm(range(10000)):
    for k in range(20): 
        Ma_plaque.iteration()
        # Ma_plaque.show()
end = time.time()
print(end-start)
Ma_plaque.enregistre_rep_echelon()
Ma_plaque.show()
print(Ma_plaque.dt)
#print(Ma_plaque.grille.size)
#print(Ma_plaque.grille.shape)

