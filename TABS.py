# pour lire un document JSON
import os
import json

# pour faire des documents nommés selon l'heure actuelle
from datetime import datetime

# pour faire rouler l'interface
import tkinter as tk
from tkinter import ttk

# pour faire jouer la simulation
import mega_simulation

# pour afficher un graphique
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Interface:
    def __init__(self):
        """
        Description :
        Initialise l'interface, les variables d'entrées et lit un JSON

        Arguments : -

        Retourne : -
        """
        # Initialise l'interface
        self.inter = tk.Tk()
        self.inter.title('Contrôle de la simulation Python')

        # Permet les onglets dans l'interface
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[5, 5])

        # Lire un JSON si possible, sinon lire celui de base
        self.lire_json()
        
        # Initialisation des variables depuis JSON ou valeurs par défaut
        self.dim = self.data_lu.get("dimensions", [11.6,6.15]) #[y,x]
        self.e = self.data_lu.get("epaisseur", 0.156) 
        self.dx = self.data_lu.get("resolution_x", 0.15) 
        self.dy = self.data_lu.get("resolution_y", 0.1) 
        self.dt = self.data_lu.get("resolution_t", None)
        self.T_simul = self.data_lu.get("temps_simulation", 600) # [s]
        self.rho = self.data_lu.get("densite", 2700) 
        self.cp = self.data_lu.get("cap_calorifique", 897.0)
        self.k = self.data_lu.get("conduc_thermique", 167.0) 
        self.h = self.data_lu.get("coef_convection", 12) 
        self.T_plaque = self.data_lu.get("T_plaque", 25.0)
        self.T_amb = self.data_lu.get("T_ambiante", 25.0)
        self.P = self.data_lu.get("puissance_actuateur", 1.5)
        self.R_depo = self.data_lu.get("puissance_R", 0)
        self.T_depo = self.data_lu.get("puissance_ajoutee", 0)
        self.T_pos = self.data_lu.get("position_puissance", [0,0]) # [y,x]
        self.T_lon = self.data_lu.get("longueur_puissance", [0.5,0.5]) # [y,x]

        # Initier variables avec calculs
        self.alpha = self.k/(self.rho*self.cp)
        if self.dt is None:
            self.dt = min((self.dx/100)**2/(4*self.alpha), (self.dy/100)**2/(4*self.alpha))  # À regarder!!

        # Initier toutes les entrées de l'interface
        self.variables = {key: tk.DoubleVar(value=val) for key, val in {
            "dimy": self.dim[0], "dimx": self.dim[1], "e": self.e, "dx": self.dx,
            "dy": self.dy, "dt": self.dt, "rho": self.rho, "cp": self.cp, 
            "k": self.k, "h": self.h, "T_plaque": self.T_plaque,
            "T_amb": self.T_amb, "P": self.P, "R_depo": self.R_depo,
            "T_depo": self.T_depo, "T_posy": self.T_pos[0], "T_posx": self.T_pos[1],
            "T_lony": self.T_lon[0], "T_lonx": self.T_lon[1],
            "T_simul": self.T_simul
        }.items()}

        # Initialisation l'interface!
        self.main()
    

    def lire_json(self):
        """
        Description :
        Charge les données depuis le fichier JSON le plus récent ou utilise les valeurs par défaut

        Arguments : -

        Retourne : -
        """
        # Trouver les fichiers JSON
        json_files = sorted(
            (f for f in os.listdir() if f.endswith('.json')),
            key=os.path.getmtime,
            reverse=True
        )

        # Si aucun fichier n'est trouvé, utiliser les valeurs par défaut
        if not json_files:
            print("Aucun fichier JSON trouvé. Valeurs par défaut utilisées.")
            self.data_lu = self.json_de_base()
            return

        # Charger le fichier JSON le plus récent
        self.nom_json = json_files[0]
        try:
            with open(self.nom_json, "r", encoding="utf-8") as file:
                self.data_lu = json.load(file)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Erreur lors de la lecture du fichier {self.nom_json}: {e}\nValeurs par défaut utilisées.")
            self.data_lu = self.json_de_base()

    def json_de_base(self):
        """
        Description :
        Retourne le JSON de base

        Arguments : -

        Retourne : 
        Le JSON de base avec les valeurs par défaut des variables
        """
        return {
            "dimensions": [11.6,6.15], # [y,x]
            "epaisseur": 0.156,
            "resolution_x": 0.15,
            "resolution_y": 0.1,
            "resolution_t": None,
            "temps_simulation": 600, # [s]
            "densite": 2700,
            "cap_calorifique": 897,
            "conduc_thermique": 167,
            "coef_convection": 12,
            "T_plaque": 25,
            "T_ambiante": 25,
            "puissance_actuateur": 1.5,
            "puissance_R": 0,
            "puissance_ajoutee": 0,
            "position_puissance": [0,0], # [y,x]
            "longueur_puissance": [0.5,0.5] # [y,x]
        }


    def entry(self, parent, texte, var, row):
        """
        Description :
        Créer une case nommée dans l'interface permettant
            de changer la valeur d'une variable 

        Arguments :
        parent: la fenêtre de l'interface où le texte
            et la case d'entrée doivent s'afficher
        texte: le nom de la variable modifiée et ses unités
        var: le nom de la variable selon la classe
        row: l'hauteur du texte et de la case d'entrée

        Retourne : 
        Un Entry Widget TkInter, permettant
            de lier une valeur écrire par l'utilisateur
            dans l'interface à une variable 
        """
        ttk.Label(parent, text=texte).grid(column=0, row=row, padx=5, pady=5, sticky="w") # Label
        entry = ttk.Entry(parent, textvariable=self.variables[var]) # Entry
        entry.grid(column=1, row=row, padx=5, pady=5, sticky="ew") # Placement de l'entry
        return entry


    def main(self):
        """
        Description :
        Créer l'interface, initialise les entrées permettant de
            changer les valeurs des variables et affiche l'interface

        Arguments : -

        Retourne :
        self.inter : L'interface apparaît dans une nouvelle fenêtre
        """
        # Initialisation des onglets
        self.tabs = ttk.Notebook(self.inter)
        self.tabs.grid(column=0, row=0, rowspan=2, columnspan=2, sticky="nsew")

        # Initialisation des différents onglets
        self.controle_frame() # Contrôle de base
        self.plaque() # Variables de la plaque
        self.mat() # Paramètres du matériau de la plaque
        self.resolution() # Résolutions de la simulation de la plaque
        self.T_dep() # Contrôle de la puissance déposée

        # Initialisation des boutons OK et Graphique
        self.etat_OK = ttk.Button(self.inter,text = 'OK', command = self.no_graphique)
        self.etat_OK.grid(column=0, row=3, pady=5, sticky="ew")
        ttk.Button(self.inter,text = 'Graphique', command = self.yes_graphique).grid(column=1, row=3, pady=5, sticky="ew")
        
        # Initialisation du boutton Arrêt
        self.etat_arret = ttk.Button(self.inter,text = 'Arrêt', command = self.arret)
        self.etat_arret.grid(column=0, row=3, pady=5, sticky="ew")
        self.etat_arret.grid_remove()

        # Initialisation de la barre de progression
        tk.Label(self.inter, text="Progression :").grid(column=0, row=4, padx=10, pady=5, sticky="ew")
        self.progres = ttk.Progressbar(self.inter, orient="horizontal", length=100, mode="determinate")
        self.progres.grid(column=1, row=4, padx=10, pady=5, sticky="ew")

        # Initialise et cache les messages d'erreurs
        tk.Label(self.inter, text="Erreur : ").grid(column=0, row=5, padx=10, pady=5, sticky="ew")
        
        # Si les variables ne sont pas numériques
        self.var_num = tk.Label(self.inter, text="Variables entrées ne sont pas numériques")
        self.var_num.grid(column=1, row=5, padx=10, pady=5, sticky="ew")
        self.var_num.grid_remove()

        # Si les dimensions ne sont pas réalistes
        self.var_dim = tk.Label(self.inter, text="Dimensions entrées ne sont pas plus grandes que zéro")
        self.var_dim.grid(column=1, row=5, padx=10, pady=5, sticky="ew")
        self.var_dim.grid_remove()

        # Si les résolutions ne sont pas réalistes
        self.var_res = tk.Label(self.inter, text="Résolutions entrées ne sont pas plus grandes que zéro")
        self.var_res.grid(column=1, row=5, padx=10, pady=5, sticky="ew")
        self.var_res.grid_remove()

        # Si les paramètres du matériau ne sont pas réalistes
        self.var_par = tk.Label(self.inter, text="Les paramètres du matériau ne sont pas plus grands que zéro")
        self.var_par.grid(column=1, row=5, padx=10, pady=5, sticky="ew")
        self.var_par.grid_remove()

        # Roulons l'interface!
        self.inter.mainloop()
        

    def controle_frame(self):
        """
        Description :
        Créer le premier onglet de l'interface, permettant
            le contrôle des paramètres de base de la simulation

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Contrôle de base
        self.frame = ttk.Frame(self.tabs, padding=10)
        self.frame.grid()
        self.tabs.add(self.frame, text="Contrôle de base")

        # Initialisation des entrées 
        self.entry(self.frame, "Température initiale de la plaque [°C]", "T_plaque", 0) # Température initiale de la plaque
        self.entry(self.frame, "Température ambiante [°C]", "T_amb", 1) # Température ambiante
        self.entry(self.frame, "Puissance appliquée [W]", "P", 2) # Puissance appliquée
        self.entry(self.frame, "Durée de la simulation [s]", "T_simul", 3) # Durée de la simulation


    def plaque(self):
        """
        Description :
        Créer le deuxième onglet de l'interface, permettant
            le contrôle des paramètres de la plaque

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Variables de la plaque
        self.plaque_frame = ttk.Frame(self.tabs, padding=10)
        self.plaque_frame.grid()
        self.tabs.add(self.plaque_frame, text="Variables de la plaque")

        # Initialisation des entrées
        self.entry(self.plaque_frame, "Longueur en x de la plaque [cm]", "dimx", 0) # Dimensions de la plaque
        self.entry(self.plaque_frame, "Longueur en y de la plaque [cm]", "dimy", 1)
        self.entry(self.plaque_frame, "Épaisseur de la plaque [cm]", "e", 2) # Épaisseur de la plaque


    def mat(self):
        """
        Description :
        Créer le troisième onglet de l'interface, permettant
            le contrôle des paramètres du matériau de la plaque

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Paramètres du matériau de la plaque
        self.mat_frame = ttk.Frame(self.tabs, padding=10)
        self.mat_frame.grid()
        self.tabs.add(self.mat_frame, text="Paramètres du matériau de la plaque")

        # Initialisation des entrées
        self.entry(self.mat_frame, "Densité du matériau [kg / m³]", "rho", 0) # Densité du matériau
        self.entry(self.mat_frame, "Capacité calorifique du matériau [J / kg.K]", "cp", 1) # Capacité calorifique du matériau
        self.entry(self.mat_frame, "Conductivité thermique du matériau [W / m.K]", "k", 2) # Conductivité thermique du matériau
        self.entry(self.mat_frame, "Coefficient de convection du matériau [W / m².K]", "h", 3) # Coefficient de convection


    def resolution(self):
        """
        Description :
        Créer le quatrième onglet de l'interface, permettant
            le contrôle des paramètres de résolution de la plaque

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Résolution de la simulation de la plaque
        self.reso_frame = ttk.Frame(self.tabs, padding=10)
        self.reso_frame.grid()
        self.tabs.add(self.reso_frame, text="Résolution de la simulation de la plaque")

        # Initialisation des entrées
        self.entry(self.reso_frame, "Résolution en x [cm]", "dx", 0) # Résolutions de longueur
        self.entry(self.reso_frame, "Résolution en y [cm]", "dy", 1)
        self.entry(self.reso_frame, "Résolution en temps [s]", "dt", 2) # Résolution de temps


    def T_dep(self):
        """
        Description :
        Créer le cinquième onglet de l'interface, permettant
            le contrôle des paramètres des perturbations

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Contrôle des perturbations
        self.T_dep_frame = ttk.Frame(self.tabs, padding=10)
        self.T_dep_frame.grid()
        self.tabs.add(self.T_dep_frame, text="Contrôle des perturbations")

        # Initialisation des entrées
        self.entry(self.T_dep_frame, "Puissance déposée avec la résistance [W]", "R_depo", 0) # Résistance de perturbation
        self.entry(self.T_dep_frame, "Puissance déposée [W]", "T_depo", 1) # Puissance de la perturbation
        self.entry(self.T_dep_frame, "Position en x de la puissance déposée [cm]", "T_posx", 2) # Position de la perturbation
        self.entry(self.T_dep_frame, "Position en y de la puissance déposée [cm]", "T_posy", 3)
        self.entry(self.T_dep_frame, "Longueur en x de la puissance déposée [cm]", "T_lonx", 4) # Dimensions de la perturbation
        self.entry(self.T_dep_frame, "Longueur en y de la puissance déposée [cm]", "T_lony", 5)

        # Initialisation du graphique permettant de voir la position des perturbations
        ttk.Button(self.T_dep_frame, text="Voir la plaque", command=self.graphique_plaque).grid(row=6, column=0, columnspan=2, pady=5)


    def sauvegarder_json(self):
        """
        Description :
        Sauvegarde un nouveau json nommé selon la date et l'heure

        Arguments : -

        Retourne : -
        """
        # Avoir la date dans le format 'dd.mm'
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        new_nom = f'{current_date}.json'
        
        # Regarde si fichier existe déjà, fait nouveau fichier
        while os.path.exists(new_nom):
            new_nom = f'{current_date}.json'
        
        # Sauvegarder
        with open(new_nom, 'w') as f:
            json.dump(self.data_fait, f, indent=4)


    def submit(self):
        """
        Description :
        Initialise la simulation avec les variables modifiées par l'utilisateur
            dans l'interface et sauvegarde un nouveau JSON avec celles-ci

        Arguments : -

        Retourne : -
        """
        try:
            # Sauvegarder les variables selon les modifications de l'utilisateur dans l'interface
            self.dim= [self.variables["dimy"].get(), self.variables["dimx"].get()] # [y,x]
            self.e=self.variables["e"].get()
            self.T_plaque = self.variables["T_plaque"].get()
            self.T_amb=self.variables["T_amb"].get()
            self.h=self.variables["h"].get()
            self.P=self.variables["P"].get()
            self.T_simul=self.variables["T_simul"].get()
            self.rho=self.variables["rho"].get()
            self.cp=self.variables["cp"].get()
            self.k=self.variables["k"].get()
            self.dx=self.variables["dx"].get()
            self.dy=self.variables["dy"].get()
            self.dt=self.variables["dt"].get()
            self.R_depo=self.variables["R_depo"].get()
            self.T_depo=self.variables["T_depo"].get()
            self.T_pos=[self.variables["T_posy"].get(), self.variables["T_posx"].get()] # [y,x]
            self.T_lon=[self.variables["T_lony"].get(), self.variables["T_lonx"].get()] # [y,x]

            if self.dim[0] <= 0 or self.dim[1] <= 0 or self.e <= 0:
                # Si les dimensions entrées sont égales à zéro ou négatives
                self.var_dim.grid()

                # Retourne la présence d'une erreur
                return False
            
            if self.dx <= 0 or self.dy <= 0 or self.dt <= 0:
                # Si les résolutions entrées sont égales à zéro ou négatives
                self.var_res.grid()

                # Retourne la présence d'une erreur
                return False
            
            if self.h <= 0 or self.rho <= 0 or self.cp <= 0 or self.k <= 0:
                # Si les paramètres du matériau entrés sont égales à zéro ou négatives
                self.var_par.grid()

                # Retourne la présence d'une erreur
                return False
        except:
            # Si les variables entrées ne sont pas numériques
            self.var_num.grid()

            # Retourne la présence d'une erreur
            return False
        else:
            # Pas d'erreur
            self.var_num.grid_remove()
            self.var_dim.grid_remove()

            # Quantité d'itérations
            self.saut = round(( self.T_simul / (10 * self.dt))**(1/2))
            self.N = 10 * self.saut

            # Sauvegarde des données mises à jour dans le JSON
            self.data_fait = {
                "dimensions": [self.dim[0],self.dim[1]], # [y,x]
                "epaisseur": self.e,
                "resolution_x": self.dx,
                "resolution_y": self.dy,
                "resolution_t": self.dt,
                "temps_simulation": self.T_simul, # [s]
                "densite": self.rho,
                "cap_calorifique": self.cp,
                "conduc_thermique": self.k,
                "coef_convection": self.h,
                "T_plaque": self.T_plaque,
                "T_ambiante": self.T_amb,
                "puissance_actuateur": self.P,
                "puissance_R": self.R_depo,
                "puissance_ajoutee": self.T_depo,
                "position_puissance": [self.T_pos[0], self.T_pos[1]], # [y,x]
                "longueur_puissance": [self.T_lon[0], self.T_lon[1]] # [y,x]
            }
            self.sauvegarder_json()

            # Initialise la simulation
            self.Ma_plaque = mega_simulation.Plaque(
                dimensions=(self.dim[0], self.dim[1]), # (y,x)
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
                puissance_actuateur=self.P,
                perturbations=[
                    [((0.015+0.021-0.003), ((self.dim[1]/200)-0.0015)), self.R_depo, (0.6/100, 0.3/100)], # Résistance de perturbation
                    [(self.T_pos[0]/100, self.T_pos[1]/100), self.T_depo, (self.T_lon[0]/100, self.T_lon[1]/100)] # Perturbation additionnelle
                    ],
                T_simul=self.T_simul
                )


    def no_graphique(self):
        """
        Description :
        Initialise la simulation et la barre de progression. 
        Fait rouler la simulation et enregistrer un CSV avec les données obtenues

        Arguments : -

        Retourne : -
        """
        # Initialise la simulation
        if self.submit() == False:
            return
        
        # Voir le boutton arrêt
        self.etat_OK.grid_remove()
        self.etat_arret.grid()

        # Définit le maximum de la barre de progression
        self.progres.configure(maximum=self.T_simul)

        # Roule l'interface selon la quantité d'itérations
        for n in range(self.N):
            self.progres["value"] = self.Ma_plaque.rep_echelon[0][-1] # Avance la barre de progression
            self.inter.update_idletasks()
            if self.Ma_plaque.rep_echelon[0][-1] > self.T_simul: # Si le temps de simulation est dépassé
                self.etat_arret.grid_remove() # Voir le boutton OK
                self.etat_OK.grid()
                break # Arrêt de la simulation
            else:
                for k in range(self.saut):
                    self.Ma_plaque.iteration()

        # Enregistre un CSV            
        self.Ma_plaque.enregistre_rep_echelon()


    def yes_graphique(self):
        """
        Description :
        Initialise la simulation et la barre de progression. 
        Fait rouler la simulation et enregistrer un CSV avec les données obtenues

        Arguments : -

        Retourne :
        Une fenêtre des graphiques de la température de la plaque selon le temps
        """
        # Initialise la simulation
        self.submit()

        # Voir le boutton arrêt
        self.etat_OK.grid_remove()
        self.etat_arret.grid()

        # Définit le maximum de la barre de progression
        self.progres.configure(maximum=self.T_simul)

        # Roule l'interface selon la quantité d'itérations
        for n in range(self.N):
            self.progres["value"] = self.Ma_plaque.rep_echelon[0][-1] # Avance la barre de progression
            self.inter.update_idletasks()
            if self.Ma_plaque.rep_echelon[0][-1] > self.T_simul: # Si le temps de simulation est dépassé
                self.etat_arret.grid_remove() # Voir le boutton OK
                self.etat_OK.grid()
                break # Arrêt de la simulation
            else:
                self.Ma_plaque.show() # Réitère les graphiques
                for k in range(self.saut):
                    self.Ma_plaque.iteration()
        self.Ma_plaque.enregistre_rep_echelon()


    def arret(self):
        # Voir le boutton OK
        self.etat_arret.grid_remove()
        self.etat_OK.grid()
    

    def graphique_plaque(self):
        """
        Description :
        Affiche un graphique affichant les thermistances,
            l'actuateur et les perturbations sur la plaque

        Arguments : -

        Retourne :
        Un graphique affichant les thermistances,
            l'actuateur et les perturbations sur la plaque
        """
        # Initialise la simulation
        self.submit()

        # Supprimer l'ancien canvas s'il existe
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().grid_forget() 

        size = self.Ma_plaque.grille.shape
        plaque = np.ones((*size, 3)) * 0.5  # Fond gris

        # Positions des éléments 
        iy1, iy2, ix1, ix2 = self.Ma_plaque.actuateur_pos # Actuateur
        thermistances = [self.Ma_plaque.pos_thermi1, self.Ma_plaque.pos_thermi2, self.Ma_plaque.pos_thermi3] # Thermistances

        # Affectation des couleurs
        plaque[iy1:iy2,ix1:ix2] = [0, 1, 0]  # Vert pour l'actuateur
        for t in thermistances:
            plaque[t] = [0, 0, 1]  # Bleu pour les thermistances

        # Couleur les perturbations rouge seulement si elles existent
        if self.Ma_plaque.nouv_pertur is True:
            for p in self.Ma_plaque.perturbations: # Si il existe plusieurs perturbations
                (iy1,iy2,ix1,ix2), T = p
                plaque[iy1:iy2,ix1:ix2] = [1, 0, 0]  # Rouge pour les perturbation 
        else:
            (iy1,iy2,ix1,ix2), T = self.Ma_plaque.perturbations[0] # Si il existe seulement la résistance de perturbation
            plaque[iy1:iy2,ix1:ix2] = [1, 0, 0]

        # Affichage
        figy = plt.figure()
        axy = figy.gca()  # Récupère les axes actuels ou les crée si nécessaire
        axy.imshow(plaque, origin="lower", extent=(0, 100*self.dim[1], 0, 100*self.dim[0]))

        # Création de la légende
        legend_elements = [
            Patch(facecolor=[0, 0, 1], label='Thermistances'),
            Patch(facecolor=[0, 1, 0], label='Actuateur'),
            Patch(facecolor=[1, 0, 0], label='Perturbation(s)'),
            Patch(facecolor='gray', label='Plaque')
        ]

        # Affichage de la légende et des titres des axes
        axy.legend(handles=legend_elements, bbox_to_anchor=(1.85, 1))
        axy.set_xlabel("Position en x (cm)")
        axy.set_ylabel("Position en y (cm)")

        # Intégration à l'interface
        self.canvas = FigureCanvasTkAgg(figy, master=self.T_dep_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=2, sticky="nsew", rowspan=6)

        # Le graphique est bien placé dans l'interface
        self.T_dep_frame.grid_rowconfigure(0, weight=1)
        self.T_dep_frame.grid_columnconfigure(0, weight=1)



Inter= Interface()