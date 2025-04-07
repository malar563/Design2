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
    """
        Classe représentant une interface permettant de contrôler une plaque chauffante utilisée pour modéliser l'évolution de la température.
    
        Attributs principaux :
        - dimensions : Tuple représentant la longueur et largeur (y, x) de la plaque (centimètres)
        - epaisseur : Épaisseur de la plaque (centimètres)
        - resolution_x, resolution_y : Résolution spatiale de la grille de simulation (centimètres)
        - resolution_t : Résolution temporelle (secondes), calculée si non spécifiée pour éviter la divergence de la solution
        - t_simul : Durée de la simulation (s)
        - T_plaque : Température initiale de la plaque (Celsius)
        - T_ambiante : Température ambiante (Celsius)
        - densite : Masse volumique de la plaque (kg/m³).
        - cap_calorifique : Capacité calorifique massique de la plaque (J/kg·K).
        - conduc_thermique : Conductivité thermique de la plaque (W/m·K).
        - coef_convection : Coefficient de convection (W/m²K)
        - puissance_actuateur : Puissance fournie par l'élément chauffant (Watts)
        - position_actuateur : Tuple correspondant au centre de l'actuateur (y, x) cm
        - grosseur_actuateur : Tuple (y, x) cm
        - temps_actuateur : Tuple du temps d'application de l'actuateur tel que (temps_allumage, temps_fermeture)
        - perturbations : Liste des perturbations thermiques appliquées sous forme [(position_y, position_x), puissance, (grosseur_y, grosseur_x), (temps_allumage, temps_fermeture)] en cm
        - position_thermistances : Liste des positions des thermistances tel que [(y,x),(y,x),(y,x)] pour T1, T2, T3 en cm 
    """

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

        # Fixe la grosseur de la fenêtre
        self.screen_width = self.inter.winfo_screenwidth()
        self.screen_height = self.inter.winfo_screenheight()
        self.inter.geometry(f"{int(self.screen_width)}x{int(self.screen_height)}") # Fixe la taille de la fenêtre à la taille de l'écran

        # Permet les onglets dans l'interface
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[5, 5])

        # Lire un JSON si possible, sinon lire celui de base
        self.lire_json()

        # Initialisation des variables depuis JSON ou valeurs par défaut
        self.dim = self.data_lu.get("dimensions", [11.6,6.15]) #[y,x] doit être plus grand que zéro
        self.e = self.data_lu.get("epaisseur", 0.156) # doit être plus grand que zéro
        self.dx = self.data_lu.get("resolution_x", 0.15) # doit être plus grand que zéro
        self.dy = self.data_lu.get("resolution_y", 0.1) # doit être plus grand que zéro
        self.dt = self.data_lu.get("resolution_t", None) # doit être plus grand que zéro
        self.t_simul = self.data_lu.get("temps_simulation", 600) # [s] doit être plus grand que zéro
        self.rho = self.data_lu.get("densite", 2700) # doit être plus grand que zéro
        self.cp = self.data_lu.get("cap_calorifique", 897.0) # doit être plus grand que zéro
        self.k = self.data_lu.get("conduc_thermique", 167.0)  # doit être plus grand que zéro
        self.h = self.data_lu.get("coef_convection", 12)  # doit être plus grand que zéro
        self.T_plaque = self.data_lu.get("T_plaque", 25.0)
        self.T_amb = self.data_lu.get("T_ambiante", 25.0)
        self.P = self.data_lu.get("puissance_actuateur", 1.5) # doit être entre -5 et 5 W
        self.R_depo = self.data_lu.get("puissance_R", 0)
        self.R_delais = self.data_lu.get("delais_R", 0)
        self.R_fin = self.data_lu.get("fin_R", 10)
        self.N_perturb = self.data_lu.get("N_perturb", 0)

        # Initialisation des variables par rapport aux perturbations
        self.P_add = self.data_lu.get("puissance_add", [1])
        self.pos_add = self.data_lu.get("position_add", [[1,1]]) # [[y,x], [y,x]...]
        self.dim_add = self.data_lu.get("dimensions_add", [[1,1]]) # [[y,x], [y,x]...]
        self.temps_add = self.data_lu.get("temps_add", [[0, 10]])
        
        # Quantité de perturbations additionnelles
        self.N_perturb = 0

        # Bouton d'arrêt
        self.var_arret = False

        # Initier variables avec calculs
        self.alpha = self.k/(self.rho*self.cp)
        if self.dt is None:
            self.dt = min((self.dx/100)**2/(4*self.alpha), (self.dy/100)**2/(4*self.alpha))  # À regarder!!

        # Initier toutes les entrées de l'interface
        self.variables = {key: tk.DoubleVar(value=val) for key, val in {
            "dimy": self.dim[0], "dimx": self.dim[1], "e": self.e,
            "dx": self.dx, "dy": self.dy, "dt": self.dt,
            "t_simul": self.t_simul,
            "rho": self.rho, "cp": self.cp, 
            "k": self.k, "h": self.h,
            "T_plaque": self.T_plaque, "T_amb": self.T_amb,
            "P": self.P, "R_depo": self.R_depo,
            "R_delais": self.R_delais, "R_fin": self.R_fin,
            "N_perturb": self.N_perturb
        }.items()}

        # Initier les entrées par rapport aux perturbations
        for i in range(1, int(self.N_perturb)):
            self.variables[f"P_add_{i}"] = tk.DoubleVar(value=self.P_add[i])
            self.variables[f"posx_add_{i}"] = tk.DoubleVar(value=self.pos_add[i][1])
            self.variables[f"posy_add_{i}"] = tk.DoubleVar(value=self.pos_add[i][0])
            self.variables[f"dimx_add_{i}"] = tk.DoubleVar(value=self.dim_add[i][1])
            self.variables[f"dimy_add_{i}"] = tk.DoubleVar(value=self.dim_add[i][0])
            self.variables[f"delais_add_{i}"] = tk.DoubleVar(value=self.delais_add[i])
            self.variables[f"fin_add_{i}"] = tk.DoubleVar(value=self.fin_add[i])

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
            "delais_R": 0,
            "fin_R": 10,
            "N_perturb": 0,
            "puissance_add": [1],
            "position_add": [[1,1]], # [[y,x], [y,x]...]
            "dimensions_add": [[1,1]], # [[y,x], [y,x]...]
            "temps_add": [[0,10]]
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
        self.tabs.config(width=int(self.screen_width * 0.5), height=int(self.screen_height * 0.7))

        # Initialisation des différents onglets
        self.controle_frame() # Contrôle de base
        self.plaque() # Dimensions de la plaque
        self.mat() # Paramètres du matériau de la plaque
        self.resolution() # Résolutions de la simulation de la plaque
        self.perturb() # Contrôle de la puissance déposée

        # Initialisation des boutons OK et Graphique
        self.etat_OK = ttk.Button(self.inter,text = 'OK', command = self.no_graphique)
        self.etat_OK.grid(column=0, row=3, pady=5, sticky="ew")
        ttk.Button(self.inter,text = 'Graphique', command = self.yes_graphique).grid(column=1, row=3, pady=5, sticky="ew")
        
        # Initialisation du bouton Arrêt
        self.etat_arret = ttk.Button(self.inter,text = 'Arrêt', command = self.arret)
        self.etat_arret.grid(column=0, row=3, pady=5, sticky="ew")
        self.etat_arret.grid_remove()

        # Initialisation de la barre de progression
        tk.Label(self.inter, text="Progression :").grid(column=0, row=4, padx=10, pady=5, sticky="ew")
        self.progres = ttk.Progressbar(self.inter, orient="horizontal", length=100, mode="determinate")
        self.progres.grid(column=1, row=4, padx=10, pady=5, sticky="ew")

        # Initialise l'endroit où les erreurs s'afficherons
        tk.Label(self.inter, text="Erreur : ").grid(column=0, row=5, padx=10, pady=5, sticky="ew")

        # Création d'un dictionnaire pour stocker les messages d'erreurs
        self.var_messages = {
            "num": "Les variables entrées doivent être numériques",
            "dim": "Les dimensions doivent être plus grandes que zéro",
            "res": "Les résolutions doivent être plus grandes que zéro",
            "temps": "Le temps de simulation doit être plus grand que zéro",
            "par": "Les paramètres du matériau doivent être plus grands que zéro",
            "actuateur": "La puissance de l'actuateur doit se situer entre -5 et 5 [W]"
        }

        # Création d'un dictionnaire pour créer des Labels
        self.var_labels = {}
        for key, message in self.var_messages.items():
            label = tk.Label(self.inter, text=message)
            label.grid(column=1, row=5, padx=10, pady=5, sticky="ew")
            label.grid_remove() # Cacher l'erreur
            self.var_labels[key] = label

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
        self.entry(self.frame, "Durée de la simulation [s]", "t_simul", 3) # Durée de la simulation


    def plaque(self):
        """
        Description :
        Créer le deuxième onglet de l'interface, permettant
            le contrôle des paramètres de la plaque

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Dimensions de la plaque
        self.plaque_frame = ttk.Frame(self.tabs, padding=10)
        self.plaque_frame.grid()
        self.tabs.add(self.plaque_frame, text="Dimensions de la plaque")

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


    def perturb(self):
        """
        Description :
        Créer le cinquième onglet de l'interface, permettant
            le contrôle des paramètres des perturbations

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Contrôle des perturbations
        self.perturb_frame = ttk.Frame(self.tabs, padding=10)
        self.perturb_frame.grid()
        self.tabs.add(self.perturb_frame, text="Contrôle des perturbations")

        # Initialisation des entrées
        self.entry(self.perturb_frame, "Puissance déposée avec la résistance [W]", "R_depo", 0) # Résistance de perturbation
        self.entry(self.perturb_frame, "Délais avant l'application [s]", "R_delais", 1)
        self.entry(self.perturb_frame, "Délais avant la fin de l'application [s]", "R_fin", 2)

        # Nombre de perturbation à ajouter
        self.label_N_perturb = ttk.Label(self.perturb_frame, text="Nombre de perturbations à ajouter [-]")
        self.label_N_perturb.grid(column=0, row=3, padx=5, pady=5, sticky="w")
        self.entry_N_perturb = ttk.Entry(self.perturb_frame, textvariable=self.variables["N_perturb"])
        self.entry_N_perturb.grid(column=1, row=3, padx=5, pady=5, sticky="ew") 
        # self.entry(self.perturb_frame, "Nombre de perturbations à ajouter [-]", "N_perturb", 3) # Nombre de perturbation à ajouter
        
        # Initialisation du bouton OK
        self.bouton_OK = ttk.Button(self.perturb_frame,text = 'OK', command = self.perturbations)
        self.bouton_OK.grid(column=0, row=5, pady=5, columnspan=2,)

        # Initialisation du graphique permettant de voir la position des perturbations
        ttk.Button(self.perturb_frame, text="Voir la plaque", command=self.graphique_plaque).grid(row=6, column=0, columnspan=2, pady=5)


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
        Regarde si les variables respectent les conditions limites

        Arguments : -

        Retourne : -
        """
        # Enlève les anciens messages d'erreurs
        for key, message in self.var_messages.items():
            self.var_labels[key].grid_remove()
        
        # Sauvegarder les variables selon les modifications de l'utilisateur dans l'interface
        try:
            # Contrôle de base
            self.T_plaque = self.variables["T_plaque"].get()
            self.T_amb=self.variables["T_amb"].get()
            self.P=self.variables["P"].get()
            self.t_simul=self.variables["t_simul"].get()

            # Dimensions de la plaque
            self.dim= [self.variables["dimy"].get(), self.variables["dimx"].get()] # [y,x]
            self.e=self.variables["e"].get() 

            # Paramètres du matériau de la plaque
            self.rho=self.variables["rho"].get()
            self.cp=self.variables["cp"].get()
            self.k=self.variables["k"].get()
            self.h=self.variables["h"].get()

            # Résolution de la simulation de la plaque
            self.dx=self.variables["dx"].get()
            self.dy=self.variables["dy"].get()
            self.dt=self.variables["dt"].get()

            # Contrôle des perturbations
            self.R_depo=self.variables["R_depo"].get()
            self.R_delais=self.variables["R_delais"].get()
            self.R_fin=self.variables["R_fin"].get()

            self.P_add = []
            self.pos_add = []
            self.dim_add = []
            self.temps_add = []
            for i in range(1, int(self.N_perturb)+1):
                self.P_add.append(self.variables[f"P_add_{i}"].get())
                self.pos_add.append([self.variables[f"posy_add_{i}"].get(), self.variables[f"posx_add_{i}"].get()]) # [y,x]
                self.dim_add.append([self.variables[f"dimy_add_{i}"].get(), self.variables[f"dimx_add_{i}"].get()]) # [y,x]
                self.temps_add.append([self.variables[f"delais_add_{i}"].get(), self.variables[f"fin_add_{i}"].get()]) # [début, fin]

            # les dimensions entrées doivent être plus grandes que zéro
            if any(d <= 0 for d in (self.dim[0], self.dim[1], self.e)):
                self.var_labels["dim"].grid()
                return False
            
            # les résolutions entrées doivent être plus grandes que zéro
            elif any(r <= 0 for r in (self.dx, self.dy, self.dt)):
                self.var_labels["res"].grid()
                return False
            
            # les paramètres du matériau entrés doivent être plus grands que zéro
            elif any(p <= 0 for p in (self.h, self.rho, self.cp, self.k)):
                self.var_labels["par"].grid()
                return False
            
            # le temps de simulation entré doit être plus grand que zéro
            elif self.t_simul <= 0:
                self.var_labels["temps"].grid()
                return False
            
            # la puissance de l'actuateur entrée doit être entre -5 et 5 W
            elif self.P < -5 or self.P > 5:
                self.var_labels["actuateur"].grid()
                return False
        except:
            # Si les variables entrées ne sont pas numériques
            self.var_labels["num"].grid()
            return False
        else:
            # Quantité d'itérations
            self.saut = round(( self.t_simul / (10 * self.dt))**(1/2))
            self.N = 10 * self.saut

            # Sauvegarde des données mises à jour dans le JSON
            self.data_fait = {
                "dimensions": [self.dim[0],self.dim[1]], # [y,x]
                "epaisseur": self.e,
                "resolution_x": self.dx,
                "resolution_y": self.dy,
                "resolution_t": self.dt,
                "temps_simulation": self.t_simul, # [s]
                "densite": self.rho,
                "cap_calorifique": self.cp,
                "conduc_thermique": self.k,
                "coef_convection": self.h,
                "T_plaque": self.T_plaque,
                "T_ambiante": self.T_amb,
                "puissance_actuateur": self.P,
                "puissance_R": self.R_depo,
                "delais_R": self.R_delais,
                "fin_R": self.R_fin,
                "N_perturb": self.N_perturb,
                "puissance_add": self.P_add,
                "position_add": self.pos_add, # [y,x]
                "dimensions_add": self.dim_add, # [y,x]
                "temps_add": self.temps_add
            }
            self.sauvegarder_json()

            # Créer une variable perturbation pour gérer les perturbations dans la simulation
            # position du coin inférieur droit (y,x), P, (longueur, largeur), (t_debut, t_fin) en cm
            perturbations=[
                [((0.015+0.021-0.003), ((self.dim[1]/200)-0.0015)), self.R_depo, (0.6, 0.3), (self.R_delais, self.R_fin)], # Résistance de perturbation
                ]
            for i in range(int(self.N_perturb)):
                perturbations.append([
                    (self.pos_add[i][1], self.pos_add[i][0]), # (y,x)
                    self.P_add[i],
                    (self.dim_add[i][1], self.dim_add[i][0]), # (y,x)
                    (self.temps_add[i][0], self.temps_add[i][1]) # (début, fin)
                    ])

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
                t_simul=self.t_simul,
                perturbations=perturbations
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
        
        # Remet le bouton OK et l'ajout de perturbations dans l'onglet perturbations
        self.bouton_OK.grid()
        self.label_N_perturb.grid()
        self.entry_N_perturb.grid()

        # Supprimer l'ancien canvas s'il existe
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().grid_forget() 

        # Définit le maximum de la barre de progression
        self.progres.configure(maximum=self.t_simul)

        # Roule l'interface selon la quantité d'itérations
        for n in range(self.N):
            self.progres["value"] = self.Ma_plaque.rep_echelon[0][-1] # Avance la barre de progression
            self.inter.update_idletasks()
            if self.Ma_plaque.rep_echelon[0][-1] > self.t_simul: # Si le temps de simulation est dépassé
                self.etat_arret.grid_remove() # Voir le bouton OK
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

        # Remet le bouton OK et l'ajout de perturbations dans l'onglet perturbations
        self.bouton_OK.grid()
        self.label_N_perturb.grid()
        self.entry_N_perturb.grid()

        # Supprimer l'ancien canvas s'il existe
        if hasattr(self, "canvas"):
            self.canvas.get_tk_widget().grid_forget() 

        # Voir le bouton arrêt
        self.etat_OK.grid_remove()
        self.etat_arret.grid()

        # Définit le maximum de la barre de progression
        self.progres.configure(maximum=self.t_simul)

        # Roule l'interface selon la quantité d'itérations
        for n in range(self.N):
            self.progres["value"] = self.Ma_plaque.rep_echelon[0][-1] # Avance la barre de progression
            self.inter.update_idletasks()
            if self.Ma_plaque.rep_echelon[0][-1] > self.t_simul: # Si le temps de simulation est dépassé
                self.etat_arret.grid_remove() # Voir le bouton OK
                self.etat_OK.grid()
                break # Arrêt de la simulation
            else:
                self.Ma_plaque.show() # Réitère les graphiques
                for k in range(self.saut):
                    self.Ma_plaque.iteration()
            if self.var_arret is True: # Si le bouton arrêt est cliqué
                self.var_arret = False
                break # Arrêt de la simulation
        self.Ma_plaque.enregistre_rep_echelon()


    def arret(self):
        """
        Description :
        Arrête la simulation
        Affiche le bouton OK

        Arguments : -

        Retourne : -
        """
        self.var_arret = True

        # Voir le bouton OK
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

        # Couleur les perturbations rouge 
        for p in self.Ma_plaque.perturbations: # Si il existe plusieurs perturbations
            (iy1,iy2,ix1,ix2), Temp, t = p
            plaque[iy1:iy2,ix1:ix2] = [1, 0, 0]  # Rouge pour les perturbation 

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
        self.canvas = FigureCanvasTkAgg(figy, master=self.inter)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=5, rowspan=5)

        # Le graphique est bien placé dans l'interface
        self.perturb_frame.grid_rowconfigure(0, weight=1)
        self.perturb_frame.grid_columnconfigure(0, weight=1)


    def perturbations(self):
        """
        Description :
        Affiche un onglet pour chaque perturbation permettant de contrôler
            sa puissance, sa position, sa taille et son temps d'application

        Arguments : -

        Retourne : -
        """
        # Sauvegarde le nombre de perturbations antérieur
        self.ancien_N_perturb = self.N_perturb

        # Regarde le nombre de perturbations voulues
        self.N_perturb = self.variables["N_perturb"].get()

        # Enlever le bouton OK et la possibilité d'ajouter des perturbations
        self.bouton_OK.grid_remove()
        self.label_N_perturb.grid_remove()
        self.entry_N_perturb.grid_remove()

        # Conditions limites
        if self.N_perturb < 0:
            raise KeyError  # perturb doit être positif
        elif self.N_perturb == 0:
            raise KeyError  # pas de perturbation ajoutée

        # Initialisation des onglets
        self.perturb_tabs = ttk.Notebook(self.perturb_frame)
        self.perturb_tabs.grid(column=2, row=0, rowspan=4, columnspan=2)

        # Stocker les références des frames
        self.perturb_frames = []
        self.perturb_noms = []

        # Initier les entrées par rapport aux perturbations
        if self.ancien_N_perturb < self.N_perturb:
            for i in range(int(self.ancien_N_perturb) + 1, int(self.N_perturb) + 1):
                self.variables[f"P_add_{i}"] = tk.DoubleVar(value=1)
                self.variables[f"posx_add_{i}"] = tk.DoubleVar(value=1)
                self.variables[f"posy_add_{i}"] = tk.DoubleVar(value=1)
                self.variables[f"dimx_add_{i}"] = tk.DoubleVar(value=1)
                self.variables[f"dimy_add_{i}"] = tk.DoubleVar(value=1)
                self.variables[f"delais_add_{i}"] = tk.DoubleVar(value=0)
                self.variables[f"fin_add_{i}"] = tk.DoubleVar(value=10)

        for i in range(1, int(self.N_perturb) + 1):
            # Création d'une frame pour chaque perturbation
            frame = ttk.Frame(self.perturb_tabs, padding=10)
            frame.grid()

            # Ajout de la frame à l'onglet
            self.perturb_tabs.add(frame, text=f"Perturbation #{i}")
            self.perturb_noms.append(f"Perturbation #{i}")

            # Stocker la frame dans la liste
            self.perturb_frames.append(frame)

            # Ajout du nom de la perturbation
            tk.Label(frame, text=f"Perturbation #{i}").grid(column=0, row=0, padx=5, pady=5, sticky="ew")

            # Ajout de la puissance, position et grosseur
            self.entry(frame, "Puissance déposée [W]", f"P_add_{i}", 1)  # Puissance de la perturbation
            self.entry(frame, "Position en x de la puissance déposée [cm]", f"posx_add_{i}", 2)  # Position de la perturbation
            self.entry(frame, "Position en y de la puissance déposée [cm]", f"posy_add_{i}", 3)
            self.entry(frame, "Longueur en x de la puissance déposée [cm]", f"dimx_add_{i}", 4)  # Dimensions de la perturbation
            self.entry(frame, "Longueur en y de la puissance déposée [cm]", f"dimy_add_{i}", 5)
            self.entry(frame, "Délais avant l'application [s]", f"delais_add_{i}", 6)  # Durée de l'application de la perturbation
            self.entry(frame, "Délais avant la fin de l'application [s]", f"fin_add_{i}", 7)

        # Faire un menu des perturbations
        self.tab_selector = ttk.Combobox(self.perturb_frame, state="readonly")
        self.tab_selector.grid(column=0, row=4, pady=5, columnspan=2)
        self.tab_selector["values"] = self.perturb_noms

        # Faire un menu déroulant
        self.tab_selector.current(0)
        self.tab_selector.bind("<<ComboboxSelected>>", self.changer_tab)

        # Cacher les perturbations sauf la première
        self.perturb_tabs.grid_remove()
        self.changer_tab()


    def changer_tab(self, event=None):
        """
        Description :
        Affiche un onglet et cache les autres

        Arguments : -

        Retourne : -
        """
        self.perturb_tabs.grid()

        selected_tab = self.tab_selector.get()
        tab_index = list(self.perturb_noms).index(selected_tab)

        print(selected_tab, tab_index)

        # Cache toutes les frames
        for i, frame in enumerate(self.perturb_frames):
            if i != tab_index:
                frame.grid_forget()

        # Affiche la frame correspondant à l'onglet sélectionné
        self.perturb_frames[tab_index].grid()

        # Sélectionner l'onglet
        self.perturb_tabs.select(tab_index)




Inter= Interface()