# pour lire un document JSON
import os
import json

# pour faire des documents nommés selon l'heure actuelle
from datetime import datetime

# pour faire rouler l'interface
import tkinter as tk
from tkinter import ttk

# pour faire jouer la simulation 
import SIMULATEUR

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
        self.dx = self.data_lu.get("resolution_x", 0.15) # doit être entre [0.1, dim plaque]
        self.dy = self.data_lu.get("resolution_y", 0.1) # doit être entre [0.1, dim plaque]
        self.dt = self.data_lu.get("resolution_t", None) # doit être plus grand que zéro
        self.t_simul = self.data_lu.get("temps_simulation", 600) # [s] doit être plus grand que 60 s
        self.T_plaque = self.data_lu.get("T_plaque", 25.0)
        self.T_amb = self.data_lu.get("T_ambiante", 25.0)
        self.rho = self.data_lu.get("densite", 2700) # doit être plus grand que zéro
        self.cp = self.data_lu.get("cap_calorifique", 897.0) # doit être plus grand que zéro
        self.k = self.data_lu.get("conduc_thermique", 167.0)  # doit être plus grand que zéro
        self.h = self.data_lu.get("coef_convection", 12)  # doit être plus grand que zéro
        self.P = self.data_lu.get("puissance_actuateur", 1.5) # doit être entre -5 et 5 W
        self.actuateur_pos = self.data_lu.get("position_actuateur", [1.5, 3]) # actuateur doit être entièrement sur la plaque
        self.actuateur_gros = self.data_lu.get("grosseur_actuateur", [1.5, 1]) # doivent être plus grandes que zéro
        self.t_actuateur = self.data_lu.get("temps_actuateur", 0) # doit être plus grand que zéro et plus petit que le temps de simulation
        self.R_depo = self.data_lu.get("puissance_R", 0) 
        self.R_delais = self.data_lu.get("delais_R", 0) # doit être plus grand que zéro et plus petit que le temps de simulation
        self.R_fin = self.data_lu.get("fin_R", 10) # temps d'arrêt ne peut être plus petit que le temps d'application
        self.N_perturb = self.data_lu.get("N_perturb", 0) # doit être positif 
        self.pos_therm = self.data_lu.get("position_thermistances", None) # [y, x] 
        
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
            self.dt = min((self.dx/100)**2/(4*self.alpha), (self.dy/100)**2/(4*self.alpha)) 
        if self.pos_therm is None:
            self.pos_therm = [
                [1.5, self.dim[1]/2],
                [6, self.dim[1]/2],
                [10.4, self.dim[1]/2]
                ] # centrer les thermistances sur la plaque

        # Initier toutes les entrées de l'interface
        self.variables = {key: tk.DoubleVar(value=val) for key, val in {
            "dimy": self.dim[0], "dimx": self.dim[1], "e": self.e,
            "dx": self.dx, "dy": self.dy, "dt": self.dt,
            "t_simul": self.t_simul,
            "rho": self.rho, "cp": self.cp, 
            "k": self.k, "h": self.h,
            "T_plaque": self.T_plaque, "T_amb": self.T_amb, "P": self.P,
            "act_posy": self.actuateur_pos[0], "act_posx": self.actuateur_pos[1],
            "act_grosy": self.actuateur_gros[0], "act_grosx": self.actuateur_gros[1],
            "act_t": self.t_actuateur,
            "R_depo": self.R_depo, "R_delais": self.R_delais, "R_fin": self.R_fin,
            "N_perturb": self.N_perturb,
            "posy_therm_1": self.pos_therm[0][0], "posx_therm_1": self.pos_therm[0][1],
            "posy_therm_2": self.pos_therm[1][0], "posx_therm_2": self.pos_therm[1][1],
            "posy_therm_3": self.pos_therm[2][0], "posx_therm_3": self.pos_therm[2][1]
        }.items()}

        # Initier les entrées par rapport aux perturbations
        for i in range(1, len(self.P_add)+1):
            j = i-1
            self.variables[f"P_add_{i}"] = tk.DoubleVar(value=self.P_add[j])
            self.variables[f"posx_add_{i}"] = tk.DoubleVar(value=self.pos_add[j][1])
            self.variables[f"posy_add_{i}"] = tk.DoubleVar(value=self.pos_add[j][0])
            self.variables[f"dimx_add_{i}"] = tk.DoubleVar(value=self.dim_add[j][1])
            self.variables[f"dimy_add_{i}"] = tk.DoubleVar(value=self.dim_add[j][0])
            self.variables[f"delais_add_{i}"] = tk.DoubleVar(value=self.temps_add[j][0])
            self.variables[f"fin_add_{i}"] = tk.DoubleVar(value=self.temps_add[j][0])

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
            "T_plaque": 25,
            "T_ambiante": 25,
            "densite": 2700,
            "cap_calorifique": 897,
            "conduc_thermique": 167,
            "coef_convection": 12,
            "puissance_actuateur": 1.5,
            "position_actuateur": [1.5, 3],
            "grosseur_actuateur": [1.5, 1.5], # [y, x]
            "temps_actuateur": 0,
            "puissance_R": 0,
            "delais_R": 0,
            "fin_R": 10,
            "N_perturb": 0,
            "puissance_add": [1],
            "position_add": [[1,1]], # [[y,x], [y,x]...]
            "dimensions_add": [[1,1]], # [[y,x], [y,x]...]
            "temps_add": [[0,10]],
            "position_thermistances": None
        }
    

    def entry(self, parent, texte, var, column, row):
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
        ttk.Label(parent, text=texte).grid(column=column, row=row, padx=5, pady=5, sticky="w") # Label
        entry = ttk.Entry(parent, textvariable=self.variables[var]) # Entry
        entry.grid(column=column+1, row=row, padx=5, pady=5, sticky="ew") # Placement de l'entry
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
        self.tabs.grid(column=0, row=0, rowspan=2, sticky="nsew", columnspan=2)
        self.tabs.config(width=int(self.screen_width * 0.5), height=int(self.screen_height * 0.5))

        # Initialisation des différents onglets
        self.controle_frame() # Contrôle de base
        self.plaque() # Paramètres de la plaque
        self.perturb() # Contrôle de la puissance déposée
        self.thermistances() # Contrôle des thermistances

        # Initialisation des boutons OK, Voir la plaque et Graphique
        self.etat_OK = ttk.Button(self.inter,text = 'Enregistrer un CSV', command = self.no_graphique)
        self.etat_OK.grid(column=0, row=3, pady=5, sticky="ew", columnspan=2)
        ttk.Button(self.inter, text="Voir la plaque", command=self.graphique_plaque).grid(column=0, row=4, pady=5, sticky="ew", columnspan=2)  # Initialisation du graphique permettant de voir la position des perturbations
        ttk.Button(self.inter,text = 'Graphique', command = self.yes_graphique).grid(column=0, row=5, pady=5, sticky="ew", columnspan=2)
        
        # Initialisation du bouton Arrêt
        self.etat_arret = ttk.Button(self.inter,text = 'Arrêt', command = self.arret)
        self.etat_arret.grid(column=0, row=5, pady=5, sticky="ew", columnspan=2)
        self.etat_arret.grid_remove()

        # Initialisation de la barre de progression
        tk.Label(self.inter, text="Progression :").grid(column=0, row=6, padx=10, pady=5, sticky="ew")

        # Initialise l'endroit où les erreurs s'afficherons
        tk.Label(self.inter, text="Erreur : ").grid(column=0, row=7, padx=10, pady=5, sticky="ew")

        # Création d'un dictionnaire pour stocker les messages d'erreurs
        self.var_messages = {
            "num": "Les variables entrées doivent être numériques",
            "dim": "Les dimensions de la plaque doivent être plus grandes que zéro et plus petites que 1000 fois les résolutions",
            "res": "Les résolutions doivent être entre 0.1 cm et les dimensions de la résistance de perturbation (x=0.3 et y=0.6) cm",
            "temps": "Le temps de simulation doit être plus grand que 60 s",
            "par": "Les paramètres du matériau doivent être plus grands que zéro",
            "P_actuateur": "La puissance de l'actuateur doit être entre -5 et 5 [W]",
            "pos_actuateur": "L'actuateur doit entièrement se situer sur la plaque",
            "res_actuateur": "Les dimensions de l'actuateur doivent être supérieures aux résolutions de la plaque",
            "dim_actuateur": "Les dimensions de l'actuateur doivent être plus grandes que zéro",
            "temps_actuateur": "Le temps d'application de l'actuateur doit être entre 0 et la durée de la simulation",
            "N_perturb": "La quantité de perturbations doit être positive",
            "delais_perturb": "Le temps d'application des perturbations doit être entre 0 et la durée de la simulation",
            "fin_perturb": "Le temps d'arrêt des perturbations ne peut être plus petit que le temps d'application",
            "pos_perturb": "Les perturbations doivent entièrement se situer sur la plaque",
            "dim_perturb": "Les dimensions des perturbations doivent être plus grandes que zéro",
            "res_perturb": "Les dimensions des perturbations doivent être supérieures aux résolutions de la plaque",
            "pos_therm": "Les thermistances doivent entièrement se situer sur la plaque",
        }

        # Création d'un dictionnaire pour créer des Labels
        self.var_labels = {}
        for key, message in self.var_messages.items():
            label = tk.Label(self.inter, text=message)
            label.grid(column=1, row=7, padx=10, pady=5, sticky="ew")
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
        tk.Label(self.frame, text="Contrôle de la simulation").grid(column=0, row=0, padx=5, pady=5, columnspan=2)
        self.entry(self.frame, "Température initiale de la plaque [°C]", "T_plaque", 0, 1) # Température initiale de la plaque
        self.entry(self.frame, "Température ambiante [°C]", "T_amb", 0, 2) # Température ambiante
        self.entry(self.frame, "Durée de la simulation [s]", "t_simul", 0, 3) # Durée de la simulation

        tk.Label(self.frame, text="Paramètres de l'actuateur").grid(column=2, row=0, padx=5, pady=5, columnspan=2)
        self.entry(self.frame, "Puissance appliquée à l'actuateur [W]", "P", 2, 1) # Puissance appliquée à l'actuateur
        self.entry(self.frame, "Temps d'application de l'actuateur [s]", "act_t", 2, 2) # Temps d'application de l'actuateur
        self.entry(self.frame, "Position en x de l'actuateur [cm]", "act_posx", 2, 3) # Position de l'actuateur
        self.entry(self.frame, "Position en y de l'actuateur [cm]", "act_posy", 2, 4) 
        self.entry(self.frame, "Longueur en x de l'actuateur [cm]", "act_grosx", 2, 5) # Grosseur de l'actuateur
        self.entry(self.frame, "Longueur en y de l'actuateur [cm]", "act_grosy", 2, 6) 


    def plaque(self):
        """
        Description :
        Créer le deuxième onglet de l'interface, permettant
            le contrôle des paramètres de la plaque

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Paramètres de la plaque
        self.plaque_frame = ttk.Frame(self.tabs, padding=10)
        self.plaque_frame.grid()
        self.tabs.add(self.plaque_frame, text="Paramètres de la plaque")

        # Dimensions de la plaque
        tk.Label(self.plaque_frame, text="Dimensions de la plaque").grid(column=0, row=0, padx=5, pady=5, columnspan=2)

        # Initialisation des entrées
        self.entry(self.plaque_frame, "Longueur en x de la plaque [cm]", "dimx", 0, 1) # Dimensions de la plaque
        self.entry(self.plaque_frame, "Longueur en y de la plaque [cm]", "dimy", 0, 2)
        self.entry(self.plaque_frame, "Épaisseur de la plaque [cm]", "e", 0, 3) # Épaisseur de la plaque

        # Résolution de la simulation de la plaque
        tk.Label(self.plaque_frame, text="Résolution de la simulation de la plaque").grid(column=0, row=5, padx=5, pady=5, columnspan=2)

        # Initialisation des entrées
        self.entry(self.plaque_frame, "Résolution en x [cm]", "dx", 0, 6) # Résolutions de longueur
        self.entry(self.plaque_frame, "Résolution en y [cm]", "dy", 0, 7)

        # Paramètres du matériau de la plaque
        tk.Label(self.plaque_frame, text="Paramètres du matériau de la plaque").grid(column=2, row=0, padx=5, pady=5, columnspan=2)

        # Initialisation des entrées
        self.entry(self.plaque_frame, "Densité du matériau [kg / m³]", "rho", 2, 1) # Densité du matériau
        self.entry(self.plaque_frame, "Capacité calorifique du matériau [J / kg.K]", "cp", 2, 2) # Capacité calorifique du matériau
        self.entry(self.plaque_frame, "Conductivité thermique du matériau [W / m.K]", "k", 2, 3) # Conductivité thermique du matériau
        self.entry(self.plaque_frame, "Coefficient de convection du matériau [W / m².K]", "h", 2, 4) # Coefficient de convection


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
        self.label_R = ttk.Label(self.perturb_frame, text="Résistance de perturbation")
        self.label_R.grid(column=0, row=0, padx=5, pady=5, sticky='new', columnspan=2)
        self.entry(self.perturb_frame, "Puissance déposée avec la résistance [W]", "R_depo", 0, 1)
        self.entry(self.perturb_frame, "Délais avant l'application [s]", "R_delais", 0, 2)
        self.entry(self.perturb_frame, "Délais avant la fin de l'application [s]", "R_fin", 0, 3)

        # Nombre de perturbation à ajouter
        self.label_N_perturb = ttk.Label(self.perturb_frame, text="Nombre de perturbations à ajouter [-]")
        self.label_N_perturb.grid(column=0, row=4, padx=5, pady=5, sticky='ew')
        self.entry_N_perturb = ttk.Entry(self.perturb_frame, textvariable=self.variables["N_perturb"])
        self.entry_N_perturb.grid(column=1, row=4, padx=5, pady=5, sticky='ew') 
        
        # Initialisation du bouton OK
        self.bouton_OK = ttk.Button(self.perturb_frame,text = 'OK', command = self.perturbations)
        self.bouton_OK.grid(column=0, row=5, pady=5, columnspan=2,)

    
    def thermistances(self):
        """
        Description :
        Créer le sixième onglet de l'interface, permettant
            le contrôle des thermistances

        Arguments : -

        Retourne : -
        """
        # Initialisation de l'onglet Contrôle des thermistances
        self.therm_frame = ttk.Frame(self.tabs, padding=10)
        self.therm_frame.grid()
        self.tabs.add(self.therm_frame, text="Contrôle des thermistances")

        # 1ere thermistance
        tk.Label(self.therm_frame, text="Thermistance à l'actuateur").grid(column=0, row=0, padx=5, pady=5, columnspan=2)
        self.entry(self.therm_frame, "Position en x de la thermistance", "posx_therm_1", 0, 1) 
        self.entry(self.therm_frame, "Position en y de la thermistance", "posy_therm_1", 0, 2)

        # 2e thermistance
        tk.Label(self.therm_frame, text="Thermistance au milieu").grid(column=0, row=4, padx=5, pady=5, columnspan=2)
        self.entry(self.therm_frame, "Position en x de la thermistance", "posx_therm_2", 0, 5) 
        self.entry(self.therm_frame, "Position en y de la thermistance", "posy_therm_2", 0, 6)

        # 3e thermistance
        tk.Label(self.therm_frame, text="Thermistance au laser").grid(column=0, row=8, padx=5, pady=5, columnspan=2)
        self.entry(self.therm_frame, "Position en x de la thermistance", "posx_therm_3", 0, 9) 
        self.entry(self.therm_frame, "Position en y de la thermistance", "posy_therm_3", 0, 10)


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

    
    def condition_respectee(self, condition, error_key):
        """
        Description :
        Vérifie la condition et affiche le message d'erreur associé si non vérifiée

        Arguments : -

        Retourne :
        True si la condition est respectée
        False si la condition n'est pas respectée
        """
        if not condition:
            self.var_labels[error_key].grid()
            return False
        return True
    

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
        for label in self.var_labels.values():
            label.grid_remove()
        
        # Sauvegarder les variables selon les modifications de l'utilisateur dans l'interface
        try:
            # Contrôle de base
            self.T_plaque = self.variables["T_plaque"].get()
            self.T_amb=self.variables["T_amb"].get()
            self.P=self.variables["P"].get()
            self.t_simul=self.variables["t_simul"].get()
            self.actuateur_pos=[
                self.variables["act_posy"].get(),
                self.variables["act_posx"].get()
                ] 
            self.actuateur_gros=[
                self.variables["act_grosy"].get(),
                self.variables["act_grosx"].get()
                ]
            self.t_actuateur=self.variables["act_t"].get()

            # Dimensions de la plaque
            self.dim= [
                self.variables["dimy"].get(),
                self.variables["dimx"].get()
                ] # [y,x]
            self.e=self.variables["e"].get() 

            # Paramètres du matériau de la plaque
            self.rho=self.variables["rho"].get()
            self.cp=self.variables["cp"].get()
            self.k=self.variables["k"].get()
            self.h=self.variables["h"].get()

            # Résolution de la simulation de la plaque
            self.dx=self.variables["dx"].get()
            self.dy=self.variables["dy"].get()
            # self.dt=self.variables["dt"].get()

            # Contrôle des perturbations
            self.N_perturb = self.variables["N_perturb"].get()
            self.R_depo=self.variables["R_depo"].get()
            self.R_delais=self.variables["R_delais"].get()
            self.R_fin=self.variables["R_fin"].get()

            self.P_add = []
            self.pos_add = []
            self.dim_add = []
            self.temps_add = []
            for i in range(1, int(self.N_perturb)+1):
                self.P_add.append(self.variables[f"P_add_{i}"].get())
                self.pos_add.append([
                    self.variables[f"posy_add_{i}"].get(),
                     self.variables[f"posx_add_{i}"].get()
                     ]) # [y,x]
                self.dim_add.append([
                    self.variables[f"dimy_add_{i}"].get(),
                    self.variables[f"dimx_add_{i}"].get()
                    ]) # [y,x]
                self.temps_add.append([
                    self.variables[f"delais_add_{i}"].get(),
                    self.variables[f"fin_add_{i}"].get()
                    ]) # [début, fin]

            # Contrôle des thermistances
            self.pos_therm=[
                [self.variables["posy_therm_1"].get(), self.variables["posx_therm_1"].get()],
                [self.variables["posy_therm_2"].get(), self.variables["posx_therm_2"].get()],
                [self.variables["posy_therm_3"].get(), self.variables["posx_therm_3"].get()]
                ]
        except:
            self.var_labels["num"].grid()
            return False
        
        # Créer une variable perturbation pour gérer les perturbations dans la simulation
        # position du coin inférieur droit (y,x), P, (longueur, largeur), (t_debut, t_fin) en cm
        perturbations=[[
            ((3.3), ((self.dim[1]/2)-0.0015)),
              self.R_depo,
              (0.6, 0.3),
              (self.R_delais, self.R_fin)
              ]] # Résistance de perturbation

        # les dimensions entrées doivent être plus grandes que zéro et plus petites que 1000*résolutions
        if not self.condition_respectee(all(0 < val for val in (self.dim[0], self.dim[1], self.e)), "dim"):
            return False
        if not self.condition_respectee(self.dim[0] <= 1000*self.dy and self.dim[1] <= 1000*self.dx, "dim"):
            return False
            
        # les résolutions entrées doivent être plus grandes que zéro et plus petites que la R de perturbation
        if not self.condition_respectee(all(r >= 0.1 for r in (self.dx, self.dy)), "res"):
            return False
        if not self.condition_respectee(self.dx < perturbations[0][2][1] and self.dy < perturbations[0][2][0], "res"):
            return False
            
        # les paramètres du matériau entrés doivent être plus grands que zéro
        if not self.condition_respectee(all(p > 0 for p in (self.h, self.rho, self.cp, self.k)), "par"):
            return False
            
        # le temps de simulation entré doit être plus grand que 60 s
        if not self.condition_respectee(self.t_simul > 60, "temps"):
            return False
            
        # la puissance de l'actuateur entrée doit être entre -5 et 5 W
        if not self.condition_respectee(-5 <= self.P <= 5, "P_actuateur"):
            return False
            
        # l'actuateur doit se trouver sur la plaque et
        # ses dimensions doivent être plus grandes que la résolution de la plaque
        act_y_pos = self.actuateur_pos[0]+self.actuateur_gros[0]
        act_y_neg = self.actuateur_pos[0]-self.actuateur_gros[0]
        act_x_pos = self.actuateur_pos[1]+self.actuateur_gros[1]
        act_x_neg = self.actuateur_pos[1]-self.actuateur_gros[1]
        if not self.condition_respectee(0 <= act_y_pos <= self.dim[0] and 0 <= act_x_pos <= self.dim[1], "pos_actuateur"):
            return False
        elif not self.condition_respectee(0 <= act_y_neg <= self.dim[0] and 0 <= act_x_neg <= self.dim[1], "pos_actuateur"):
            return False
        elif not self.condition_respectee(self.dy <= self.actuateur_gros[0] and self.dx <= self.actuateur_gros[1], "res_actuateur"):
            return False
        
        # les dimensions de l'actuateur doivent être plus grandes que zéro
        if not self.condition_respectee(0 < self.actuateur_gros[0] and 0 < self.actuateur_gros[1], "dim_actuateur"):
            return False
        
        # le temps d'application de l'actuateur doit être positif et plus petit que la durée de la simulation
        if not self.condition_respectee(0 <= self.t_actuateur <= self.t_simul, "temps_actuateur"):
            return False
            
        # le temps d'application des perturbations doit être positif et plus petit que la durée de la simulation
        # le temps d'arrêt des perturbations ne peut être plus petit que le temps d'application
        # les perturbations doivent se trouver sur la plaque
        # les dimensions des perturbations doivent être plus grandes que la résolution de la plaque

        # résistance de perturbation
        if not self.condition_respectee(0 <= self.R_delais <= self.t_simul, "delais_perturb"):
            return False
        if not self.condition_respectee(0 <= self.R_fin <= self.t_simul, "delais_perturb"):
            return False
        if not self.condition_respectee(self.R_fin >= self.R_delais, "fin_perturb"):
            return False
            
        # perturbations
        for i in range(int(self.N_perturb)):
            perturb_y_pos = self.pos_add[i][0] + self.dim_add[i][0]
            perturb_y_neg = self.pos_add[i][0] - self.dim_add[i][0]
            perturb_x_pos = self.pos_add[i][1] + self.dim_add[i][1]
            perturb_x_neg = self.pos_add[i][1] - self.dim_add[i][1]

            if not self.condition_respectee(0 <= self.temps_add[i][0] <= self.t_simul, "delais_perturb"):
                return False
            elif not self.condition_respectee(0 <= self.temps_add[i][1] <= self.t_simul, "delais_perturb"):
                return False
            elif not self.condition_respectee(self.temps_add[i][1] >= self.temps_add[i][0], "fin_perturb"):
                return False
            elif not self.condition_respectee(0 <= perturb_y_pos <= self.dim[0] and 0 <= perturb_x_pos <= self.dim[1], "pos_perturb"):
                return False
            elif not self.condition_respectee(0 <= perturb_y_neg <= self.dim[0] and 0 <= perturb_x_neg <= self.dim[1], "pos_perturb"):
                return False
            elif not self.condition_respectee(0 < self.dim_add[i][0] and 0 < self.dim_add[i][1], "dim_perturb"):
                return False
            elif not self.condition_respectee(self.dy <= self.dim_add[i][0] and self.dx <= self.dim_add[i][1], "res_actuateur"):
                return False

        # les thermistances doivent se trouver sur la plaque  
        for i in self.pos_therm:
            if not self.condition_respectee(0 <= i[0] <= self.dim[0] and 0 <= i[1] <= self.dim[1], "pos_therm"):
                return False

        # Quantité d'itérations
        self.saut = round(( self.t_simul / (10 * self.dt))**(1/2))
        self.N = 10 * self.saut

        # Sauvegarde des données mises à jour dans le JSON
        self.data_fait = {
            "dimensions": self.dim, # [y,x]
            "epaisseur": self.e,
            "resolution_x": self.dx,
            "resolution_y": self.dy,
            "resolution_t": None,
            "temps_simulation": self.t_simul, # [s]
            "T_plaque": self.T_plaque,
            "T_ambiante": self.T_amb,
            "densite": self.rho,
            "cap_calorifique": self.cp,
            "conduc_thermique": self.k,
            "coef_convection": self.h,
            "puissance_actuateur": self.P,
            "position_actuateur": self.actuateur_pos,
            "grosseur_actuateur": self.actuateur_gros,
            "temps_actuateur": self.t_actuateur,
            "puissance_R": self.R_depo,
            "delais_R": self.R_delais,
            "fin_R": self.R_fin,
            "N_perturb": self.N_perturb,
            "puissance_add": self.P_add,
            "position_add": self.pos_add, # [y,x]
            "dimensions_add": self.dim_add, # [y,x]
            "temps_add": self.temps_add, # [début, fin]
            "position_thermistances": self.pos_therm
        }
        self.sauvegarder_json()

        # Ajouter les perturbations additionnelles
        for i in range(int(self.N_perturb)):
            perturbations.append([
                (self.pos_add[i][1], self.pos_add[i][0]), # (y,x)
                self.P_add[i],
                (self.dim_add[i][1], self.dim_add[i][0]), # (y,x)
                (self.temps_add[i][0], self.temps_add[i][1]) # (début, fin)
                ])

        # Initialise la simulation
        self.Ma_plaque = SIMULATEUR.Plaque(
            dimensions=(self.dim[0], self.dim[1]), # (y,x)
            epaisseur=self.e,
            resolution_x=self.dx,
            resolution_y=self.dy,
            resolution_t=self.dt,
            t_simul=self.t_simul,
            T_plaque=self.T_plaque,
            T_ambiante=self.T_amb,
            densite=self.rho,
            cap_calorifique=self.cp,
            conduc_thermique=self.k,
            coef_convection=self.h,
            puissance_actuateur=self.P,
            position_actuateur=(self.actuateur_pos[0], self.actuateur_pos[1]),
            grosseur_actuateur=(self.actuateur_gros[0], self.actuateur_gros[1]),
            temps_actuateur=self.t_actuateur,
            perturbations=perturbations,
            position_thermistances=self.pos_therm
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

        # Initialise la barre de progression
        self.progres = ttk.Progressbar(self.inter, orient="horizontal", length=100, mode="determinate")
        self.progres.grid(column=1, row=6, padx=10, pady=5, sticky="ew")

        # Définit le maximum de la barre de progression
        self.progres.configure(maximum=self.t_simul)

        # Réinitialise la barre de progression
        self.progres["value"] = 0

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
        if self.submit() == False:
            return

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

         # Initialise la barre de progression
        self.progres = ttk.Progressbar(self.inter, orient="horizontal", length=100, mode="determinate")
        self.progres.grid(column=1, row=6, padx=10, pady=5, sticky="ew")

        # Définit le maximum de la barre de progression
        self.progres.configure(maximum=self.t_simul)

        # Réinitialise la barre de progression
        self.progres["value"] = 0

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
        if self.submit() == False:
            return

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

        # Enlever les perturbations avec aucune puissance
        # Couleur les perturbations rouge 
        for p in self.Ma_plaque.perturbations: # Si il existe plusieurs perturbations
            if p[1][0][0] == 0:
                continue
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
        axy.legend(handles=legend_elements, bbox_to_anchor=(-0.2, 1))
        axy.set_xlabel("Position en x (mm)")
        axy.set_ylabel("Position en y (mm)")

        # Intégration à l'interface
        self.canvas = FigureCanvasTkAgg(figy, master=self.inter)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=5)


    def perturbations(self):
        """
        Description :
        Affiche un onglet pour chaque perturbation permettant de contrôler
            sa puissance, sa position, sa taille et son temps d'application

        Arguments : -

        Retourne : -
        """
        # Regarde le nombre de perturbations voulues
        self.N_perturb = self.variables["N_perturb"].get()
        
        # le nombre de perturbation doit être positif
        if self.N_perturb < 0:
            self.var_labels["N_perturb"].grid()
            return
        elif self.N_perturb == 0:
            return
        self.var_labels["N_perturb"].grid_remove()
        
        # Enlever le bouton OK et la possibilité d'ajouter des perturbations
        self.bouton_OK.grid_remove()
        self.label_N_perturb.grid_remove()
        self.entry_N_perturb.grid_remove()
            
        # Initialisation des onglets
        self.perturb_tabs = ttk.Notebook(self.perturb_frame)
        self.perturb_tabs.grid(column=3, row=0, rowspan=4)

        # Stocker les références des frames
        self.perturb_frames = []
        self.perturb_noms = []

        # Initier les entrées par rapport aux perturbations
        N_ancien_perturb = len(self.P_add)
        if N_ancien_perturb < self.N_perturb:
            for i in range(int(N_ancien_perturb) + 1, int(self.N_perturb) + 1):
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
            label_i = tk.Label(frame, text=f"Perturbation #{i}")
            label_i.grid(column=0, row=0, padx=5, pady=5)

            # Ajout de la puissance, position et grosseur
            self.entry(frame, "Puissance déposée [W]", f"P_add_{i}", 0, 1)  # Puissance de la perturbation
            self.entry(frame, "Position en x de la puissance déposée [cm]", f"posx_add_{i}", 0, 2)  # Position de la perturbation
            self.entry(frame, "Position en y de la puissance déposée [cm]", f"posy_add_{i}", 0, 3)
            self.entry(frame, "Longueur en x de la puissance déposée [cm]", f"dimx_add_{i}", 0, 4)  # Dimensions de la perturbation
            self.entry(frame, "Longueur en y de la puissance déposée [cm]", f"dimy_add_{i}", 0, 5)
            self.entry(frame, "Délais avant l'application [s]", f"delais_add_{i}", 0, 6)  # Durée de l'application de la perturbation
            self.entry(frame, "Délais avant la fin de l'application [s]", f"fin_add_{i}", 0, 7)

        # Faire un menu des perturbations
        self.tab_selector = ttk.Combobox(self.perturb_frame)
        self.tab_selector.grid(column=0, row=6, pady=5)
        self.tab_selector["values"] = self.perturb_noms

        # Faire un menu déroulant
        self.tab_selector.current(0)
        self.tab_selector.bind("<<ComboboxSelected>>", self.changer_tab)


    def changer_tab(self, event=None):
        """
        Description :
        Affiche un onglet et cache les autres

        Arguments : -

        Retourne : -
        """
        selected_tab = self.tab_selector.get()
        tab_index = list(self.perturb_noms).index(selected_tab)

        # Sélectionner l'onglet
        self.perturb_tabs.select(tab_index)




Inter= Interface()