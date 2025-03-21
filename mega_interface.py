import os
from tqdm import tqdm
import json
from datetime import datetime
import tkinter as tk # module de base
from tkinter import ttk # mettre ça beau
import mega_simulation


class Interface:
    def __init__(self):
        self.inter = tk.Tk()
        self.frame = ttk.Frame(self.inter, padding=10)
        self.frame.grid()
        self.inter.title('Contrôle de la simulation Python')
        self.graph = False

        # lire json
        self.lire_json()
        
        # Initialisation des variables depuis JSON ou valeurs par défaut (certaines valeurs =! 0)
        self.dim = self.data_lu.get("dimensions", [11.6,6.15]) if self.data_lu.get("dimensions", [11.6,6.15]) != 0 else [11.6,6.15]
        self.e = self.data_lu.get("epaisseur", 0.156) if self.data_lu.get("epaisseur", 0.156) != 0 else 0.156
        self.dx = self.data_lu.get("resolution_x", 0.15) if self.data_lu.get("resolution_x", 0.15) != 0 else 0.15
        self.dy = self.data_lu.get("resolution_y", 0.1) if self.data_lu.get("resolution_y", 0.1) != 0 else 0.1
        self.dt = self.data_lu.get("resolution_t", None)
        self.rho = self.data_lu.get("densite", 2700) if self.data_lu.get("densite", 2700) != 0 else 2700
        self.cp = self.data_lu.get("cap_calorifique", 897.0) if self.data_lu.get("cap_calorifique", 897.0) != 0 else 897.0
        self.k = self.data_lu.get("conduc_thermique", 167.0) if self.data_lu.get("conduc_thermique", 167.0) != 0 else 167.0
        self.h = self.data_lu.get("coef_convection", 12) if self.data_lu.get("coef_convection", 12) != 0 else 12
        self.T_plaque = self.data_lu.get("T_plaque", 25.0)
        self.T_amb = self.data_lu.get("T_ambiante", 23.0)
        self.P = self.data_lu.get("puissance_actuateur", 1.5)
        self.R_depo = self.data_lu.get("puissance_R", 0)
        self.T_depo = self.data_lu.get("puissance_ajoutee", 0)
        self.T_pos = self.data_lu.get("position_puissance", [12,12])

        # Initier variables avec calculs
        self.alpha = self.k/(self.rho*self.cp)
        if self.dt is None:
            self.dt = min((self.dx/100)**2/(4*self.alpha), (self.dy/100)**2/(4*self.alpha))  # À regarder!!

        # Initier toutes les entrées 
        self.variables = {key: tk.DoubleVar(value=val) for key, val in {
            "dimx": self.dim[0], "dimy": self.dim[1], "e": self.e, "dx": self.dx,
            "dy": self.dy, "dt": self.dt, "rho": self.rho, "cp": self.cp, 
            "k": self.k, "h": self.h, "T_plaque": self.T_plaque,
            "T_amb": self.T_amb, "P": self.P, "R_depo": self.R_depo,
            "T_depo": self.T_depo, "T_posx": self.T_pos[0], "T_posy": self.T_pos[1]
        }.items()}

        # Go go main interface
        self.main()
    

    def lire_json(self):
        """ Charge les données depuis le fichier JSON le plus récent ou utilise les valeurs par défaut. """
        
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
        return {
            "dimensions": [11.6,6.15],
            "epaisseur": 0.156,
            "resolution_x": 0.15,
            "resolution_y": 0.1,
            "resolution_t": None,
            "densite": 2700,
            "cap_calorifique": 897,
            "conduc_thermique": 167,
            "coef_convection": 12,
            "T_plaque": 23,
            "T_ambiante": 25,
            "puissance_actuateur": 1.5,
            "puissance_R": 0,
            "puissance_ajoutee": 0,
            "position_puissance": [12,12]
        }


    def entry(self, parent, texte, var, row):
        ttk.Label(parent, text=texte).grid(column=0, row=row)
        entry = ttk.Entry(parent, textvariable=self.variables[var])
        entry.grid(column=1, row=row)
        return entry


    def main(self):
        # Température initiale de la plaque
        self.entry(self.frame, "Température initiale de la plaque [°C]", "T_plaque", 0)

        # Température ambiante
        self.entry(self.frame, "Température ambiante [°C]", "T_amb", 1)

        # Coefficient de convection
        self.entry(self.frame, "Coefficient de convection [??]", "h", 2)

        # Puissance appliquée
        self.entry(self.frame, "Puissance appliquée [W]", "P", 3)

        # Boutons pour autres fenêtres
        ttk.Button(self.inter,text = 'Variables de la plaque', command = self.plaque).grid(column=0, row=4)
        ttk.Button(self.inter,text = 'Résolutions de la simulation', command = self.resolution).grid(column=0, row=5)
        ttk.Button(self.inter,text = 'Puissance déposée', command = self.T_dep).grid(column=0, row=6)

        # finish
        ttk.Button(self.inter,text = 'OK', command = self.submit).grid(column=2, row=0)
        ttk.Button(self.inter,text = 'Graphique', command = self.graphique).grid(column=3, row=0)
        self.inter.mainloop()
        

    def plaque(self):
        self.plaque_frame = tk.Toplevel()
        self.plaque_frame.title('Variables de la plaque')
        ttk.Button(self.plaque_frame,text = 'OK', command = self.submit_plaque).grid(column=2, row=0)

        # Dimensions de la plaque
        self.entry(self.plaque_frame, "Longueur en x de la plaque [cm]", "dimx", 0)
        self.entry(self.plaque_frame, "Longueur en y de la plaque [cm]", "dimy", 1)

        # Épaisseur de la plaque
        self.entry(self.plaque_frame, "Épaisseur de la plaque [cm]", "e", 2)

        # Bouton pour autre fenêtre
        ttk.Button(self.plaque_frame,text = 'Paramètres du matériau de la plaque', command = self.mat).grid(column=0, row=3)


    def mat(self):
        self.mat_frame = tk.Toplevel()
        self.mat_frame.title('Paramètres du matériau de la plaque')
        ttk.Button(self.mat_frame,text = 'OK', command = self.submit_mat).grid(column=2, row=0)

        # Densité du matériau
        self.entry(self.mat_frame, "Densité du matériau [kg / m^3]", "rho", 0)

        # Capacité calorifique du matériau
        self.entry(self.mat_frame, "Capacité calorifique du matériau [J / kg.K]", "cp", 1)

        # Conductivité thermique du matériau
        self.entry(self.mat_frame, "Conductivité thermique du matériau [W / m.K]", "k", 2)


    def resolution(self):
        self.reso_frame = tk.Toplevel()
        self.reso_frame.title('Résolutions de la simulation de la plaque')
        ttk.Button(self.reso_frame,text = 'OK', command = self.submit_reso).grid(column=2, row=0)

        # Résolutions de longueur
        self.entry(self.reso_frame, "En x [cm]", "dx", 0)
        self.entry(self.reso_frame, "En y [cm]", "dy", 1)

        # Résolution de temps
        self.entry(self.reso_frame, "Temps [s]", "dt", 2)


    def T_dep(self):
        self.T_dep_frame = tk.Toplevel()
        self.T_dep_frame.title('Contrôle de la puissance déposée')
        ttk.Button(self.T_dep_frame,text = 'OK', command = self.submit_T_dep).grid(column=2, row=0)

        # Température déposée
        self.entry(self.T_dep_frame, "Puissance déposée avec la résistance [W]", "R_depo", 0)

        # Position de la temp
        self.entry(self.T_dep_frame, "Puissance déposée [W]", "T_depo", 2)
        self.entry(self.T_dep_frame, "Position en x de la puissance déposée [cm]", "T_posx", 3)
        self.entry(self.T_dep_frame, "Position en y de la puissance déposée [cm]", "T_posy", 4)


    def sauvegarder_json(self):
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
        self.T_plaque = self.variables["T_plaque"].get()
        self.T_amb=self.variables["T_amb"].get()
        self.h=self.variables["h"].get()
        self.P=self.variables["P"].get()

        # Sauvegarde des données mises à jour dans le JSON
        self.data_fait = {
            "dimensions": [self.dim[0],self.dim[1]],
            "epaisseur": self.e,
            "resolution_x": self.dx,
            "resolution_y": self.dy,
            "resolution_t": self.dt,
            "densite": self.rho,
            "cap_calorifique": self.cp,
            "conduc_thermique": self.k,
            "coef_convection": self.h,
            "T_plaque": self.T_plaque,
            "T_ambiante": self.T_amb,
            "puissance_actuateur": self.P,
            "puissance_R": self.R_depo,
            "puissance_ajoutee": self.T_depo,
            "position_puissance": [self.T_pos[0], self.T_pos[1]]
        }
        self.sauvegarder_json()
        self.Ma_plaque = mega_simulation.Plaque(
            dimensions=(self.dim[0], self.dim[1]),
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
            puissance_actuateur=self.P
            )
        if self.graph == False:
            for n in tqdm(range(1000)):
                for k in range(20): 
                    self.Ma_plaque.iteration()
        else:
            for n in tqdm(range(1000)):
                self.Ma_plaque.show()
                for k in range(20):
                    self.Ma_plaque.iteration()
        self.Ma_plaque.enregistre_rep_echelon()


    def graphique(self):
        self.graph = True
        self.submit()


    def submit_plaque(self):
        self.dim=(self.variables["dimx"].get(), self.variables["dimy"].get())
        self.e=self.variables["e"].get()
        self.plaque_frame.destroy()


    def submit_mat(self):
        self.rho=self.variables["rho"].get()
        self.cp=self.variables["cp"].get()
        self.k=self.variables["k"].get()
        self.mat_frame.destroy()
    

    def submit_reso(self):
        self.dx=self.variables["dx"].get()
        self.dy=self.variables["dy"].get()
        self.dt=self.variables["dt"].get()
        self.reso_frame.destroy()


    def submit_T_dep(self):
        self.R_depo=self.variables["R_depo"].get()
        self.T_depo=self.variables["T_depo"].get()
        self.T_pos=(self.variables["T_posx"].get(), self.variables["T_posy"].get())
        self.T_dep_frame.destroy()


        
Inter= Interface()