import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import json
from datetime import datetime
import tkinter as tk # module de base
from tkinter import ttk # mettre ça beau
import temperature_plaque


class Interface:
    def __init__(self):
        self.inter = tk.Tk()
        self.frame = ttk.Frame(self.inter, padding=10)
        self.frame.grid()
        self.inter.title('Contrôle de la simulation Python')

        # lire json
        self.lire_json()
        
        # Initialisation des variables depuis JSON ou valeurs par défaut (certaines valeurs =! 0)
        self.dim = self.data_lu.get("dimensions", [0.117, 0.061]) if self.data_lu.get("dimensions", [0.117, 0.061]) != 0 else [0.117, 0.061]
        self.e = self.data_lu.get("epaisseur", 0.001) if self.data_lu.get("epaisseur", 0.001) != 0 else 0.001
        self.dx = self.data_lu.get("resolution_x", 0.001) if self.data_lu.get("resolution_x", 0.001) != 0 else 0.001
        self.dy = self.data_lu.get("resolution_y", 0.001) if self.data_lu.get("resolution_y", 0.001) != 0 else 0.001
        self.dt = self.data_lu.get("resolution_t", None)
        self.rho = self.data_lu.get("densite", 2699) if self.data_lu.get("densite", 2699) != 0 else 2699
        self.cp = self.data_lu.get("cap_calorifique", 900.0) if self.data_lu.get("cap_calorifique", 900.0) != 0 else 900.0
        self.k = self.data_lu.get("conduc_thermique", 666.0) if self.data_lu.get("conduc_thermique", 666.0) != 0 else 666.0
        self.h = self.data_lu.get("coef_convection", 20.0) if self.data_lu.get("coef_convection", 20.0) != 0 else 20.0
        self.T_plaque = self.data_lu.get("T_plaque", 25.0)
        self.T_amb = self.data_lu.get("T_ambiante", 23.0) 
        self.T_depo = 0
        self.T_pos = (0,0)
        self.P = self.data_lu.get("puissance_actuateur", 1.5)

        # Initier toutes les entrées 
        self.variables = {key: tk.DoubleVar(value=val) for key, val in {
            "dimx": self.dim[0], "dimy": self.dim[1], "e": self.e, "dx": self.dx,
            "dy": self.dy, "dt": self.dt, "rho": self.rho, "cp": self.cp, 
            "k": self.k, "h": self.h, "T_plaque": self.T_plaque,
            "T_amb": self.T_amb, "T_depo": 0, "T_posx": 0, "T_posy": 0, "P": self.P
        }.items()}


        # Initier variables avec calculs
        self.alpha = self.k/(self.rho*self.cp)
        if self.dt is None:
            self.dt = min(self.dx**2/(4*self.alpha), self.dy**2/(4*self.alpha))  # À regarder!!

        # Go go main interface
        self.main()
    
    def lire_json(self):
        # Rassembler les jsons
        json_files = [f for f in os.listdir() if f.endswith('.json')]
        if not json_files:
            print("Aucun fichier JSON trouvé. Valeurs par défaut utilisées.")
            self.json_de_base()
            self.data_lu = self.data
            return
        
        # Mettre en ordre
        json_files.sort(key=lambda x: x.split('.')[::-1], reverse=True)
        self.nom_json = json_files[0]

        # lire
        try:
            with open(self.nom_json, mode="r") as file:
                self.data_lu = json.load(file)
        except json.JSONDecodeError:
            print(f"Erreur de décodage dans le fichier {self.nom_json}. Valeurs par défaut utilisées.")
            self.json_de_base()
            self.data_lu = self.data


    def json_de_base(self):
        self.data = {
            "dimensions": [0.117,0.061],
            "epaisseur": 0.001,
            "resolution_x": 0.001,
            "resolution_y": 0.001,
            "resolution_t": None,
            "densite": 2699,
            "cap_calorifique": 900,
            "conduc_thermique": 237,
            "coef_convection": 20,
            "T_plaque": 25,
            "T_ambiante": 23,
            "puissance_actuateur": 1.5
        }

    def entry(self, parent, texte, var, row):
        ttk.Label(parent, text=texte).grid(column=0, row=row)
        entry = ttk.Entry(parent, textvariable=self.variables[var])
        entry.grid(column=1, row=row)
        entry.delete(0, "end")
        entry.insert(0, self.variables[var].get())
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
        ttk.Button(self.inter,text = 'Température déposée', command = self.T_dep).grid(column=0, row=6)

        # finish
        ttk.Button(self.inter,text = 'OK', command = self.submit).grid(column=2, row=0)
        ttk.Button(self.inter,text = 'Graphique', command = self.graphique).grid(column=3, row=0)
        self.inter.mainloop()
        

    def plaque(self):
        self.plaque_frame = tk.Toplevel()
        self.plaque_frame.title('Variables de la plaque')
        ttk.Button(self.plaque_frame,text = 'OK', command = self.submit_plaque).grid(column=2, row=0)

        # Dimensions de la plaque
        self.entry(self.plaque_frame, "Longueur en x de la plaque [m]", "dimx", 0)
        self.entry(self.plaque_frame, "Longueur en y de la plaque [m]", "dimy", 1)

        # Épaisseur de la plaque
        self.entry(self.plaque_frame, "Épaisseur de la plaque", "e", 2)

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
        self.entry(self.reso_frame, "En x", "dx", 0)
        self.entry(self.reso_frame, "En y", "dy", 1)

        # Résolution de temps
        self.entry(self.reso_frame, "Temps", "dt", 2)


    def T_dep(self):
        self.T_dep_frame = tk.Toplevel()
        self.T_dep_frame.title('Contrôle de la température déposée')
        ttk.Button(self.T_dep_frame,text = 'OK', command = self.submit_T_dep).grid(column=2, row=0)

        # Température déposée
        self.entry(self.T_dep_frame, "Température déposée [K]", "T_depo", 0)

        # Position de la temp
        self.entry(self.T_dep_frame, "Position en x de la température déposée", "posx", 1)
        self.entry(self.T_dep_frame, "Position en y de la température déposée", "posy", 2)


    def sauvegarder_json(self):
        # Avoir la date dans le format 'dd.mm'
        current_date = datetime.now().strftime('%d.%m')

        extension = '.json'
        i = 1
        new_nom = f'{current_date}.{i}{extension}'
        
        # Regarde si fichier existe déjà, si oui ajoute 1 au nombre final
        while os.path.exists(new_nom):
            i += 1
            new_nom = f'{current_date}.{i}{extension}'
        
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
            "puissance_actuateur": self.P
        }
        self.sauvegarder_json()
        self.Ma_plaque = temperature_plaque.Plaque(
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


    def graphique(self):
        self.submit()
        #plt.ion()
        #start = time.time()
        for n in tqdm(range(10)):
            self.Ma_plaque.show()
            for k in range(85): # Vérifie que cette boucle tourne aussi
                self.Ma_plaque.iteration()

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
        self.T_depo=self.variables["T_depo"].get()
        self.T_pos=(self.variables["T_posx"].get(), self.variables["T_posy"].get())
        self.T_dep_frame.destroy()


        
Inter= Interface()