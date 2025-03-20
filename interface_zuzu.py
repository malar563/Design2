import serial # Communication avec l'Arduino
import time   # Gestion des pauses et du temps
import tkinter as tk # Création de l'interface graphique
from tkinter import ttk
import matplotlib.pyplot as plt # Traçage des courbes de température
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque # Stockage des données
 
# Notes: (dans l'ordre)
# Décommenter partie 'Connexion Arduino'
# Ajout de la ligne 'self.read_serial()'
# Ajout des boutons 'Démarrer/Arrêter' + nouvel affichage pour les deux
# Décommenter 'self.arduino.write...'
# Ajout de 3 fonctions 'start_arduino, stop_arduino, read_serial'
 
class Interface_Arduino():
    def __init__(self, name):
        self.name = name
        self.boucle_fermée = False
        self.modification = True
 
        # Création de la fenêtre Tkinter
        self.inter = tk.Tk() # Crée une fenêtre princiaple Tkinter
        self.frame = ttk.Frame(self.inter, padding=10)
        self.frame.grid()
        self.inter.title("Contrôle de l'Arduino")
 
        # Connexion à l'Arduino
        self.arduino = serial.Serial(self.name, 115200, timeout=1.0)
        time.sleep(2)  # Pause pour l'initialisation de l'Arduino
 
        self.start_time = time.time() # Moment où le programme commence
 
        # Lancement de l'interface Tkinter
        self.initialisation_variables()
        self.initialisation_graph()
        self.read_serial() # Afficher la fenêtre Arduino dans python
        self.frame.after(5000, self.update_asservissement)
        self.frame.after(2000, self.update_graph)
        self.inter.mainloop()
   
    def entry(self, parent, texte, var, row):
        ttk.Label(parent, text=texte).grid(column=0, row=row)
        entry = ttk.Entry(parent, textvariable=self.variables[var])
        entry.grid(column=1, row=row)
        return entry
 
    def initialisation_variables(self):
        # valeurs de base des variables
        self.consigne = 25 # degrées Celcius
        self.gain_reg = 0
        self.cte_temps_reg = 0
        self.asser = False
 
        # initialiser les variables dans l'interface
        self.variables = {key: tk.DoubleVar(value=val) for key, val in {
            "consigne": self.consigne,
            "gain_reg": self.gain_reg,
            "cte_temps_reg": self.cte_temps_reg
        }.items()} 
 
        # Bouton pour fermer l'interface
        ttk.Button(self.frame,text = 'Quitter', command = self.quitter_programme).grid(column=2, row=1)
 
        # Statut de l'asservissement
        self.asser_var = tk.StringVar(value="Système non asservi")
        self.asser_label = tk.Label(self.frame, textvariable=self.asser_var, bg='yellow')
        self.asser_label.grid(column=0, row=2, columnspan=2)
 
         # Message d'asservissement démarrer/arrêter
        self.message_asservissement_var = tk.StringVar(value="Prêt à démarrer.")
        tk.Label(self.frame, textvariable = self.message_asservissement_var, bg='orange').grid(column=0, row=3, columnspan=2)
 
        # Bouton pour démarrer l'Arduino
        ttk.Button(self.frame, text = 'Démarrer', command = self.start_arduino).grid(column=3, row=0)
       
        # Bouton pour arrêter l'Arduino
        ttk.Button(self.frame, text = 'Arrêter', command = self.stop_arduino).grid(column=3, row=1)
 
 
    def initialisation_graph(self):
        """
        Initialisation des données et de la figure
        """
        # Données pour le graphique
        longueur_max = 50 # Stocke au max ?? points sur le graphique
        self.x_data = deque(maxlen=longueur_max)
        self.T1_data = deque(maxlen=longueur_max)
        self.T2_data = deque(maxlen=longueur_max)
        self.T3_data = deque(maxlen=longueur_max)
        self.T3_esti_data = deque(maxlen=longueur_max)

        # Création de la figure Matplotlib
        self.fig, self.ax = plt.subplots()
        
        # Créer une nouvelle fenêtre pour le graphique
        self.graph_window = tk.Toplevel(self.inter)  # Crée une fenêtre secondaire
        self.graph_window.title("Graphique des Températures")  # Donne un titre à la fenêtre

        # Crée le widget canvas pour le graphique et l'ajoute à la nouvelle fenêtre
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_window)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initialise les lignes dans le graph
        self.T1, = self.ax.plot([], [], 'o-', label="Température à l'actuateur")
        self.T2, = self.ax.plot([], [], 'o-', label="Température au milieu")
        self.T3, = self.ax.plot([], [], 'o-', label="Température au laser")
        self.T3_esti, = self.ax.plot([], [], 'o-', label="Température estimée au laser")

        # axes et titre
        self.ax.set_xlabel("Temps [s]") 
        self.ax.set_ylabel("Température [°C]")
        self.ax.set_title("Températures au fil du temps")
        self.ax.grid(True)
        self.ax.legend()
 
 
    def send_variables(self):
        """
        Fonction pour envoyer les variables à l'Arduino À CHANGER
        """
        time.sleep(1)
        self.boucle_fermée = True
        try:
            self.consigne=self.variables["consigne"].get()
            self.gain_reg=self.variables["gain_reg"].get()
            self.cte_temps_reg=self.variables["cte_temps_reg"].get()
            self.arduino.write(f"{self.modification}\n".encode())
            self.arduino.write(f"{self.consigne}\n".encode())
            self.arduino.write(f"{self.gain_reg}\n".encode())  
            self.arduino.write(f"{self.cte_temps_reg}\n".encode())  
            self.arduino.write(f"{self.boucle_fermée}\n".encode())  
            self.statut_consigne.set(f"Consigne envoyée: {self.consigne} [°C]")
            self.statut_gain_reg.set(f"Gain du régulateur envoyé: {self.gain_reg} [-]")
            self.statut_cte_temps_reg.set(f"Constante de temps du régulateur envoyée: {self.cte_temps_reg} [1/s]")
            self.message_asservissement_var.set("L'asservissement est actif.")
        except:
            self.statut_consigne.set("Entrée invalide !")
            self.statut_gain_reg.set("")
            self.statut_cte_temps_reg.set("")
            self.asser_var.set("Erreur d'envoi !")
 
 
    def quitter_programme(self):
        """
        Fonction pour quitter
        """
        #self.arduino.close()  # Ferme la connexion série
        plt.close('all')  # Ferme toutes les figures Matplotlib
        self.inter.destroy() # Détruit la fenêtre
 
 
    def start_arduino(self):
        """
        Envoie la commande pour démarrer l'Arduino, boucle fermée
        """
        self.asser_frame = tk.Toplevel()
        self.asser_frame.title("Contrôle de l'asservissement")

        # Champ pour la consigne
        self.entry(self.asser_frame, "Température désirée [°C] :", 'consigne', 0)

        # Champ pour le gain du régulateur
        self.entry(self.asser_frame, "Gain du régulateur [-] :", 'gain_reg', 1)

        # Champ pour la constante de temps du régulateur
        self.entry(self.asser_frame, "Constante de temps du régulateur [1/s] :", 'cte_temps_reg', 2)
 
        # Statut de l'envoie des variables
        self.statut_consigne = tk.StringVar(value="En attente de variables...")
        ttk.Label(self.asser_frame, textvariable=self.statut_consigne).grid(column=0, row=3, columnspan=2)

        self.statut_gain_reg = tk.StringVar(value="")
        ttk.Label(self.asser_frame, textvariable=self.statut_gain_reg).grid(column=0, row=4, columnspan=2)

        self.statut_cte_temps_reg = tk.StringVar(value="")
        ttk.Label(self.asser_frame, textvariable=self.statut_cte_temps_reg).grid(column=0, row=5, columnspan=2)
 
        # Bouton pour envoyer les variables
        ttk.Button(self.asser_frame,text = 'Envoyer', command = self.send_variables).grid(column=1, row=6)
 
 
    def stop_arduino(self):
        """
        Envoie la commande pour arrêter l'Arduino, boucle ouverte
        """
        self.boucle_fermée = False
        try:
            self.arduino.write(f"{self.modification}\n".encode())
            self.arduino.write(f"{self.boucle_fermée}\n".encode())
            # self.asser_var.set("Système non asservi")
            self.message_asservissement_var.set("L'asservissement est désactivé.")
        except:
            self.asser_var.set("Erreur d'envoi !")
 
    def read_serial(self):
        """
        Fonction qui lit les données série et les affiche
        """
        if self.arduino.in_waiting > 0:
            message = self.arduino.readline().decode().strip()
            print("Arduino:", message)
 
        if self.inter.winfo_exists():
            self.inter.after(100, self.read_serial) # Planifie un nouvel appel à cette fonction dans 100 ms
 
    def update_graph(self):
        line = self.arduino.readline().decode().strip()  # Lire les données série
        if "T3: " in line:
            try:
                temp1 = float(line.split("T1: ")[1].split("|")[0])
                temp2 = float(line.split("T2: ")[1].split("|")[0])
                temp3 = float(line.split("T3: ")[1].split("|")[0])
                temp3_esti = float(line.split("T3 estimée: ")[1].split("|")[0])
                temps = float(line.split("Temps: ")[1].split("|")[0])
                
                # Mise à jour des données
                self.x_data.append(temps)
                self.T1_data.append(temp1)
                self.T2_data.append(temp2)
                self.T3_data.append(temp3)
                self.T3_esti_data.append(temp3_esti)

                # Met à jour les données des courbes sans effacer les axes
                self.T1.set_data(self.x_data, self.T1_data)
                self.T2.set_data(self.x_data, self.T2_data)
                self.T3.set_data(self.x_data, self.T3_data)
                self.T3_esti.set_data(self.x_data, self.T3_esti_data)
                
                # Met à jour les limites des axes si nécessaire
                self.ax.relim()  # Recalcule les limites des axes
                self.ax.autoscale_view()  # Ajuste les limites de vue

                self.canvas.draw()  # Redessine le graphique
            except ValueError:
                pass  # Ignore les erreurs de conversion
        self.frame.after(2000, self.update_graph)  # Répète la mise à jour

 
    def update_asservissement(self):
        if self.T3_data[-1] > (self.consigne-0.1) and self.T3_data[-1] < (self.consigne+0.1):
            if self.asser is True:
                self.asser_var.set("Système asservi")
                self.asser_label.config(bg='lightgreen')
            self.asser = True
        else:
            if self.asser is False:
                self.asser_var.set("Système non asservi")
                self.asser_label.config(bg='yellow')
            self.asser = False
        self.frame.after(5000, self.update_asservissement)  # Répète la mise à jour
 
 
Inter_Arduino = Interface_Arduino('COM3') # changer pour 'COM3' si sur Windows sinon pour MAC '/dev/cu.usbserial-14110'
 