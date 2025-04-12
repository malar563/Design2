import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# À changer pour vous
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_30_degre.csv"
fich_matl = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Matlab_30_degre.csv"

# Lire les fichiers
data = pd.read_csv(fich, encoding="latin1", sep=";") # Marche habituellement
# data = pd.read_csv(fich, encoding="latin1", sep=",") # Si , au lieu de ;
data_matl = pd.read_csv(fich_matl)

# Pour voir les noms de colonne (à changer plus bas si différents)
column_names = data.columns
column_names_matl = data_matl.columns
print(column_names_matl)
print(column_names)

# Matlab : Devrait être ok, ne pas toucher
t_mat=data_matl['Temps']
Th1_mat=data_matl['Temperature_a_lactuateur']-data_matl['Temperature_a_lactuateur'][0]
Th2_mat=data_matl['Temperature_au_milieu']-data_matl['Temperature_au_milieu'][0]
Th3_mat=data_matl['Temperature_au_laser']-data_matl['Temperature_au_laser'][0]

# Test : Mettre les indices de où ça commence et finit, changer les noms de colonne si nécessaire
t=data["Temps"][468:]-data["Temps"][468]
Th1=data["T1"][468:]-data["T1"][468]
Th2=data["T2"][468:]-data["T2"][468]
Th3=data["T3"][468:]-data["T3"][468]

# Réponse en boucle fermé du simulateur
def trace(t, Th1, Th2, Th3, pt_op):
    # Rajouter une colonne pour la commande u?
    plt.plot(t, Th1+pt_op, color="gold", label="Thermistance 1")
    plt.plot(t, Th2+pt_op, color="darkorange", label="Thermistance 2")
    plt.plot(t, Th3+pt_op, color="red", label="Thermistance 3")
    plt.xlabel("Temps (s)")
    plt.ylabel("Variation de température (°C)")
    plt.legend()
    plt.show()

# TRÈS IMPORTANT : CHANGER LE POINT D'OPÉRATION ICI!!
trace(t_mat, Th1_mat, Th2_mat, Th3_mat, 30)


# Comparer prototype et simulateur 
from scipy.interpolate import CubicSpline
liste_rep = [[t_mat, Th1_mat, Th2_mat, Th3_mat],[t, Th1, Th2, Th3]]

def compare_echelon(liste_rep, pt_op):
    ax1 = plt.subplot(211)
    ax2= plt.subplot(212, sharex=ax1)

    ax1.plot(liste_rep[0][0], liste_rep[0][1]+pt_op, color="blue", label="Thermistance 1")
    ax1.plot(liste_rep[0][0], liste_rep[0][2]+pt_op, color="green", label="Thermistance 2")
    ax1.plot(liste_rep[0][0], liste_rep[0][3]+pt_op, color="red", label="Thermistance 3")

    ax1.plot(liste_rep[1][0], liste_rep[1][1]+pt_op, color="blue", linestyle="dotted", linewidth=3)
    ax1.plot(liste_rep[1][0], liste_rep[1][2]+pt_op, color="green", linestyle="dotted", linewidth=3)
    ax1.plot(liste_rep[1][0], liste_rep[1][3]+pt_op, color="red", linestyle="dotted", linewidth=3)

    # ax1.plot(t, cooling_law(t, params[0], params[1], params[2])-273.15, color="red", linestyle="dotted", linewidth=3, label="curve_fit")
    ax1.set_ylabel("Variation de température [°C]")
    ax1.legend()

    spline_sim0 = CubicSpline(liste_rep[0][0],liste_rep[0][3]+pt_op)
    ax2.plot(liste_rep[1][0],((liste_rep[1][3]+pt_op)-(spline_sim0(liste_rep[1][0])))/(liste_rep[1][3]+pt_op)*100, color="black", label="Erreur")
    ax2.axhline(0, 0, 1400, color="red", linewidth=2, linestyle="--")
    ax2.set_ylabel("Pourcentage d'écart à T3 [%]")
    ax2.set_xlabel("Temps [s]")
    ax2.legend()

    plt.show()

# TRÈS IMPORTANT : CHANGER LE POINT D'OPÉRATION ICI!!
compare_echelon(liste_rep, 30)






