import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

temperatures = [19,24.5,30]
gain = [2.3,3.88,5.4]

temperatures = np.array(temperatures)#+273.15 # En Kelvin
gain = np.array(gain)

def linear(x, a, b):
    return x*a+b

res, cov = curve_fit(linear, temperatures, gain)
uncertainties = np.sqrt(np.diag(cov))
# temp_theoriques = np.linspace(283.15, 313.15, 1000)
temp_theoriques = np.linspace(18, 32, 1000)

print(f"Parameters:")
for i in range(len(res)):
    print(f"Param #{i+1} = {res[i]} +/- {uncertainties[i]}")
ax1 = plt.subplot(111)
plt.plot(temperatures, gain, "o", label="donnÃ©es")
plt.plot(temp_theoriques, linear(temp_theoriques, *res), label="fit")
plt.xlabel(r"$T \mathrm{[K]}$", fontsize=16)
plt.ylabel(r"Gain", fontsize=16)
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\arduino_data_test_echelon_vontilateur.csv"
# # fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\arduino_data_marylise2.csv"
# fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rapportV2\output.csv"

# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_25_degre.csv"
# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_25_degre_perturbation.csv"
# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_20_degre_perturbation.csv"
# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_30_degre_perturbation.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_30_degre.csv"
# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_30_degre.csv"
# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\arduino_data_marylise2.csv"

fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\Design2\output.csv"#Pour rouler vite
fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Python_30_degre.csv"
# fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Simul_20_degre.csv"
# fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Simul_30_degre.csv"

fich_matl = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Matlab_30_degre.csv"


data = pd.read_csv(fich, encoding="latin1", sep=";") #pour tests 25 perturb30
# data = pd.read_csv(fich, encoding="latin1", sep=",") #pour tests 20
data_sim = pd.read_csv(fich_sim)
data_matl = pd.read_csv(fich_matl)
column_names = data.columns
column_names_sim = data_sim.columns
column_names_matl = data_matl.columns
print(column_names_matl)
print(column_names)

## Python
plt.plot(data_sim["0"],(data_sim["2"]-data_sim["2"][0]),color="red", label="Simulateur")
plt.plot(data_sim["0"],(data_sim["3"]-data_sim["3"][0]),color="red")
plt.plot(data_sim["0"],(data_sim["4"]-data_sim["4"][0]),color="red")
t_py=data_sim["0"]
Th1_py=data_sim["2"]-data_sim["2"][0]
Th2_py=data_sim["3"]-data_sim["3"][0]
Th3_py=data_sim["4"]-data_sim["4"][0]
# plt.show()

##Matlab
plt.plot(data_matl['Temps'],(data_matl['Temperature_a_lactuateur']-data_matl['Temperature_a_lactuateur'][0]),color="green", label="Simulateur")
plt.plot(data_matl['Temps'],(data_matl['Temperature_au_milieu']-data_matl['Temperature_au_milieu'][0]),color="green")
plt.plot(data_matl['Temps'],(data_matl['Temperature_au_laser']-data_matl['Temperature_au_laser'][0]),color="green")
t_mat=data_matl['Temps']
Th1_mat=data_matl['Temperature_a_lactuateur']-data_matl['Temperature_a_lactuateur'][0]
Th2_mat=data_matl['Temperature_au_milieu']-data_matl['Temperature_au_milieu'][0]
Th3_mat=data_matl['Temperature_au_laser']-data_matl['Temperature_au_laser'][0]

# # Test_25_degre
# plt.plot(data["Temps"][614:]-data["Temps"][614],data["T1"][614:]-data["T1"][614], color="blue", label="Prototype")
# plt.plot(data["Temps"][614:]-data["Temps"][614],data["T2"][614:]-data["T2"][614], color="blue")
# plt.plot(data["Temps"][614:]-data["Temps"][614],data["T3"][614:]-data["T3"][614], color="blue")
# t=data["Temps"][614:]-data["Temps"][614]
# Th1=data["T1"][614:]-data["T1"][614]
# Th2=data["T2"][614:]-data["T2"][614]
# Th3=data["T3"][614:]-data["T3"][614]
# # Test_25_degre_perturbation
# plt.plot(data["Temps"][116:]-data["Temps"][116],data["T1"][116:]-data["T1"][116], color="blue", label="Prototype")
# plt.plot(data["Temps"][116:]-data["Temps"][116],data["T2"][116:]-data["T2"][116], color="blue")
# plt.plot(data["Temps"][116:]-data["Temps"][116],data["T3"][116:]-data["T3"][116], color="blue")
# # Test_20_degre_perturbation
# plt.plot(data["Temps"][529:]-data["Temps"][529],data["T1"][529:]-data["T1"][529], color="blue", label="Prototype")
# plt.plot(data["Temps"][529:]-data["Temps"][529],data["T2"][529:]-data["T2"][529], color="blue")
# plt.plot(data["Temps"][529:]-data["Temps"][529],data["T3"][529:]-data["T3"][529], color="blue")
# # Test_30_degre_perturbation
# plt.plot(data["Temps"][1213:]-data["Temps"][1213],data["T1"][1213:]-data["T1"][1213], color="blue", label="Prototype")
# plt.plot(data["Temps"][1213:]-data["Temps"][1213],data["T2"][1213:]-data["T2"][1213], color="blue")
# plt.plot(data["Temps"][1213:]-data["Temps"][1213],data["T3"][1213:]-data["T3"][1213], color="blue")
# # Test_20_degre
# plt.plot(data["Temps"][801:]-data["Temps"][801],data["T1"][801:]-data["T1"][801], color="blue", label="Prototype")
# plt.plot(data["Temps"][801:]-data["Temps"][801],data["T2"][801:]-data["T2"][801], color="blue")
# plt.plot(data["Temps"][801:]-data["Temps"][801],data["T3"][801:]-data["T3"][801], color="blue")
# t=data["Temps"][801:]-data["Temps"][801]
# Th1=data["T1"][801:]-data["T1"][801]
# Th2=data["T2"][801:]-data["T2"][801]
# Th3=data["T3"][801:]-data["T3"][801]
# # Test_30_degre
plt.plot(data["Temps"][468:]-data["Temps"][468],data["T1"][468:]-data["T1"][468], color="blue", label="Prototype")
plt.plot(data["Temps"][468:]-data["Temps"][468],data["T2"][468:]-data["T2"][468], color="blue")
plt.plot(data["Temps"][468:]-data["Temps"][468],data["T3"][468:]-data["T3"][468], color="blue")
t=data["Temps"][468:]-data["Temps"][468]
Th1=data["T1"][468:]-data["T1"][468]
Th2=data["T2"][468:]-data["T2"][468]
Th3=data["T3"][468:]-data["T3"][468]

plt.xlabel("Temps (s)")
plt.ylabel("Variation de température (°C)")
plt.legend()
plt.show()




def trace(t, Th1, Th2, Th3):
    plt.plot(t, Th1, color="gold", label="Thermistance 1")
    plt.plot(t, Th2, color="darkorange", label="Thermistance 2")
    plt.plot(t, Th3, color="red", label="Thermistance 3")
    plt.xlabel("Temps (s)")
    plt.ylabel("Variation de température (°C)")
    plt.legend()
    plt.show()

trace(t_py, Th1_py, Th2_py, Th3_py)
trace(t_mat, Th1_mat, Th2_mat, Th3_mat)


# Faire une liste de liste : 1ere liste = simulation, 2e liste = prototype
liste_rep = [[t_py, Th1_py, Th2_py, Th3_py],[t, Th1, Th2, Th3]]
# liste_rep = [[t_mat, Th1_mat, Th2_mat, Th3_mat],[t, Th1, Th2, Th3]]

from scipy.interpolate import CubicSpline

def compare_echelon(liste_rep, pt_op):
    ax1 = plt.subplot(211)
    ax2= plt.subplot(212, sharex=ax1)

    ax1.plot(liste_rep[0][0], liste_rep[0][1], color="blue", label="Thermistance 1")
    ax1.plot(liste_rep[0][0], liste_rep[0][2], color="green", label="Thermistance 2")
    ax1.plot(liste_rep[0][0], liste_rep[0][3], color="red", label="Thermistance 3")

    ax1.plot(liste_rep[1][0], liste_rep[1][1], color="blue", linestyle="dotted", linewidth=3)
    ax1.plot(liste_rep[1][0], liste_rep[1][2], color="green", linestyle="dotted", linewidth=3)
    ax1.plot(liste_rep[1][0], liste_rep[1][3], color="red", linestyle="dotted", linewidth=3)

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

compare_echelon(liste_rep, 24.5)





def make_same_length(array1, array2):
    """make array1 the same length as array2, where one of the arrays is undersampled"""
    length_ratio = int(len(array1)/len(array2)) if len(array1) > len(array2) else int(len(array2)/len(array1))
    new_array = []
    if len(array1) > len(array2):
        for i in range(len(array2)):
            for k in range(length_ratio):
                new_array.append(array2[i])
        return array1, new_array
    for i in range(len(array1)):
        for k in range(length_ratio):
            new_array.append(array1[i])
    return new_array, array2