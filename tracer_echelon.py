import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

temperatures = [19,24.5,30]
gain = [0.23,0.388,0.54]

temperatures = np.array(temperatures)+273.15 # En Kelvin
gain = np.array(gain)

def linear(x, a, b):
    return x*a+b

res, cov = curve_fit(linear, temperatures, gain)
uncertainties = np.sqrt(np.diag(cov))
temp_theoriques = np.linspace(283.15, 313.15, 1000)

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

#fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_25_degre.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_25_degre_perturbation.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_20_degre_perturbation.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_30_degre_perturbation.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_20_degre.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_30_degre.csv"
# fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\arduino_data_marylise2.csv"
fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\Design2\output.csv"

data = pd.read_csv(fich, encoding="latin1", sep=";") #pour tests 25 perturb30
# data = pd.read_csv(fich, encoding="latin1", sep=",") #pour tests 20
data_sim = pd.read_csv(fich_sim)
column_names = data.columns
column_names_sim = data_sim.columns
# print(column_names_sim)

print(column_names)

plt.plot(data_sim["0"],(data_sim["2"]-data_sim["2"][0]),color="red", label="Simulateur")
plt.plot(data_sim["0"],(data_sim["3"]-data_sim["3"][0]),color="red")
plt.plot(data_sim["0"],(data_sim["4"]-data_sim["4"][0]),color="red")
# plt.show()

# # Test_25_degre
# plt.plot(data["Temps"][614:]-data["Temps"][614],data["T1"][614:]-data["T1"][614], color="blue", label="Prototype")
# plt.plot(data["Temps"][614:]-data["Temps"][614],data["T2"][614:]-data["T2"][614], color="blue")
# plt.plot(data["Temps"][614:]-data["Temps"][614],data["T3"][614:]-data["T3"][614], color="blue")
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
# # Test_30_degre_perturbation
# plt.plot(data["Temps"][801:]-data["Temps"][801],data["T1"][801:]-data["T1"][801], color="blue", label="Prototype")
# plt.plot(data["Temps"][801:]-data["Temps"][801],data["T2"][801:]-data["T2"][801], color="blue")
# plt.plot(data["Temps"][801:]-data["Temps"][801],data["T3"][801:]-data["T3"][801], color="blue")
# Test_30_degre_perturbation
plt.plot(data["Temps"][468:]-data["Temps"][468],data["T1"][468:]-data["T1"][468], color="blue", label="Prototype")
plt.plot(data["Temps"][468:]-data["Temps"][468],data["T2"][468:]-data["T2"][468], color="blue")
plt.plot(data["Temps"][468:]-data["Temps"][468],data["T3"][468:]-data["T3"][468], color="blue")




#echelon ventilateur
# plt.plot(data["Temps (s)"],data["T1"]-data["T1"][0], color="blue", label="Prototype")
# plt.plot(data["Temps (s)"],data["T2"]-data["T2"][0], color="blue")
# plt.plot(data["Temps (s)"],data["T3"]-data["T3"][0], color="blue")
# plt.plot(data["Temps"][6101:]-61,data["T1"][6101:]-data["T1"][6101])
# plt.plot(data["Temps"][6101:]-61,data["T2"][6101:]-data["T2"][6101])
# plt.plot(data["Temps"][6101:]-61,data["T3"][6101:]-data["T3"][6101])
plt.xlabel("Temps (s)")
plt.ylabel("Variation de température (°C)")
plt.legend()
plt.show()


