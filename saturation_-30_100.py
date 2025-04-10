import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def I_quadratique(I, a, b):
    return a*I**2 + b*I

I_associee = [(-2.02152-(-0.020491803)), (-1.347131148-(-0.860983607)),(-0.814188-(-1.024590164)), (0.18278689-(-0.020491803)), (0.89344262-(0.621311475)), (2.00292-(-0.020491803))]#0.1, 0.09
P_deposee = [-2.1, -0.5, 0.23, 0.34, 0.54, 3.6]


# p0 = [np.max(pic_bleu), px_blue[np.argmax(pic_bleu)], 20, 30]
params, _ = curve_fit(I_quadratique, I_associee, P_deposee)#, p0=p0
a, b = params
print("Coefficient a :", a)
print("Coefficient b :", b)
# Graphique
x = np.linspace(-2,2,1000)
plt.plot(x,I_quadratique(x, a, b), color="black", label = r"Évolution en $1/\sqrt{n}$")
plt.plot(I_associee, P_deposee, color="r", label = "Écart-type des valeurs hautes")
plt.xlabel("Nombre de données n")
plt.ylabel("Écart-type (V)")
plt.legend() 
plt.show()


fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_consigne_-30_degre.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_consigne_100_degre.csv"
fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\rep_echelon\Test_echelon_froid_20_degré.csv"
fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\Design2\output.csv"


# data = pd.read_csv(fich, encoding="latin1", sep=";") #pour tests 25 perturb30
data = pd.read_csv(fich, encoding="latin1", sep=",") #pour tests 20
# data = pd.read_csv(fich, sep=';', encoding="latin1")
data_sim = pd.read_csv(fich_sim)
column_names = data.columns
column_names_sim = data_sim.columns
print(column_names)

## Python
plt.plot(data_sim["0"],(data_sim["2"]-data_sim["2"][0]),color="red", label="Simulateur")
plt.plot(data_sim["0"],(data_sim["3"]-data_sim["3"][0]),color="red")
plt.plot(data_sim["0"],(data_sim["4"]-data_sim["4"][0]),color="red")
t_py=data_sim["0"]

# # # Test_-30degre
# plt.plot(data["Temps"][13:]-data["Temps"][13],data["T1"][13:]-data["T1"][13], color="blue", label="Prototype")
# plt.plot(data["Temps"][13:]-data["Temps"][13],data["T2"][13:]-data["T2"][13], color="blue")
# plt.plot(data["Temps"][13:]-data["Temps"][13],data["T3"][13:]-data["T3"][13], color="blue")
# # # Test_100degre
# plt.plot(data["Temps"][10:]-data["Temps"][10],data["T1"][10:]-data["T1"][10], color="blue", label="Prototype")
# plt.plot(data["Temps"][10:]-data["Temps"][10],data["T2"][10:]-data["T2"][10], color="blue")
# plt.plot(data["Temps"][10:]-data["Temps"][10],data["T3"][10:]-data["T3"][10], color="blue")
# # # Test_20degre froid
plt.plot(data["Temps"][137:]-data["Temps"][137],data["T1"][137:]-data["T1"][137], color="blue", label="Prototype")
plt.plot(data["Temps"][137:]-data["Temps"][137],data["T2"][137:]-data["T2"][137], color="blue")
plt.plot(data["Temps"][137:]-data["Temps"][137],data["T3"][137:]-data["T3"][137], color="blue")



plt.xlabel("Temps (s)")
plt.ylabel("Variation de température (°C)")
plt.legend()
plt.show()