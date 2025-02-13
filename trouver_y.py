import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


fich = r"C:\Users\maryl\Documents\Universite\Session_4\design2\arduino_data_test_marylise.csv"
fich_sim = r"C:\Users\maryl\Documents\Universite\Session_4\design2\Design2\output.csv"

data = pd.read_csv(fich, sep=';')
data_sim = pd.read_csv(fich_sim)
column_names = data.columns
column_names_sim = data_sim.columns
print(column_names_sim)

plt.plot(data_sim["0"],data_sim["2"])
plt.plot(data_sim["0"],data_sim["3"])
plt.plot(data_sim["0"],data_sim["4"])
# plt.plot(t2,Therm_2)
# plt.plot(t3,Therm_3)
plt.show()

plt.plot(data["Temps (s)"],data["T_celsius1"])
plt.plot(data["Temps (s)"],data["T_celsius2"])
plt.plot(data["Temps (s)"],data["T_celsius3"])
plt.show()


data_filtered1 = data[data["Temps (s)"] > 120]
data_filtered2 = data[data["Temps (s)"] > 145]
data_filtered3 = data[data["Temps (s)"] > 165]

# Extraire les nouvelles colonnes filtrées
t1 = data_filtered1["Temps (s)"].values
t2 = data_filtered2["Temps (s)"].values
t3 = data_filtered3["Temps (s)"].values
Therm_1 = (data_filtered1["T_celsius1"]+273.15).values
Therm_2 = ((data_filtered2["T_celsius2"]+273.15).values)#[1210:]
Therm_3 = ((data_filtered3["T_celsius3"]+273.15).values)#[4770:]

# print(data)
plt.plot(t1,Therm_1)
plt.plot(t2,Therm_2)
plt.plot(t3,Therm_3)
plt.show()


T_0 = Therm_1[1]
T_02 = Therm_2[1]
T_03 = Therm_3[1]


def cooling_law(t, h, t0, T_air):
    #T_air = 20+273.15
    Aire = (2*0.06*0.116) + (0.001*2*0.116) + (0.001*2*0.06)
    c_p = 897
    rho = 2698.9
    Volume = 0.06*0.116*0.001
    exposant = -(h*Aire)/(rho*c_p*Volume) * (t-t0)
    return (T_air) + (T_0 - T_air)*np.exp(exposant)

def cooling_law2(t, t0, T_air):
    #T_air = 20+273.15
    h = 30.00172516
    Aire = (2*0.06*0.116) + (0.001*2*0.116) + (0.001*2*0.06)
    c_p = 897
    rho = 2698.9
    Volume = 0.06*0.116*0.001
    exposant = -(h*Aire)/(rho*c_p*Volume) * (t-t0)
    return (T_air) + (T_0 - T_air)*np.exp(exposant)

# T1
# liste_h = []
# for t, Therm in [(t1, Therm_1), (t2, Therm_2), (t3, Therm_3)]:
#     params, _ = curve_fit(cooling_law, t, Therm, p0=[30.00170822, 100, 280])
#     print(params)

#     plt.plot(t, Therm)
#     plt.plot(t, cooling_law(t, params[0], params[1], params[2]))
#     plt.show()

for t, Therm in [(t1, Therm_1), (t2, Therm_2), (t3, Therm_3)]:
    params, _ = curve_fit(cooling_law2, t, Therm, p0=[100, 280])
    print(params)

    plt.plot(t, Therm)
    plt.plot(t, cooling_law2(t, params[0], params[1]))
    plt.show()


