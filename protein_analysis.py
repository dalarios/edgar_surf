import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from skimage import io
import glob
import csv

# Initialize global variables
meanIntensity_List = []
proteinConcentration_List = []
proteinConcentration_nM_List = []
numberOfProteinMolecules_List = []
rateOfChangeProteinMolecules_List = []
timeValues_List = []
optimizedParameters = []

def calculateMeanIntensity(paths):
    global meanIntensity_List
    meanIntensity_List = []
    for path in paths: 
        image_matrix = io.imread(path)
        meanIntensity = image_matrix.mean()
        meanIntensity_List.append(meanIntensity)

def getConcentration(calibrationCurvePaths, mw_kda):
    global proteinConcentration_List, proteinConcentration_nM_List
    proteinConcentration_List = []
    proteinConcentration_nM_List = []

    meanIntensity_CalibrationCurve_List = []
    for path in calibrationCurvePaths:
        image_matrix = io.imread(path)
        meanIntensity = image_matrix.mean()
        meanIntensity_CalibrationCurve_List.append(meanIntensity) 

    df = pd.DataFrame(meanIntensity_CalibrationCurve_List).reset_index()
    df = df.rename(columns={"index": "Protein Concentration (ng/µl)", 0: "Mean Intensity"})
    sampleConcentration_Values = [0, 2, 5, 10, 20, 40, 80, 160, 320]
    df["Protein Concentration (ng/µl)"] = sampleConcentration_Values

    x = df["Protein Concentration (ng/µl)"]
    y = df["Mean Intensity"]

    slope, intercept = np.polyfit(x, y, 1)
    line_of_best_fit = slope * x + intercept

    plt.figure(figsize=(10, 6))
    plt.plot(df["Protein Concentration (ng/µl)"], df["Mean Intensity"], marker='o', linestyle='none', label='Data points')
    plt.plot(x, line_of_best_fit, label=f'Line of Best Fit: y = {slope:.2f}x + {intercept:.2f}', color='red')
    plt.title('Mean Intensity vs Protein Concentration')
    plt.xlabel('Protein Concentration (ng/µl)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.legend()
    plt.show()

    for meanIntensity in meanIntensity_List:
        proteinConcentration = (meanIntensity - intercept) / slope
        proteinConcentration_List.append(proteinConcentration)
        proteinConcentration_nM = ((proteinConcentration * 1e-3) / (mw_kda * 1e3)) * 1e9
        proteinConcentration_nM_List.append(proteinConcentration_nM)

def constructDataFrames(timeInterval, time_unit='seconds'):
    global meanIntensity_List, proteinConcentration_List, proteinConcentration_nM_List, df

    # Convert time interval to seconds if necessary
    if time_unit == 'minutes':
        timeInterval *= 60

    minimumIntensityValue = min(meanIntensity_List)
    meanIntensity_List = [x - minimumIntensityValue for x in meanIntensity_List]

    minimumProteinConcentration = min(proteinConcentration_List)
    proteinConcentration_List = [x - minimumProteinConcentration for x in proteinConcentration_List]

    minimumProteinConcentration_nM = min(proteinConcentration_nM_List)
    proteinConcentration_nM_List = [x - minimumProteinConcentration_nM for x in proteinConcentration_nM_List]

    df = pd.DataFrame({
        "Time (s)": [i * timeInterval for i in range(len(meanIntensity_List))],
        "Mean Intensity": meanIntensity_List,
        "Protein Concentration (ng/µl)": proteinConcentration_List,
        "Protein Concentration (nM)": proteinConcentration_nM_List
    })

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (s)'], df['Mean Intensity'], marker='o')
    plt.title('Mean Intensity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (s)'], df['Protein Concentration (ng/µl)'], marker='o')
    plt.title('Protein Concentration vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Protein Concentration (ng/µl)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (s)'], df['Protein Concentration (nM)'], marker='o')
    plt.title('Protein Concentration (nM) vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Protein Concentration (nM)')
    plt.grid(True)
    plt.show()

def getNumberOfProteinMolecules(dropletVolume, timeInterval, mw_kda, time_unit='seconds'):
    global numberOfProteinMolecules_List

    # Convert time interval to seconds if necessary
    if time_unit == 'minutes':
        timeInterval *= 60

    proteinMass_List = [i * dropletVolume for i in proteinConcentration_List]
    numberOfProteinMolecules_List = [(j * 6e14) / (mw_kda * 1e3) for j in proteinMass_List]

    df_num_proteins = pd.DataFrame({
        "Time (s)": [i * timeInterval for i in range(len(numberOfProteinMolecules_List))],
        "Number of Protein Molecules": numberOfProteinMolecules_List
    })

    plt.figure(figsize=(10, 6))
    plt.plot(df_num_proteins['Time (s)'], df_num_proteins['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(df_num_proteins['Time (s)'], df_num_proteins['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Protein Molecules')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

    global df
    df = df.merge(df_num_proteins, on="Time (s)")

def getRateOfChangeProteinMolecules(timeInterval, time_unit='seconds'):
    global timeValues_List, rateOfChangeProteinMolecules_List

    # Convert time interval to seconds if necessary
    if time_unit == 'minutes':
        timeInterval *= 60
    
    p_vals = np.array(numberOfProteinMolecules_List)
    length = len(numberOfProteinMolecules_List)
    maxTimeValue = (length - 1) * timeInterval 
    t_vals = np.linspace(0, maxTimeValue, length)
    timeValues_List = t_vals.tolist()

    dp_dt = np.gradient(p_vals, t_vals)
    rateOfChangeProteinMolecules_List = dp_dt.tolist()

    dp_dt = gaussian_filter1d(dp_dt, sigma=2)
    
    df_rate_change = pd.DataFrame({
        "Time (s)": timeValues_List,
        "Rate of Change of Number of PM": rateOfChangeProteinMolecules_List
    })

    plt.figure(figsize=(10, 6))
    plt.plot(df_rate_change['Time (s)'], df_rate_change['Rate of Change of Number of PM'], label='Numerical derivative', marker='o', color="green")
    plt.xlabel('Time (s)')
    plt.ylabel('Rate of change of the number of protein molecules')
    plt.title('Rate of change of the number of protein molecules with respect to time')
    plt.legend()
    plt.grid(True)
    plt.show()

    global df
    df = df.merge(df_rate_change, on="Time (s)")

def saveData(fileName):
    global df
    df.to_csv(fileName, index=False)

def calculate_RpD(R_p, D, K_TX):
    discriminant = (R_p + D + K_TX)**2 - 4 * R_p * D
    return 1e-6 if discriminant < 0 else 0.5 * (R_p + D + K_TX - np.sqrt(discriminant))

def dPdt(T, P, Q, S, tau_0, tau_f, k3, k11):
    return Q * (1 - np.exp(-(T - tau_0 - tau_f) / k3)) - (S * P) / (k11 + P) if T > tau_0 + tau_f else 0 

def solve_ODE(params, N_p, N_m, D):
    global subset_ProteinConcentration_nM_List
    k_TL, k_TX, R_p, tau_m, K_TL, R, k_deg, X_p, K_p, tau_0, tau_f = params
    RpD = calculate_RpD(R_p, D, k_TX)
    Q = (k_TL * k_TX * RpD * tau_m) / (N_p * (1 + K_TL / R) * N_m)  
    S = k_deg * X_p
    k3 = tau_m
    k11 = K_p

    T = np.linspace(0, 5000, len(subset_ProteinConcentration_nM_List))
    P_initial = 0

    p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method="LSODA")
    return p.y[0]
  
def objective_function(params, N_p, N_m, D):
    pModel = solve_ODE(params, N_p, N_m, D)
    return np.sum((subset_ProteinConcentration_nM_List - pModel) ** 2)

def optimize_parameters(initial_guesses, N_p, N_m, D):
    global optimizedParameters
    result = minimize(objective_function, initial_guesses, args=(N_p, N_m, D), method='TNC')
    optimizedParameters = result.x

def showModel(N_p, N_m, D):
    global optimizedParameters
    print("Optimized parameters:")
    params = ["k_TL", "k_TX", "R_p", "tau_m", "K_TL", "R", "k_deg", "X_p", "K_p", "tau_0", "tau_f"]
    for param, value in zip(params, optimizedParameters):
        print(f"{param}: {value}")

    optimizedModel = solve_ODE(optimizedParameters, N_p, N_m, D)
    T = np.linspace(0, 5000, len(subset_ProteinConcentration_nM_List))
    
    plt.figure(figsize=(10, 6))
    plt.plot(T, subset_ProteinConcentration_nM_List, label='Experimental Curve', linestyle='--', color='orange')
    plt.plot(T, optimizedModel, label='Theoretical Curve')
    plt.title('Protein Concentration vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Protein Concentration (nM)')
    plt.legend()
    plt.grid(True)
    plt.show()

def run_full_analysis(paths, calibration_curve_paths, time_interval, droplet_volume, mw_kda, N_p, N_m, D, initial_guesses, file_name, time_unit='seconds'):
    global subset_ProteinConcentration_nM_List, optimizedParameters
    calculateMeanIntensity(paths)
    getConcentration(calibration_curve_paths, mw_kda)
    constructDataFrames(time_interval, time_unit)
    getNumberOfProteinMolecules(droplet_volume, time_interval, mw_kda, time_unit)
    getRateOfChangeProteinMolecules(time_interval, time_unit)

    length = len(proteinConcentration_nM_List)
    proteinConcentration_nM_List_NP = np.array(proteinConcentration_nM_List)
    subset_indices = np.linspace(0, length - 1, length, dtype=int)
    subset_ProteinConcentration_nM_List = proteinConcentration_nM_List_NP[subset_indices]

    optimize_parameters(initial_guesses, N_p, N_m, D)
    showModel(N_p, N_m, D)
    
    # Add parameters to the DataFrame
    parameter_names = ["k_TL", "k_TX", "R_p", "tau_m", "K_TL", "R", "k_deg", "X_p", "K_p", "tau_0", "tau_f"]
    parameter_values = optimizedParameters
    parameters_df = pd.DataFrame([parameter_values], columns=parameter_names)
    
    for param, value in zip(["N_p", "N_m", "D"], [N_p, N_m, D]):
        parameters_df[param] = value

    global df
    df = pd.concat([df, parameters_df], axis=1)
    saveData(file_name)
