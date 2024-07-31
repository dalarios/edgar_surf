import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import csv
import glob
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import solve_ivp
from ipywidgets import interact, FloatSlider, Layout, interactive
from scipy.optimize import minimize
import random
import seaborn as sns

import multiprocessing as mp
import time
import os

# Part 1

#Initialize global variables
timeValues_List = list()
meanIntensity_List = list()
proteinConcentration_List = list()
proteinConcentration_nM_List = list()
numberOfProteinMolecules_List = list()
rateOfChangeProteinMolecules_List = list()
optimizedParameters = list()

# This function utilizes the images taken for an experiment of a kinesin motor protein
def calculateMeanIntensity(paths):
    for i in range(0, len(paths)): 
        
        image_path = paths[i] # Extract the path of each experimental image that was taken by the microscope
        # Load each image as a matrix
        image_matrix = io.imread(image_path)
        meanIntensity = image_matrix.mean() # Calculate the mean intensity value of the whole matrix 
        meanIntensity_List.append(meanIntensity) # Save this mean intensity value to a Python list

# This function utilizes 9 sample images to analyze the relationship between "Mean Intensity" and "Protein Concentration"
def getConcentration(calibrationCurvePaths, mw_kda): # This function takes a list of image paths and molecular weight in kDa as arguments
    
    meanIntensity_CalibrationCurve_List = list()
    for i in range(0, len(calibrationCurvePaths)):
        
        image_path = calibrationCurvePaths[i] # Extract the path of each of the 9 sample images
        # Load the image as a matrix
        image_matrix = io.imread(image_path)
        meanIntensity = image_matrix.mean() # Calculate the mean intensity value of the whole matrix 
        meanIntensity_CalibrationCurve_List.append(meanIntensity) # Save this mean intensity value to a Python list

    df = pd.DataFrame(meanIntensity_CalibrationCurve_List).reset_index() # Create a data frame by adding the mean intensity values to 1 column
    df = df.rename(columns={"index":"Protein Concentration (microgram / milliliter)", 0:"Mean Intensity"}) # Rename each of the 2 columns
    sampleConcentration_Values = [0, 2, 5, 10, 20, 40, 80, 160, 320]
    df["Protein Concentration (microgram / milliliter)"] = sampleConcentration_Values # Fill out the other column with the sample concentration values

    # Get the equation (linear) of best fit for the Protein Concentration (nanograms/microliter)
    x = df["Protein Concentration (microgram / milliliter)"]
    y = df["Mean Intensity"]

    slope, intercept = np.polyfit(x, y, 1) # The degree of the polynmial that will fit the data is 1. Multiple return values is allowed in Python
    

    line_of_best_fit = slope * x + intercept # Create the line of best fit using the found slope and y-intercept.

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df["Protein Concentration (microgram / milliliter)"], df["Mean Intensity"], marker='o', linestyle='none', label='Data points')
    plt.plot(x, line_of_best_fit, label=f'Line of Best Fit: y = {slope:.2f}x + {intercept:.2f}', color='red')
    plt.title('Mean Intensity vs Protein Concentration')
    plt.xlabel('Protein Concentration (nanogram / microliter)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.legend()
    plt.show()

    """Transform the dependent variables. 
    The linear model found earlier not only applies to the 9 sample images.
    This linear model also applies to ANY experimental image taken by the microscope. """
    for i in range(0, len(meanIntensity_List)):
        proteinConcentration = (meanIntensity_List[i] - intercept) / slope # Calculate each "protein concentration" value
        proteinConcentration_List.append(proteinConcentration) # Save all "protein concentration" values to a Python List
        proteinConcentration_nM = ((proteinConcentration * 1e-3) / (mw_kda * 1e3)) * 1e9 # Convert each "protein concentration" value to the units of nM
        proteinConcentration_nM_List.append(proteinConcentration_nM) # Save all "protein concentration [nM]" values to another Python List

def constructDataFrames(timeInterval):
    global meanIntensity_List
    global proteinConcentration_List
    global proteinConcentration_nM_List

    minimumIntensityValue = min(meanIntensity_List)
    adjustedMeanIntensity_List = [x - minimumIntensityValue for x in meanIntensity_List] # Subtract the minimum mean intensity value from ALL values
    meanIntensity_List = adjustedMeanIntensity_List # Adjust the entire "Mean Intensity" experimental data. The purpose is that the experimental data starts from (0,0).

    # ------ Do the same procedure as described before, to adjuste the "protein concentration" and "protein concentration [nM]" data ------
    minimumProteinConcentration = min(proteinConcentration_List)
    adjustedProteinConcentration_List = [x - minimumProteinConcentration for x in proteinConcentration_List]
    proteinConcentration_List = adjustedProteinConcentration_List

    minimumProteinConcentration_nM = min(proteinConcentration_nM_List)
    adjustedProteinConcentration_List_nM = [x - minimumProteinConcentration_nM for x in proteinConcentration_nM_List]
    proteinConcentration_nM_List = adjustedProteinConcentration_List_nM    

    # Fill out a 2-column DataFrame
    df = pd.DataFrame(meanIntensity_List).reset_index() # Create a data frame with 2 columns. One column contains all the "Mean Intensity" experimental data.
    df = df.rename(columns={"index":"Time (min)", 0:"Mean Intensity"}) # Rename each of the 2 columns, ensuring proper units.
    # Currently, the df["Time (min)"] column looks like this: 0, 1, 2, 3, ...   Therefore, we must manipulate this column.
    df["Time (min)"] = df["Time (min)"] * timeInterval # Manipulate the "time" values, according to the time interval that the user decided.

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Mean Intensity'], marker='o')
    plt.title('Mean Intensity vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Mean Intensity')
    plt.grid(True)
    plt.show()

    # Fill out a 2-column DataFrame
    df2 = pd.DataFrame(proteinConcentration_List).reset_index() # Create a data frame with 2 columns. One column contains all the "Protein Concentration" experimental data.
    df2 = df2.rename(columns={"index":"Time (min)", 0:"Protein Concentration (nanogram / microliter)"}) # Rename each of the 2 columns, ensuring proper units.
    df2["Time (min)"] = df2["Time (min)"] * timeInterval # Manipulate the "time" values, according to the time interval that the user decided.
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df2['Time (min)'], df2['Protein Concentration (nanogram / microliter)'], marker='o')
    plt.title('Protein Concentration vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (microgram / milliliter)')
    plt.grid(True)
    plt.show()

    # Fill out a 2-column DataFrame
    df3 = pd.DataFrame(proteinConcentration_nM_List).reset_index() # Create a data frame with 2 columns. One column contains all the "Protein Concentration [nM]" experimental data.
    df3 = df3.rename(columns={"index":"Time (min)", 0:"Protein Concentration (nM)"}) # Rename each of the 2 columns, ensuring proper units.
    df3["Time (min)"] = df3["Time (min)"] * timeInterval # Manipulate the "time" values, according to the time interval that the user decided.

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df3['Time (min)'], df3['Protein Concentration (nM)'], marker='o')
    plt.title('Protein Concentration (nM) vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (nM)')
    plt.grid(True)
    plt.show()

def getNumberOfProteinMolecules(dropletVolume, timeInterval, mw_kda):
    global numberOfProteinMolecules_List
    proteinMass_List = [i * dropletVolume for i in proteinConcentration_List] # The list comprehension technique was used to create a list with all the "Protein Mass" values.
    numberOfProteinMolecules_List = [(j * 6e14) / (mw_kda * 1e3) for j in proteinMass_List] # This expression was derived from several mathematical steps. A new list is being created that has all the "Number of Protein Molecules" values.

    df = pd.DataFrame(numberOfProteinMolecules_List).reset_index() # Create a data frame with 2 columns. One column contains all the "Number of Protein Molecules" values.
    df = df.rename(columns={"index":"Time (min)", 0:"Number of Protein Molecules"}) # Rename each of the 2 columns, ensuring proper units.
    df["Time (min)"] = df["Time (min)"] * timeInterval # Manipulate the "time" values, according to the time interval that the user decided.

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)
    plt.show()

    # Plot the same data again, but using a logarithmic scale to represent the "Number of Protein Molecules" data
    plt.figure(figsize=(10, 6))
    plt.plot(df['Time (min)'], df['Number of Protein Molecules'], marker='o')
    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Number of Protein Molecules')
    # y axis log scale
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def getRateOfChangeProteinMolecules(timeInterval):
    global timeValues_List
    global rateOfChangeProteinMolecules_List
    
    p_vals = np.array(numberOfProteinMolecules_List) # Save all the "Number of Protein Molecules" values to a numpy array.
    length = len(numberOfProteinMolecules_List)
    maxTimeValue = (length - 1) * timeInterval 
    t_vals = np.linspace(0, maxTimeValue, length) # Create a numpy array that has all the "time" values, in which the numerical derivative will be estimated. 
    timeValues_List = t_vals.tolist() # Save all the "time" values in a list. This list will be used later on.

    # Estimate the numerical derivative of the number of protein molecules with respect to time
    dp_dt = np.gradient(p_vals, t_vals) # Use the 2 numpy arrays previously created
    rateOfChangeProteinMolecules_List = dp_dt.tolist() # Save all the estimated derivative values into a Python list.

    # apply gaussian filter with sigma 2
    dp_dt = gaussian_filter1d(dp_dt, sigma=2)
    
    # Plot the estimated derivative
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, dp_dt, label='Numerical derivative', marker='o', color="green")
    plt.xlabel('Time (min)')
    plt.ylabel('Rate of change of the number of protein molecules')
    plt.title('Rate of change of the number of protein molecules with respect to time')
    plt.legend()
    plt.grid(True)
    plt.show()

def saveExperimentalData(experiment_fileName): # This function saves all the experimental data to a CSV file

    dataFile = open(experiment_fileName, 'w', newline="")
    writerCSV = csv.writer(dataFile)
    headerRow = list()
    headerRow.append("Time (min)")
    headerRow.append("Mean Intensity (A.U.)")
    headerRow.append("Protein Concentration (ng per µL)")
    headerRow.append("Protein Concentration (nM)")
    headerRow.append("Number of Protein Molecules")
    headerRow.append("Rate of Change of Number of Protein Molecules (PM per min)")
    writerCSV.writerow(headerRow) # Add the header row to the CSV file.

    for i in range(0, len(meanIntensity_List)): # Fill all of the columns with the experimental data.
        dataRow = list()
        dataRow.append(timeValues_List[i])
        dataRow.append(meanIntensity_List[i])
        dataRow.append(proteinConcentration_List[i])
        dataRow.append(proteinConcentration_nM_List[i])
        dataRow.append(numberOfProteinMolecules_List[i])
        dataRow.append(rateOfChangeProteinMolecules_List[i])
        writerCSV.writerow(dataRow) # Below the header row, add all the experimental data to the CSV file.
    dataFile.close()


# Part 2

""" This function should only be used in the 2nd wrapper function called "runTheoreticalAnalysis()".
    By using this function, you should already have all the experimental data saved in .csv files.
    This means that this function will only store the experimental data that will later on be compared to theoretical data. """

def loadExperimentalData(experiment_file_name): 
    global timeValues_List, proteinConcentration_nM_List 
    
    experimentalData_df = pd.read_csv(experiment_file_name) # Load all the experimental data of the .csv file, into a Pandas DataFrame.
    column_TimeValues = experimentalData_df["Time (min)"] # Only extract the "time" values from the DataFrame.
    timeValues_List = column_TimeValues.tolist() # Store all the "time" values in a list. To do so, convert from a numpy array to a list
    column_ProteinConcentration_nM = experimentalData_df["Protein Concentration (nM)"] # Only extract the "Protein Concentration (nM)" values from the DataFrame.
    proteinConcentration_nM_List = column_ProteinConcentration_nM.tolist() # Store all the "Protein Concentration (nM)" values in a list. To do so, convert from a numpy array to a list
   

# This function calculate the [R_p D] complex 
def calculate_RpD2(R_p, D, k_TX): # 3 parameters are needed to calculate the [R_p D] complex: protein production rate, DNA concentration, and transcription rate
    discriminant = (R_p + D + k_TX)**2 - 4 * R_p * D
    if discriminant < 0:
        return 1e-6 # Return a small positive value if the discriminant is negative
    else:
        return 0.5 * (R_p + D + k_TX - np.sqrt(discriminant)) # Else return the actual [R_p D] complex equation

# Define the ordinary differential equation (ODE) that describes protein concentration dynamics
def dPdt(T, P, Q, S, tau_0, tau_f, k3, k11): # Not only accept the variables T and P. Also, accept parameters that will be treated as constants in the ODE.
    if T > tau_0 + tau_f:
        return Q * (1 - np.exp(-(T - tau_0 - tau_f) / k3)) - (S * P) / (k11 + P) # Return the ODE in its most simplied form, ready to be solved numerically.
    else:
         """If the current "time" value is not greater than the sum of both of the time delay values, 
         this means that are no proteins being expressed. Hence, if this is the case, the rate 
         of change of the protein concentration with respect to time is 0.  """
        
         return 0 
        

def solve_ODE(params, N_p, N_m, D):

    k_TL, k_TX, R_p, tau_m, K_TL, R, k_deg, X_p, K_p, tau_0, tau_f = params 

    RpD = calculate_RpD2(R_p, D, k_TX) # For simplicity purposes, the [R_p D] complex was previously calculated using a function
    
    # Condense many constant terms into single variables. This is done for simplicity purposes.
    Q = (k_TL * k_TX * RpD * tau_m) / (N_p * (1 + K_TL / R) * N_m)  
    S = k_deg * X_p
    k3 = tau_m
    k11 = K_p

    # Time ranges from T = 0 to T = 5000 seconds
    T = np.linspace(0, 5000, len(proteinConcentration_nM_List)) # This has the same size as the experimental data of "Protein Concentration [nM]""

    P_initial = 0  # At t = 0, the protein concentration P(0) = 0

    """All of the constants such as Q, S, k3... need to be passed as arguments into the solve_ivp() function
    "LSODA" will be the first method to be used to numerically solve the ODE"""
    p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method ="LSODA", rtol=1e-6, atol=1e-8)
    
    # If LSODA fails, the BDF method will be used
    if p.status != 0:
        p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method='BDF', rtol=1e-6, atol=1e-8)

    # If BDF fails, the RK45 method will be used
    if p.status != 0:
        p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method='RK45', rtol=1e-6, atol=1e-8)

    # If BDF fails, then the "Radau" method will be used
    if p.status != 0:
         p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method='Radau', rtol=1e-6, atol=1e-8)

    # Handle the case if the "Radau" method also fails
    if p.status != 0:
        raise RuntimeError("The ODE could not be solved with none of the attempted methods: LSODA, BDF, RK45, or Radau")
    
    return p.y[0] # Return all the theoretical "Protein Concentration [nM]" values, by using this specific syntax.

# This objective function uses the method of "Sum of Squared Errors (SSE)"
def objective_function(params, N_p, N_m, D):
    pModel = solve_ODE(params, N_p, N_m, D)  # Extract all the theoretical "Protein Concentration [nM]" values
    """Apply the definition of the SSE method. 
    To do so, use both the experimental and theoretical values of "Protein Concentration [nM]" """
    return np.sum((proteinConcentration_nM_List-pModel)**2) 

# This function uses the PROVIDED inital guesses to find the optimized parameters only 1 time.
def optimize_parameters(initial_guesses, N_p, N_m, D):
    global optimizedParameters
    # The lower bounds of tau_m, R, and K_p are a very small positive number (not 0), to avoid having issues of dividing by 0
    bounds = [(0, 100), (0, 100), (0, 500), (1e-6, 5000), (0, 100), (1e-6, 1e3), (0, 100), (0, 500), (1e-6, 100), (0, 10), (0, 2000)]
    """  'TNC' is a popular method to minimize an objective function. It will be used for this scenario.
    'L-BFGS-B' is another method to minimize an objective function and could be used if the "TNC" method fails.
    Initial guesses are required to use any of the 2 methods. Bounds are optional, but for this scenario they are required. """
    result = minimize(objective_function, initial_guesses, args=(N_p, N_m, D), method='TNC', bounds=bounds)  
    optimizedParameters = result.x  # Since "result" is an object, we need to access a certain attribute of "result" to extract the optimized parameters

# This function uses RANDOM initial guesses to find optimized parameters many times.
def optimize_parameters_many_times(initial_guesses, N_p, N_m, D):
    bounds = [(0, 100), (0, 100), (0, 500), (1e-6, 5000), (0, 100), (1e-6, 1e3), (0, 100), (0, 500), (1e-6, 100), (0, 10), (0, 2000)] # Use the same bounds defined before
    print("El programa PASO AQUI!")
    result = minimize(objective_function, initial_guesses, args=(N_p, N_m, D), method='TNC', bounds=bounds)  
    currentOptimizedParameters = result.x 
    SSE = result.fun # Extract the Sum of Squared Errors (SSE) value, by using this specific syntax.
    return currentOptimizedParameters, SSE # Return the optimized parameters and the SSE value

"""This function shows the optimized parameters that were calculated using the PROVIDED initial guesses.
    Also, this function uses the optimized parameters to generate the theoretical curve.
    Finally, both the theoretical curve and the experimental curve are shown in the same graph. """
def showOptimizedParameters(N_p, N_m, D): 

    global optimizedParameters
    # Print the optimized parameters
    print("These are the optimized parameters that were calculated using the provided initial guesses:")
    print("k_TL:", optimizedParameters[0])
    print("k_TX:", optimizedParameters[1])
    print("R_p:", optimizedParameters[2])
    print("tau_m:", optimizedParameters[3])
    print("K_TL:", optimizedParameters[4])
    print("R:", optimizedParameters[5])
    print("k_deg:", optimizedParameters[6])
    print("X_p:", optimizedParameters[7])
    print("K_p:", optimizedParameters[8])
    print("tau_0:", optimizedParameters[9])
    print("tau_f:", optimizedParameters[10])

    optimizedModel = solve_ODE(optimizedParameters, N_p, N_m, D) # Generate the theoretical curve, by using the optimized parameters to solve the ODE. 
    T = np.linspace(0, 5000, len(proteinConcentration_nM_List)) # Same size as the experimental data of the protein concentration
    plt.figure(figsize=(10, 6))  
    plt.plot(T, proteinConcentration_nM_List, label='Experimental Curve', linestyle='--', color='orange')
    plt.plot(T, optimizedModel, label='Theoretical Curve') 
    plt.title('Protein Concentration vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Protein Concentration (nM)')
    plt.legend()
    plt.grid(True)
    plt.show()

#
def runParameterOptimization(initial_guesses, N_p, N_m, D, theory_file_name):

    global optimizedParameters

    parameter_names = ["SSE Value", "k_TL", "k_TX", "R_p", "tau_m", "K_TL", "R", "k_deg", "X_p", "K_p", "tau_0", "tau_f", "N_p", "N_m", "D"]
    parameters_df = pd.DataFrame(columns=parameter_names) # Initialize a DataFrame, by providing the names of all the columns

    # It is important that the bounds to generate the RANDOM initial guesses are the same bounds used for finding the optimal parameters
    bounds = [(0, 100), (0, 100), (0, 500), (1e-6, 5000), (0, 100), (1e-6, 1e3), (0, 100), (0, 500), (1e-6, 100), (0, 10), (0, 2000)] 
    knownParameters = [N_p, N_m, D]
    #parametersRangeMatrix = [] # This will become a matrix
    for i in range(100):

        try:
            print(i)
            """ The following line generates 11 random float values. Each of the PROVIDED initial guesses is used as the
            mean value of a normal distribution. The standard deviation (SD) of this normal distribution is proportional
            to each of the PROVIDED initial guesses (in this case, the SD is always 10% the value of each PROVIDED initial guess.)
            By following the process described before, RANDOM initial guesses are being created on every iteration."""
            random_initial_guesses = [np.clip(np.random.normal(loc=guess, scale=guess*0.1), low, high) for guess, (low, high) in zip(initial_guesses, bounds)]
            currentOptimizedParameters, SSE= optimize_parameters_many_times(random_initial_guesses, N_p, N_m, D) # Calculate new values for the optimized parameters, and extract the SSE value of this new model.
            print("Current Optimized Parameters:", currentOptimizedParameters)
            print("\n")
            print("SSE:", SSE)
            print("\n")
            currentOptimizedParameters_List = currentOptimizedParameters.tolist() # Convert a "np.ndarray" object to a Python list
            SSE_value = [SSE] # This is a list containing 1 element
            newRow = SSE_value + currentOptimizedParameters_List + knownParameters # Combine 3 Python lists into a single list. This exact order is very important.
            parameters_df.loc[len(parameters_df)] = newRow # Add a new row of optimized parameters to the DataFrame. Also add the SSE value.
            """
            if (i % 33 == 0):
                optimizedParameters = np.array([]) # Clear the contents of the global variable
                optimizedParameters = currentOptimizedParameters
                visualizeModel(optimizedParameters, N_p, N_m, D)
            """
        except IndexError:
            print("The optimized parameters could not be found using the following RANDOM initial guesses:")
            print("\n")
            print(random_initial_guesses)
            continue # Skip a bad set of optimized parameters that were calculated based on the random intial guesses

    """
    for i in parameter_names:
        minValue = parameters_df[i].min() # Extracts the minimum value of each ENTIRE column
        maxValue = parameters_df[i].max() # Extracts the maximum value of each ENTIRE column
        parameterRange = [minValue, maxValue]
        parametersRangeMatrix.append(parameterRange)
    print("Range of SSE values and parameters:")
    print()
    print(parametersRangeMatrix)
    """

    saveTheoreticalData(parameters_df, theory_file_name) # Save the DataFrame to a .csv file
    
def saveTheoreticalData(theory_df, theory_file_name):
    theory_df.to_csv(theory_file_name, index=False)


def showBestAndWorstModel(theory_file_name, N_p, N_m, D):
    theory_df = pd.read_csv(theory_file_name)
    minSSE_Value = theory_df["SSE Value"].min()
    maxSSE_Value = theory_df["SSE Value"].max()

    index_minSSE = theory_df[theory_df["SSE Value"] == minSSE_Value].index[0]
    index_maxSSE = theory_df[theory_df["SSE Value"] == maxSSE_Value].index[0]

    minSSE_row = theory_df.loc[index_minSSE] # Complete row (this is still not useful)
    minSSE_row_without_SSE = minSSE_row[minSSE_row != minSSE_Value] # Extract the other values found in the same row (NOT including the SSE value)
    minSSE_row_with_optimized_parameters = minSSE_row_without_SSE[:-3]
    print()
    print("This model was calculated using RANDOM initial guesses and has the least value of SSE (this is the best found model):")
    visualizeModel(minSSE_row_with_optimized_parameters, N_p, N_m, D)

    maxSSE_row = theory_df.loc[index_maxSSE] # Complete row (this is still not useful)
    maxSSE_row_without_SSE = maxSSE_row[maxSSE_row != maxSSE_Value] # Extract the other values found in the same row (NOT including the SSE value)
    maxSSE_row_with_optimized_parameters = maxSSE_row_without_SSE[:-3]
    print()
    print("This model was calculated using RANDOM initial guesses and has the greatest value of SSE (this is the worst found model):")
    visualizeModel(maxSSE_row_with_optimized_parameters, N_p, N_m, D)


# Part 3

# Calculate the [R_p D] complex using the equation in the paper's supplementary information. 
def calculate_RpD(R_p, D, K_TX): # Accept parameters to calculate the [R_p D] complex
    return 0.5 * (R_p + D + K_TX - np.sqrt((R_p + D + K_TX)**2 - 4 * R_p * D))

# Define the differential equation for protein concentration
def dPdt(T, P, Q, S, tau_0, tau_f, k3, k11): # Not only accept the variables T and P. Also, accept parameters that will be treated as constants in the ODE.
    if T > tau_0 + tau_f:
        return Q * (1 - np.exp(-(T - tau_0 - tau_f) / k3)) - (S * P) / (k11 + P)
    else:
        return 0 

# Create a function to plot the oscillators with given parameters
def plot_proteinConcentration(k_TL, k_TX, R_p, D, tau_m, N_p, K_TL, R, N_m, k_deg, X_p, K_p, tau_0, tau_f): #Parameters that will be able to be modified by the sliders
    
    RpD = calculate_RpD(R_p, D, k_TX) # For simplicity purposes, calculate [R_p D] complex using a function
    Q = (k_TL * k_TX * RpD * tau_m) / (N_p * (1 + K_TL / R) * N_m)  
    S = k_deg * X_p
    k3 = tau_m
    k11 = K_p

    # Time ranges from T = 0 to T = 5000 seconds
    T = np.linspace(0, 5000, len(proteinConcentration_nM_List)) # Same size as the experimental data of the protein concentration

    P_initial = 0  # At t = 0, the protein concentration P(0) = 0

    # All of the constants such as Q, S, tau_0... need to be passed as arguments into the solve_ivp()function
    p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method ="LSODA", rtol=1e-6, atol=1e-8) # The "LSODA" method will be used to numerically solve the ODE
    
    # If LSODA fails, the BDF method will be used
    if p.status != 0:
        p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method='BDF', rtol=1e-6, atol=1e-8)

    # If BDF fails, the RK45 method will be used
    if p.status != 0:
        p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method='RK45', rtol=1e-6, atol=1e-8)

    # If BDF fails, then the "Radau" method will be used
    if p.status != 0:
         p = solve_ivp(dPdt, [T[0], T[-1]], [P_initial], t_eval=T, args=(Q, S, tau_0, tau_f, k3, k11), method='Radau', rtol=1e-6, atol=1e-8)

    # Handle the case if the "Radau" method also fails
    if p.status != 0:
        raise RuntimeError("ODE solver failed for all attempted methods (LSODA, BDF, RK45, Radau).")
    
    plt.figure(figsize=(10, 6))  # Clear the figure before plotting
    plt.plot(T, proteinConcentration_nM_List, label='Experimental Curve', linestyle='--', color='orange')
    plt.plot(T, p.y[0], label='Theoretical Curve') # We need to access the "y" values from the object 'p' we created before
    plt.title('Protein Concentration vs. Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Protein Concentration (nM)')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualizeModel(optimizedParameters, N_p, N_m, D):

    k_TL, k_TX, R_p, tau_m, K_TL, R , k_deg, X_p, K_p, tau_0, tau_f = optimizedParameters

    # Create interactive sliders for the parameters used to calculate the protein concentration
    style = {'description_width': '300px'}  # Adjust the width as needed
   
    interact(plot_proteinConcentration, 
            k_TL=FloatSlider(value=k_TL , min=0.0, max=100, step=0.1, description='k_TL (amino acids/s)', layout=Layout(width='900px'), style=style, readout_format='.6f'),
            k_TX=FloatSlider(value=k_TX , min=0.0, max=100, step=0.1, description='k_TX (rNTP/s)', layout=Layout(width='900px'), style=style, readout_format='.6f'),
            R_p=FloatSlider(value=R_p, min=0.0, max=500, step=0.1, description='RNA polymerase concentration (nM)', layout=Layout(width='900px'), style=style, readout_format='.6f'), 
            D=FloatSlider(value=D, min=0.0, max=1000, step=1, description='DNA concentration (nM)', layout=Layout(width='900px'), style=style), ## We know for sure the value of DNA concentration
            tau_m=FloatSlider(value=tau_m , min=1e-6, max=5000, step=0.1, description='mRNA lifetime (seconds)', layout=Layout(width='900px'), style=style, readout_format='.6f'),
            N_p=FloatSlider(value=N_p, min=0.0, max=10000, step=1, description='protein length (amino acids)', layout=Layout(width='900px'), style=style, readout_format='.6f'), ## We know for sure the number of aminoacids
            K_TL = FloatSlider(value=K_TL, min=0.0, max=100, step=0.1, description='Michaelis-Menten constant for translation (nM)', layout=Layout(width='900px'), style=style, readout_format='.6f'),
            R=FloatSlider(value=R, min=1e-6, max=1e3, step=0.1, description='ribosome concentration (nM)', layout=Layout(width='900px'), style=style, readout_format='.6f'), 
            N_m=FloatSlider(value=N_m, min=0.0, max=10000, step=1, description='mRNA Length (Nucleotides)', layout=Layout(width='900px'), style=style, readout_format='.6f'), ## We know for sure the number of nucleotides (this is based on the DNA design)
            k_deg=FloatSlider(value=k_deg, min=0.0, max=100, step=0.1, description='protein degradation rate constant (1/s)', layout=Layout(width='900px'), style=style, readout_format='.6f'), 
            X_p=FloatSlider(value=X_p, min=0.0, max=500, step=0.1, description='protease concentration (nM)', layout=Layout(width='900px'), style=style, readout_format='.6f'), 
            K_p=FloatSlider(value=K_p, min=1e-6, max=100, step=0.01, description='Michaelis-Menten constant for degradation (nM)', layout=Layout(width='900px'), style=style, readout_format='.6f'),
            tau_0=FloatSlider(value=tau_0, min=0.0, max=10, step=0.01, description='transcription delay (seconds)', layout=Layout(width='900px'), style=style, readout_format='.6f'), 
            tau_f=FloatSlider(value=tau_f, min=0.0, max=2000, step=0.1, description='protein folding delay (seconds)', layout=Layout(width='900px'), style=style, readout_format='.6f'))


# *** Wrapper function 
def runFullAnalysis(paths, calibration_curve_paths, time_interval, droplet_volume, mw_kda, N_p, N_m, D, initial_guesses, experiment_file_name, theory_file_name):
    global optimizedParameters, timeValues_List, meanIntensity_List, proteinConcentration_List, proteinConcentration_nM_List, numberOfProteinMolecules_List, rateOfChangeProteinMolecules_List

    # Part 1
    calculateMeanIntensity(paths)
    getConcentration(calibration_curve_paths, mw_kda)
    constructDataFrames(time_interval)
    getNumberOfProteinMolecules(droplet_volume, time_interval, mw_kda)
    getRateOfChangeProteinMolecules(time_interval)
    saveExperimentalData(experiment_file_name)

    # Part 2

    # This is to generate and show a "demo" optimized model. It directly uses the initial guesses provided (without generating random initial guesses!!)
    optimize_parameters(initial_guesses, N_p, N_m, D)
    showOptimizedParameters(N_p, N_m, D)
    visualizeModel(optimizedParameters, N_p, N_m, D)

    # Part 3

    """ Multiprocessing will be used to carry out the optimization of the parameters 100 times.
        This will signifcantly reduce the execution time of the program."""
    # ----------------------------------------------------------------
    num_processes = os.cpu_count()  # Get the number of available CPU cores

    # Timing multiprocessing
    start_time = time.time()
    jobs = []
    for _ in range(num_processes):
        # The runParameterOptimization(initial_guesses, N_p, N_m, D, theory_file_name) gets called here
        process = mp.Process(target=runParameterOptimization, args=(initial_guesses, N_p, N_m, D, theory_file_name))
        jobs.append(process)
        process.start()
    for job in jobs:
        job.join()
    multiprocessing_time = time.time() - start_time
    print(f'Multiprocessing time: {multiprocessing_time:.4f} seconds')
    # ----------------------------------------------------------------
    showBestAndWorstModel(theory_file_name, N_p, N_m, D)
    
    
    #Clear the contents of all the lists used. All of these lists are global variables and need to be cleared for the next protein analysis
    optimizedParameters = np.array([]) # Empty the numpy array to prepare the analysis of the next protein experiment
    timeValues_List.clear()
    meanIntensity_List.clear()
    proteinConcentration_List.clear()
    proteinConcentration_nM_List.clear()
    numberOfProteinMolecules_List.clear()
    rateOfChangeProteinMolecules_List.clear()

    # *** 2nd Wrapper function 
def runTheoreticalAnalysis(experiment_file_name, N_p, N_m, D, initial_guesses, theory_file_name):
    global optimizedParameters, timeValues_List, proteinConcentration_nM_List

    # Part 2
    loadExperimentalData(experiment_file_name)

    # Generates and shows a "demo" optimized model. This directly uses the initial guesses provided (without generating random initial guesses!!)
    optimize_parameters(initial_guesses, N_p, N_m, D)
    showOptimizedParameters(N_p, N_m, D)
    visualizeModel(optimizedParameters, N_p, N_m, D)

    # Part 3

    """ Multiprocessing will be used to carry out the optimization of the parameters 100 times.
        This will signifcantly reduce the execution time of the program."""
    # ----------------------------------------------------------------
    num_processes = os.cpu_count()  # Get the number of available CPU cores

    # Timing multiprocessing
    start_time = time.time()
    jobs = []
    for _ in range(num_processes):
        # The runParameterOptimization(initial_guesses, N_p, N_m, D, theory_file_name) gets called here
        process = mp.Process(target=runParameterOptimization, args=(initial_guesses, N_p, N_m, D, theory_file_name))
        jobs.append(process)
        process.start()
    for job in jobs:
        job.join()
    multiprocessing_time = time.time() - start_time
    print(f'Multiprocessing time: {multiprocessing_time:.4f} seconds')
    # ----------------------------------------------------------------
    showBestAndWorstModel(theory_file_name, N_p, N_m, D)
    

    #Clear the contents of all the lists used. All of these lists are global variables and need to be cleared for the next protein analysis
    optimizedParameters = np.array([]) # Empty the numpy array to prepare the analysis of the next protein experiment
    timeValues_List.clear()
    proteinConcentration_nM_List.clear()

 
# Part 4

def showExperimentalDataTogether():

    experimentalFiles = sorted(glob.glob("experimentalData_*"))
    motorProteins_Names = [file.replace("experimentalData_", "").replace(".csv", "") for file in experimentalFiles]

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Mean Intensity (A.U.)'], marker='o') # Plot the "Mean Intensity" data only
    
    plt.title('Mean Intensity vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Mean Intensity (A.U.)')
    plt.grid(True)
    plt.legend(motorProteins_Names)
    plt.savefig("Mean_Intensity.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    """
    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Protein Concentration (ng per µL)'], marker='o') # Plot the "Protein Concentration" data only
    
    plt.title('Protein Concentration vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (ng per µL)')
    plt.grid(True)
    plt.legend(motorProteins_Names)
    plt.savefig("Protein_Concentration.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop
    """
    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Protein Concentration (nM)'], marker='o') # Plot the "Number of Protein Molecules" data only

    plt.title('Protein Concentration (nM) vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Protein Concentration (nM)')
    plt.grid(True)
    plt.legend(motorProteins_Names)  
    plt.savefig("Protein_Concentration_(nM).png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Number of Protein Molecules'], marker='o') # Plot the "Number of Protein Molecules" data only

    plt.title('Number of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Number of Protein Molecules')
    plt.grid(True)
    plt.legend(motorProteins_Names)  
    plt.savefig("Number_of_Protein_Molecules.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop

    plt.figure(figsize=(10, 6)) # Preparation for plotting
    for i in experimentalFiles:
        dataFrame_Protein = pd.read_csv(i) # Save into a Pandas dataframe ALL the data for the current protein
        plt.plot(dataFrame_Protein['Time (min)'], dataFrame_Protein['Rate of Change of Number of Protein Molecules (PM per min)'], marker='o') # Plot the "Rate of Change of Protein Molecules" data only

    plt.title('Rate of Change of Protein Molecules vs Time')
    plt.xlabel('Time (min)')
    plt.ylabel('Rate of Change of Protein Molecules (PM per min)')
    plt.grid(True)
    plt.legend(motorProteins_Names)
    plt.savefig("Rate_of_Change_of_Protein_Molecules.png")
    plt.show() # At the end of the loop, call the .show() function to combine all the plots created inside the for loop


# Part 5

def showTheoreticalDataTogether():

    theoreticalFiles = glob.glob("optimizedParameters_*")
    motorProteins_Names = [file.replace("optimizedParameters_", "").replace(".csv", "") for file in theoreticalFiles]

    completeDataFrame = pd.DataFrame() # Initialize an empty DataFrame that will be filled in a loop
    for i in range(len(theoreticalFiles)):
        dataFrame_Protein = pd.read_csv(theoreticalFiles[i]) # Save into a Pandas dataframe ALL the data for the current protein
        dataFrame_Protein = dataFrame_Protein.iloc[:, :-3] # Remove the last 3 columns of the DataFrame, which corresponds to the columns of the KNOWN PARAMETERS
        dataFrame_Protein["Kinesin Motor Protein"] = motorProteins_Names[i]
        completeDataFrame = pd.concat([completeDataFrame, dataFrame_Protein], ignore_index=True)

    # "Melt" the data. Convert the DataFrame from wide format to long format.
    """With the "long" format, the DataFrame has 3 columns. One for the kinesin motor proteins, 
    another one for the names of the parameters, and the third column is for the values of the parameters.
    """
    melted_data = completeDataFrame.melt(id_vars='Kinesin Motor Protein', var_name='Parameter Name', value_name='Value of Parameter')

    # Create a categorical plot to show the SSE values of the optimized models
    plt.figure(figsize=(15, 10))

    SSE_Parameter = ["SSE Value"]
    dataFrame_SSE_Values = melted_data[melted_data["Parameter Name"].isin(SSE_Parameter)]
    sns.stripplot(x='Parameter Name', y='Value of Parameter', hue='Kinesin Motor Protein', data=dataFrame_SSE_Values, dodge=True, jitter=True, alpha=0.7)
    plt.title('SSE Values of the Different Optimized Models That Were Generated')
    plt.xticks(rotation=90)
    plt.show()

    # Create a categorical plot to show the parameters with the smaller values
    plt.figure(figsize=(15, 10))

    small_parameters = ["k_TL", "k_TX", "R_p", "K_TL", "k_deg", "X_p", "K_p", "tau_0"]
    dataFrame_small = melted_data[melted_data["Parameter Name"].isin(small_parameters)]
    sns.stripplot(x='Parameter Name', y='Value of Parameter', hue='Kinesin Motor Protein', data=dataFrame_small, dodge=True, jitter=True, alpha=0.7)
    plt.title('Parameter Values for the Kinesin Motor Proteins')
    plt.xticks(rotation=90)
    plt.show()
    
    # Create a categorical plot to show the parameters with the larger values
    plt.figure(figsize=(15, 10))

    large_parameters = ["R", "tau_m", "tau_f"]
    dataFrame_large = melted_data[melted_data["Parameter Name"].isin(large_parameters)]
    sns.stripplot(x='Parameter Name', y='Value of Parameter', hue='Kinesin Motor Protein', data=dataFrame_large, dodge=True, jitter=True, alpha=0.7)
    plt.xticks(rotation=90)
    plt.title('Parameter Values for the Kinesin Motor Proteins')
    plt.show()
    
    