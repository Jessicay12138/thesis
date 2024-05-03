
"""
1/25/2024, Optimal Weight Allocation Using Simulated Annealing
Attempts to find the optimal weight allocation between equity, bond and cash over 60-year age
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import random
import time
import bisect
from joblib import Parallel, delayed
import csv
# import multiprocessing
np.random.seed(37319) 
death_age=80
#########################
###I: HELPER FUNCTIONS###
#########################
"""Helper Function to Find the Closest Point of Given Value on a Grid"""
def point_on_grid(grid, value):
    id = (np.abs(grid - value)).argmin()
    return id 

"""Starting Asset Allocation Weights using TRowePrice 2020 Target TDF Weights; 
Returns equity, bond and cash weights respectively"""
def weights_trowprice_target():
    start = [0.98,0.98,0.98,0.96,0.92,0.84,0.74,0.63,0.52,0.425,0.38,0.36,0.34]#,0.33,0.32,0.30]
    allWeights = []
    for i in range(len(start)-1):
        equity = np.linspace(start[i],start[i+1],5)
        bond = np.linspace(1-start[i],1-start[i+1],5)
        riskless = np.zeros(5)
        stacked = np.column_stack((equity, bond, riskless))
        allWeights.extend(stacked)
    # np.savetxt("troweprice_target_weights.csv", np.array(allWeights), delimiter=",")
    return np.array(allWeights)

"""Models Income Over 60-Year Life-Cycle"""
def stochastic_income(working_age=20, retirement_age=65, death_age=80):
    work_age = np.linspace(working_age, retirement_age, retirement_age-working_age)
    work_income = work_age * 0.1682 -0.0323 * (work_age **2)/10 + 0.002 *(work_age**3)/100 +2.7004 + -2.17 
    work_income = np.exp(work_income)
    retirment_amount =  (work_income[-1]) *  0.68212 
    retirement_time  = np.full(death_age-retirement_age, retirment_amount )
    return np.concatenate((work_income, retirement_time))

###########################################################
###II: POLICY RULE AND SIMULATION FOR ONE SET OF WEIGHTS###
###########################################################
"""Global Setup Function containing Policy Rule and Simulation"""
def setup(weights=weights_trowprice_target(), gamma=4, numModels=2, artificial=False, filter=False):
    # start = time.time()
    #Investor Preferences setup
    gamma = gamma
    income = stochastic_income()
    numages = len(income) 

    #Asset Grid for Policy Rule Setup
    lowest_wealth = 0
    highest_wealth = 350
    numPossibleAssets = 350
    assets = np.linspace(lowest_wealth, highest_wealth, numPossibleAssets)
    
    pos_equity_mean = 0.096
    pos_bond_mean = 0.018
    neg_equity_mean = 0.07
    neg_bond_mean = 0.019
    
    ###SETH"S DATA (2 States)
    # positive_df = pd.read_excel("positive.xlsx", header=None)
    # negative_df = pd.read_excel("negative.xlsx", header=None)
    # positive_equity = positive_df.iloc[:, 0].tolist()
    # positive_bond = positive_df.iloc[:, 1].tolist()
    # positive_riskless = np.zeros(len(positive_equity))#positive_df.iloc[:, 2]
    # negative_equity = negative_df.iloc[:, 0].tolist()
    # negative_bond = negative_df.iloc[:, 1].tolist()
    # negative_riskless = np.zeros(len(negative_equity))#negative_df.iloc[:, 2]

   # JESS' DATA (2 States)
    if artificial == False:
        print("natural")
        positive_df = pd.read_csv("positive.csv")
        negative_df = pd.read_csv("negative.csv")
        positive_equity = positive_df["Real_Annual_Equity"].tolist()
        positive_bond = positive_df["Real_Annual_Bond"].tolist()
        positive_riskless = np.zeros(len(positive_equity)).tolist()#positive_df.iloc[:, 2]
        negative_equity = negative_df["Real_Annual_Equity"].tolist()
        negative_bond = negative_df["Real_Annual_Bond"].tolist()
        negative_riskless = np.zeros(len(negative_equity)).tolist()#negative_df.iloc[:, 2]
    #ONE STATE - JESS/SETH
        if numModels == 1:
            print("One-state")
            positive_equity = positive_equity + negative_equity
            positive_bond= positive_bond + negative_bond
            positive_riskless = np.zeros(len(positive_bond))
            negative_bond= positive_bond
            negative_equity = positive_equity
            negative_riskless = positive_riskless
    #ARTIFICIALLY GENDERATED - 2 States
    elif artificial == True and numModels == 2:
        print("artificial")
        draws = np.random.multivariate_normal([pos_equity_mean,pos_bond_mean], [[0.186**2, -0.64], [-0.64, 0.076**2]], 100)
        positive_equity = draws[:,0]
        positive_bond = draws[:,1]
        positive_riskless = np.zeros(len(positive_equity)).tolist()
        neg_draws = np.random.multivariate_normal([neg_equity_mean,neg_bond_mean], [[0.205**2, 0.66], [0.66, 0.104**2]], 100)
        negative_equity = neg_draws[:,0]
        negative_bond = neg_draws[:,1]
        negative_riskless = np.zeros(len(positive_equity)).tolist()
    elif artificial == True and numModels == 1:
        #ARTIFICIALLY GENDERATED - 1 State
        draws = np.random.multivariate_normal([0.084,0.019], [[0.194**2, 0.09], [0.09, 0.089**2]], 250)
        positive_equity = negative_equity = draws[:,0]
        positive_bond = negative_bond = draws[:,1]
        positive_riskless = negative_riskless = np.zeros(len(positive_equity)).tolist()
    else:
        print("error, not a valid option")

    #Stacking 3 asset classes together
    positive_returns = np.column_stack((positive_equity, positive_bond, positive_riskless))
    negative_returns  = np.column_stack((negative_equity, negative_bond, negative_riskless))
    if filter == True:
        print("filtered")
        positive_returns = positive_returns[((positive_returns[:,0] >= pos_equity_mean) & (positive_returns[:,1] >= pos_bond_mean)) | ((positive_returns[:,0] <= pos_equity_mean) & (positive_returns[:,1] <= pos_bond_mean))]
        negative_returns = negative_returns[((negative_returns[:,0] >= neg_equity_mean) & (negative_returns[:,1] <= neg_bond_mean)) | ((negative_returns[:,0] <= neg_equity_mean) & (negative_returns[:,1] >= neg_bond_mean))]

    print(len(positive_bond), len(positive_equity), len(positive_riskless))
    print(len(negative_bond), len(negative_equity), len(negative_riskless))

    prob_positive = 1/len(positive_bond)
    prob_negative = 1/len(negative_bond)

    #Macro-environment setup
    beta = 0.96
    if True:
        print("not persistent")
        prob_given_pos =[0.886,0.114] #[1,0]#
        prob_given_neg = [0.222,0.778] #[0,1]#
    else: 
        print("persistent")
        prob_given_pos =[0.99,0.01]#
        prob_given_neg = [0.01,0.99]#
    # corrs = np.array([0,0])
    corrs = np.array([0.66,-0.64])
    numCorrelations = len(corrs)
    num_jobs = 1 #For parallel processing
    opt_savings = np.empty((numages, numCorrelations, numPossibleAssets)) #Initialization for policy rule
    vectorized_bisect_left = np.vectorize(lambda x: bisect.bisect_left(assets, x))
    
    startingage = 0
    #Find all possible returns weighted at each age for each correlation
    returns_by_age = []
    outcomes = []
    for corr in [positive_returns, negative_returns]: #Explicit for loop to address irregular shapes
        # print("Number of Datapoints for this state of the world:", corr.shape)
        broadcasted_weights= np.tile(weights, (len(corr), 1, 1)) #dimensions = (numDataPoints - 16 for positive, 13 for negative, 33 for opposite, numAges, 3Assets)
        broadcasted_returns = broadcasted_weights * (1 + corr[:, np.newaxis, :]) #dimensions = (numDataPoints, numAges, 3Assets)
        returns = np.sum(broadcasted_returns, axis=2) - 1 #dimensions = (numDataPoints, numAges)
        savings = assets #to avoid overwriting
        end_value = savings[:, np.newaxis, np.newaxis]*(1 + returns) #dimensions = (numAssets,numDataPoints,numAge) 
        end_value = np.transpose(end_value, (2,0,1)) #dimensions = (numAge, numAssets, numDataPoints) 
        outcomes.append(end_value)
        # print(end_value.shape)
        return_by_age = np.transpose(returns) #dimensions =(numAges, numDataPoints); for simulation
        returns_by_age.append(return_by_age)

    """
    Policy Rule For Finding Optimal Savings given All Possible Correlations and This age's Savings,
    By Backsolving Each Age Recursively
    Returns First age Utility for Each Correlation State and Possible First age Savings
    """
    def policy_rule(age, total_utility): 
        # start = time.time()
        #Initialization to store optimal savings and resulting utility
        optimal_saving = np.empty((numCorrelations, numPossibleAssets))
        new_total_utility = np.empty((numCorrelations, numPossibleAssets))
        
        #Find indices of all possible outcomes at a given age
        closest_indices = []
        # print(len(outcomes))
        for i in range(len(outcomes)): #Explicit for loop b/c irregularly shaped outcome matrix caused by diff number of datapoints for correlation states
            this_age_outcomes = outcomes[i][age] #Find Set of Possible Outcomes for this age
            indices = np.apply_along_axis(vectorized_bisect_left, axis=1, arr=this_age_outcomes) #Snap outcomes to indices
            indices_before = np.clip(indices, 0, len(savings) - 1) # Get all valid indices one before the one returned by bisect_left to check if found closest indices
            differences_before = np.abs(savings[indices_before] - this_age_outcomes) #Calculate diff b/w actual outcome and outcome at index before
            indices_after = np.clip(indices + 1, 0, len(savings) - 1) # Get all valid indices one after the one returned by bisect_left to check if found closest indices
            differences_after = np.abs(savings[indices_after] - this_age_outcomes) #Calculate diff b/w actual outcome and outcome at index after
            min_diff_indices = np.where(differences_before <= differences_after, indices_before, indices_after)  #Return the indices w/minimal difference
            closest_indices.append(min_diff_indices)
            # print(min_diff_indices.shape)

        #Find the optimal saving amount and the new utility at an age for each correlation
        for k in range(len(corrs)):
            prob_given_current_state = prob_given_pos if k == 0 else prob_given_neg 
            results = Parallel(n_jobs=num_jobs, backend='multiprocessing')(
            delayed(parallel_processing)(
               prob_given_current_state, current_wealth, assets, income, age, total_utility, closest_indices, prob_positive, prob_negative, beta
            ) for current_wealth in assets)
            optimums_list, new_total_utility_list = zip(*results) #Unpack the results
            optimal_saving[k] = optimums_list
            new_total_utility[k] = new_total_utility_list
        opt_savings[age, :] = optimal_saving
        if age == startingage: #startingage = 50
            return np.array(new_total_utility)  #Base Case (End of Recursive Policy Rule); Return Final Utility in Year 0
        return policy_rule(age-1, np.array(new_total_utility))

    """
    Parallel Processing for Each Possible Asset at an Age
    Returns the optimal saving amount and the new utility at this age
    """
    def parallel_processing(prob_given_current_state, current_wealth, assets, income, age, total_utility, closest_indices, prob_positive, prob_negative, beta):
        c = current_wealth  + income[age] - assets
        try:
            utility = np.where(c <= 0, float("-inf"), (c**(1 - gamma)) / (1 - gamma)) 
        except RuntimeWarning as rw:
            utility = np.zeros(c)
        expected_utility = (
            prob_given_current_state[0] * np.mean(total_utility[0][closest_indices[0]], axis=1) +
            prob_given_current_state[1] * np.mean(total_utility[1][closest_indices[1]], axis=1) 
        )
        val = utility + beta * expected_utility
        opt_saving = np.argmax(val)
        new_total_utility =  np.max(val)
        return opt_saving, new_total_utility
        
    """
    Simulation Function For One Human
    """
    def simulation(age, saving, corr, consumptions, wealths, saving_rate, returns):
        # print(age)
        probability = prob_given_pos if corr ==  corrs[0] else prob_given_neg
        random_index = random.choices([0,1], probability)[0]
        this_age_correlation = corrs[random_index]
        saving_on_grid = int(opt_savings[age][random_index][saving])
        wealths[age-startingage] = saving_on_grid
        saving_amount = assets[saving_on_grid]
        labour_income = income[age]
        if age == startingage:
            consumption = assets[saving] + income[age] - saving_amount
            saving_rate[age-startingage] = (labour_income - consumption) / (labour_income)
            returns[age-startingage] = 0
        else:
            this_age_returns = returns_by_age[random_index][age]
            interest_rate = float(random.choice(this_age_returns))
            capital_gain = assets[saving] * (1+interest_rate)
            consumption = capital_gain + labour_income - saving_amount
            # saving_rate[age-startingage] = (capital_gain + labour_income - consumption) / (capital_gain + labour_income)
            saving_rate[age-startingage] = saving_amount / (capital_gain + labour_income)
            returns[age-startingage] = interest_rate
        consumptions[age-startingage] = consumption
        if age != numages-1:
            return simulation(age+1, int(saving_on_grid), this_age_correlation, consumptions, wealths, saving_rate, returns)
        else: 
            return consumptions, wealths, saving_rate, returns

    """
    Run policy_rule and simulation for one set of weights
    Returns starting period utility averaged across 3 correlation states
    """
    def run_one_weight_set(weights=weights_trowprice_target()):
        numHumans = 100000
        initial_wealth = 0
        wealth_list = []
        consumption_list = []
        saving_rate_list = []
        startingUtility = np.vstack([np.where(assets < 0, float("-inf"), 0)] * 3)
        final_utility = policy_rule(numages-1, startingUtility)
        for human in range(numHumans):
            consumptions = np.empty(len(income)-startingage)#len(income))
            wealths = np.empty(len(income)-startingage)
            saving_rate = np.empty(len(income)-startingage)
            returns = np.empty(len(income)-startingage)
            starting_corr = np.random.choice(corrs, p=[0.698,0.302]) #0 #Referenced
            simulation(startingage, int(point_on_grid(assets, initial_wealth)),starting_corr,consumptions, wealths, saving_rate, returns)
            savings = []
            [savings.append(assets[int(wealth)]) for wealth in wealths]
            wealth_list.append(savings)
            consumption_list.append(consumptions)
            saving_rate_list.append(saving_rate)
        if numHumans != 0: #Graphically Check Results
            averageWealth = np.average(np.array(wealth_list), axis=0)
            averageConsumption = np.average(np.array(consumption_list), axis=0)
            # print(averageWealth)
            x_values = range(20, 20 + len(averageWealth), 1)
            # plt.plot(x_values, averageWealth)
            # plt.plot(x_values, averageConsumption, color="black")
            # plt.xlabel('Age')
            # plt.ylabel('Monetary Amount')
            # plt.legend(['Optimal Saving', 'Consumption'])
            # # plt.xticks(x_values)
            # plt.show()
            averageSavingRate = np.convolve(np.array(saving_rate), np.ones(2)/2, mode='valid')
            # plt.xlabel("Age")
            # plt.ylabel("Average Saving Rate")
            # plt.plot(averageSavingRate)
            # plt.show()
        utiility = 0.698 * final_utility[0][point_on_grid(assets, initial_wealth)] + 0.302 * final_utility[1][point_on_grid(assets, initial_wealth)] 
        return utiility
        # return averageWealth, averageConsumption, averageSavingRate, returns
    return run_one_weight_set(weights)
    
##############################
###III: WEIGHTS###
##############################
def weights_blackrock():
    equity = [0.99,0.9856,0.9874,0.9868,0.9662,0.9035,0.8081,0.6976,0.5812,0.4537,0.3981,0.3981,0.3981]
    bonds = [0.0101,0.0101,0.0103,0.031,0.0938,0.1901,0.3012,0.417,0.5447,0.6015,0.6015,0.6015,0.6015]
    cash = [0.0043,0.0025,0.0029,0.0028,0.0027,0.0018,0.0012,.0018,0.0016,0.0004,0.0004,0.0004,0.0004]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("blackrock_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_statestreet():
    equity = [0.9,0.9,0.9,0.8974,0.8475,0.7962,0.7212,0.6274,0.4775,0.3438, 0.35,0.35,0.35]
    bonds = [0.1,0.1,0.1,0.1,0.1,0.1025,0.1525,0.2037,0.2788,0.3563,0.4416,0.4619,0.45]
    cash = [0,0,0,0,0,0,0,0,0,0,0.0396,0.1444,0.2]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("statestreet_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_ameircanfunds():
    equity = [0.871,0.871,0.871,0.863,0.853,0.813,0.694,0.592,0.499,0.448,0.406,0.388,0.388]
    bonds = [0.088,0.088,0.088,0.094,0.104,0.144,0.261,0.363,0.455,0.503,0.544,0.562,0.562]
    cash = [0.041,0.041,0.042,0.043,0.043,0.043,0.045,0.045,0.047,0.049,0.049,0.05,0.05]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("americanfunds_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_schwab(): ##Changed
    equity = [0.8993,0.8904,0.8712,0.9141,0.8735,0.8099,0.7318,0.6367,0.5064,0.4312,0.4047,0.3675,0.3675]
    bonds = [0.0922,0.1035,0.1218,0.0783,0.1114,0.1717,0.2404,0.3203,0.4198,0.4743,0.4927,0.5228,0.5228]
    cash = [0.0085,0.0061,0.0071,0.0077,0.015,0.0186,0.0278,0.0429,0.0737,0.0944,0.1026,0.1097,0.1097]

    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("schwab_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_voya():
    equity = [0.95,0.95,0.94,0.94,0.91,0.85,0.75,0.64,0.53,0.33,0.33,0.33,0.33]
    bonds = [0.05,0.05,0.06,0.06,0.09,0.15,0.25,0.34,0.44,0.58,0.58,0.58,0.58]
    cash = [0,0,0,0,0,0,0,0.02,0.03,0.09,0.09,0.09,0.09]

    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("voya_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_tiaa(): ####BACKWARDS
    equity = [0.9969,0.9848,0.9728,0.9606,0.9259,0.8523,0.7389,0.6327,0.5421,0.4757,0.4257,0.3756,0.4051]
    bonds = [0.0031,0.0152,0.0272,0.0394,0.0741,0.1477,0.2494,0.3359,0.4067,0.4532,0.4834,0.496,0.4959]
    cash = [0,0,0,0,0,0,0.0117,0.0314,0.0512,0.0711,0.0909,0.1284,0.099]

    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("tiaa_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_pimco():
    equity = [0.96,0.96,0.952,0.938,0.91,0.853,0.767,0.675,0.562,0.422,0.422,0.422,0.422]
    bonds = [0.04,0.04,0.048,0.061,0.09,0.146,0.234,0.324,0.436,0.579,0.579,0.579,0.579]
    cash = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("pimco_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights


def weights_fidelity():
    equity = [0.9,0.9,0.9,0.9,0.9,0.87,0.74,0.61,0.55,0.47,0.38,0.28,0.28]
    bonds = [0.1,0.1,0.1,0.1,0.1,0.1,0.13,0.26,0.4,0.45,0.49,0.49,0.5]
    cash = [0,0,0,0,0,0,0,0,0,0,0.05,0.12,0.21]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("fidelity_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_vanguard():
    equity = [0.902,0.903,0.901,0.901,0.854,0.778,0.705,0.631,0.538,0.404,0.295,0.295,0.295]
    bonds = [0.098,0.097,0.099,0.099,0.146,0.222,0.295,0.369,0.462,0.596,0.705,0.705,0.705]
    cash = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("vanguard_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_troweprice():
    equity = [0.981,0.981,0.981,0.975,0.959,0.897,0.804,0.691,0.586,0.528,0.496,0.466,0.429]
    bonds = [0.007,0.01,0.009,0.011,0.027,0.087,0.176,0.287,0.391,0.45,0.483,0.511,0.512]
    cash = [0.013,0.01,0.009,0.014,0.014,0.017,0.02,0.023,0.021,0.022,0.022,0.023,0.06]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    # np.savetxt("troweprice_weights.csv", np.array(all_weights), delimiter=",")
    return all_weights

def weights_all_equity():
    equity = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    bonds = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cash = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    return all_weights

def weights_all_bonds():
    equity = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    bonds = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    cash = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    return all_weights

def weights_all_cash():
    equity = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    bonds = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    cash = [1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    return all_weights

def weights_60_40():
    equity = [0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6]
    bonds = [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,]
    cash = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    all_weights = np.array([]).reshape(-1, 3)
    for i in range(12):
        equity_weights  = np.linspace(equity[i],equity[i+1],5)
        bond_weights = np.linspace(bonds[i],bonds[i+1],5)
        cash_weights = np.linspace(cash[i],cash[i+1],5)
        weights = np.column_stack((equity_weights, bond_weights, cash_weights))
        all_weights= np.concatenate((all_weights, weights))
    return all_weights

def find_and_calculate_CEV(arr, gamma=4):
    biggest_element = max(arr)
    cev = [(biggest_element / element)**(1/(1-gamma))-1 for element in arr]
    return cev

"""Main For Testing"""
def main():
    american = weights_ameircanfunds()
    blackrock = weights_blackrock()
    fidelity = weights_fidelity()
    voya = weights_voya()
    vanguard = weights_vanguard()
    troweprice = weights_troweprice()
    tiaa = weights_tiaa()
    schwab = weights_schwab()
    statestreet = weights_statestreet()
    pimco = weights_pimco()
    all_equity = weights_all_equity()
    all_bonds = weights_all_bonds()
    all_cash = weights_all_cash()
    sixty_fourty = weights_60_40()
    weights = [american, blackrock, fidelity, voya, vanguard, troweprice, tiaa, schwab, statestreet, pimco, all_equity, all_bonds, all_cash, sixty_fourty]
    utilities = []
    for gamma in [4]:
        print("Gamma", gamma)
        for weight in weights: 
            for i in range(0,2):
                util = setup(weights=weight, gamma = gamma, numModels=2, artificial=False,filter=False)
                utilities.append(util)
                # averageWealth, averageConsumption, averageSavingRate, returns = setup(weights=weight, gamma = gamma, numModels=i, artificial=True,filter=False)
    #             plt.figure(fig1.number)
    #             # averageWealth = np.linspace(0,20,20)
    #             # averageConsumption = np.linspace(1,21,20)
    #             # averageSavingRate = np.linspace(0,1,20)
    #             x_values = range(death_age-len(averageWealth), death_age, 1)
    #             linestyle = '-' if i == 1 else '--' 
    #             wealth_label = f"Optimal Savings - {'One' if i == 1 else 'Two'} State"
    #             consumption_label = f"Consumption - {'One' if i == 1 else 'Two'} State"
    #             saving_rate_label = f"Saving Rate - {'One' if i == 1 else 'Two'} State"
    #             plt.plot(x_values, averageWealth, label=wealth_label, color="blue", linestyle=linestyle)
    #             plt.plot(x_values, averageConsumption, label=consumption_label, color="black", linestyle=linestyle)
    #             print("averageWealth", np.average(averageWealth))
    #             print("averageConsumption", np.average(averageConsumption))
    #             plt.ylabel("Monetary Amount ($)")
    #             plt.xlabel("Age")
    #             plt.ylim(0, 250) 
    #             plt.xlim(20,80)
    #             plt.figure(fig2.number)
    #             plt.plot(range(death_age-len(averageSavingRate), death_age, 1), averageSavingRate, label=saving_rate_label, color="blue", linestyle=linestyle)
    #             plt.ylabel("Saving Rate %")
    #             plt.xlabel("Age")
    #             plt.ylim(0, 3) 
    #             plt.xlim(20,80)
    #             print("averageSavingRate", np.average(averageSavingRate))
    #             print("averageReturns", np.average(returns))
    #         # plt.figure(fig1.number)
    #         # plt.legend()
            
    #         # plt.figure(fig2.number)
    #         # plt.legend()
    #     plt.show()
    cevs = find_and_calculate_CEV(utilities)
    for cev in cevs: 
        print(cev)
if __name__ == '__main__':
    main()

    