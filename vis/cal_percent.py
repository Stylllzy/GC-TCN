# Function to calculate improvement for minimized metrics (MAE, RMSE, MAPE)
def calculate_minimization_improvement(old, new):
    return ((old - new) / old) * 100


# Function to calculate improvement for maximized metrics (R?)
def calculate_maximization_improvement(old, new):
    return ((new - old) / old) * 100


# CEEMDAN-TAE scores
ceemdan_tae = [0.4073, 0.5323, 0.0101, 0.9507]

# Calculate the improvement percentages for each model
improvement = {
    "CEEMDAN-SVR": [
        calculate_minimization_improvement(1.9472, ceemdan_tae[0]),  # MAE
        calculate_minimization_improvement(2.4700, ceemdan_tae[1]),  # RMSE
        calculate_minimization_improvement(0.0464, ceemdan_tae[2]),  # MAPE
        calculate_maximization_improvement(0.3375, ceemdan_tae[3])  # R?
    ],
    "CEEMDAN-LSTM": [
        calculate_minimization_improvement(0.9603, ceemdan_tae[0]),  # MAE
        calculate_minimization_improvement(1.2262, ceemdan_tae[1]),  # RMSE
        calculate_minimization_improvement(0.0241, ceemdan_tae[2]),  # MAPE
        calculate_maximization_improvement(0.7384, ceemdan_tae[3])  # R?
    ],
    "CEEMDAN-GRU": [
        calculate_minimization_improvement(1.1168, ceemdan_tae[0]),  # MAE
        calculate_minimization_improvement(1.3681, ceemdan_tae[1]),  # RMSE
        calculate_minimization_improvement(0.0282, ceemdan_tae[2]),  # MAPE
        calculate_maximization_improvement(0.6744, ceemdan_tae[3])  # R?
    ],
    "CEEMDAN-TCN": [
        calculate_minimization_improvement(0.9028, ceemdan_tae[0]),  # MAE
        calculate_minimization_improvement(1.1227, ceemdan_tae[1]),  # RMSE
        calculate_minimization_improvement(0.0225, ceemdan_tae[2]),  # MAPE
        calculate_maximization_improvement(0.7807, ceemdan_tae[3])  # R?
    ]
}

# Create a pandas DataFrame to represent the improvement percentages
import pandas as pd

# Convert to a DataFrame
improvement_df = pd.DataFrame.from_dict(improvement, orient='index', columns=["MAE", "RMSE", "MAPE", "R2"])
print(improvement_df)
