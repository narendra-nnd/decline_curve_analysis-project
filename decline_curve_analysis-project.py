
"""
Arps Model Selector and Forecaster
Description:
    This script uses synthetic production data (time vs rate),
    fits Exponential, Harmonic, and Hyperbolic Arps decline models,
    selects the best one using mean squared error,
    and uses it to forecast future production rate and cumulative production.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------------------
# Arps Decline Models
def exp_model(t, qi, Di):
    return qi * np.exp(-Di * t)

def harm_model(t, qi, Di):
    return qi / (1 + Di * t)

def hyp_model(t, qi, Di, b):
    return qi / ((1 + b * Di * t) ** (1 / b))

# ----------------------------
# Cumulative Production Functions
def cumulative_exp(qi, Di, t):
    return (qi - exp_model(t, qi, Di)) / Di

def cumulative_harm(qi, Di, t):
    return qi / Di * np.log(1 + Di * t)

def cumulative_hyp(qi, Di, b, t):
    if b == 1:
        return cumulative_harm(qi, Di, t)
    return (qi * ((1 + b * Di * t) ** ((1 - b) / b) - 1)) / ((1 - b) * Di)

# ----------------------------
# Generate Synthetic Production Data
np.random.seed(42)  # For reproducibility
t_data = np.linspace(0, 3, 37)  # Time: 0 to 3 years, monthly
true_qi, true_Di, true_b = 1200, 0.25, 0.5
q_data = hyp_model(t_data, true_qi, true_Di, true_b) + np.random.normal(0, 20, len(t_data))

# ----------------------------
# Fit All Decline Models
popt_exp, _ = curve_fit(exp_model, t_data, q_data, p0=[1200, 0.2], bounds=(0, np.inf))
popt_harm, _ = curve_fit(harm_model, t_data, q_data, p0=[1200, 0.2], bounds=(0, np.inf))
popt_hyp, _ = curve_fit(hyp_model, t_data, q_data, p0=[1200, 0.2, 0.5], bounds=(0, [np.inf, 1, 1]))

# ----------------------------
# Calculate Mean Squared Errors
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_dict = {
    'Exponential': mse(q_data, exp_model(t_data, *popt_exp)),
    'Harmonic': mse(q_data, harm_model(t_data, *popt_harm)),
    'Hyperbolic': mse(q_data, hyp_model(t_data, *popt_hyp))
}
best_model = min(mse_dict, key=mse_dict.get)

# ----------------------------
# Forecast Future Production
t_future = np.linspace(0, 8, 100)
if best_model == 'Exponential':
    qi, Di = popt_exp
    q_future = exp_model(t_future, qi, Di)
    Gp_future = cumulative_exp(qi, Di, t_future)
elif best_model == 'Harmonic':
    qi, Di = popt_harm
    q_future = harm_model(t_future, qi, Di)
    Gp_future = cumulative_harm(qi, Di, t_future)
elif best_model == 'Hyperbolic':
    qi, Di, b = popt_hyp
    q_future = hyp_model(t_future, qi, Di, b)
    Gp_future = cumulative_hyp(qi, Di, b, t_future)

# ----------------------------
# Plot Results
plt.figure(figsize=(14, 6))

# Rate Plot
plt.subplot(1, 2, 1)
plt.scatter(t_data, q_data, color='black', label='Observed Data', s=25)
plt.plot(t_future, q_future, label=f'{best_model} Fit', color='red')
plt.xlabel('Time (years)')
plt.ylabel('Production Rate (bbl/day)')
plt.title(f'Best Fit: {best_model} Decline Model')
plt.grid(True)
plt.legend()

# Cumulative Plot
plt.subplot(1, 2, 2)
plt.plot(t_future, Gp_future, label='Cumulative Forecast', color='green')
plt.xlabel('Time (years)')
plt.ylabel('Cumulative Production (bbl)')
plt.title('Cumulative Production Forecast')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# Print Summary
print(f"\nBest fitting model: {best_model}")
if best_model == 'Hyperbolic':
    print(f"Fitted Parameters: qi = {qi:.2f}, Di = {Di:.3f}, b = {b:.2f}")
else:
    print(f"Fitted Parameters: qi = {qi:.2f}, Di = {Di:.3f}")

index = np.searchsorted(t_future, 5)
print(f"\nEstimated production rate at 5 years: {q_future[index]:.2f} bbl/day")
print(f"Cumulative production at 5 years: {Gp_future[index]:,.2f} bbl")
