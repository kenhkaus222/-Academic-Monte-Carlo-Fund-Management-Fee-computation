import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

# Global parameters - unified and optimized
W_0 = 1000              # Initial wealth
RF = 0.05               # Risk-free rate (unified with discount rate)
SIGMA = 0.20            # Fund SIGMA
YEARS = 10              # Investment horizon
TOLERANCE = 0.5         # Convergence tolerance
PATHS = 10001           # Number of Monte Carlo paths
DT = 1.0                # Time step (annual)

# Function to simulate fund values paths
def simulate_fund_paths(fee_rate, n_sims):

    np.random.seed(42)  # Ensure reproducibility
    
    # Generate independent standard normal random variables
    # Z_t ~ N(0,1) for t = 0, M-1 across N paths
    Z = np.random.standard_normal((n_sims, YEARS))
    
    # Initialize fund values matrix
    fund_values = np.zeros((n_sims, YEARS + 1))
    fund_values[:, 0] = W_0  # Initial value V_0 = W_0
    
    # Simulate paths using the discrete log-return model
    for t in range(YEARS):
        log_return = (RF - fee_rate - 0.5 * SIGMA**2) * DT + SIGMA * np.sqrt(DT) * Z[:, t]
        fund_values[:, t+1] = fund_values[:, t] * np.exp(log_return)
    
    return fund_values

# Function to calculate present_values for payoffs and fees
def calculate_present_values(fee_rate):

    # Call fund_values paths from simulate_fund_paths
    fund_values = simulate_fund_paths(fee_rate, PATHS)
    n_sims, _ = fund_values.shape
    
    # Calculate PV of fees
    pv_fees = np.zeros(n_sims)
    for j in range(n_sims):  # For each simulation path j
        for i in range(YEARS):  # For each time step i = 0 to M-1
            discount_factor = np.exp(-RF * i * DT)  # e^{-r·i·DT}
            fee_amount = fund_values[j, i] * fee_rate * DT  # V_i^j · f · DT
            pv_fees[j] += discount_factor * fee_amount
    
    # Calculate PV of payoffs
    pv_payoffs = np.zeros(n_sims)
    for j in range(n_sims):  # For each simulation path j
        max_value = np.max(fund_values[j, :])  # max_{i=0,...,M}V_i^j
        terminal_value = fund_values[j, -1]    # V_M^j
        payoff = max_value - terminal_value    # Guarantee benefit
        pv_payoffs[j] = np.exp(-RF * YEARS) * payoff  # e^{-r·T} · payoff
    
    return np.mean(pv_fees), np.mean(pv_payoffs), pv_fees, pv_payoffs

# Function to calculate the absolute difference between mean of pv_fees and mean of pv_payoffs
def objective_function(fee_rate):

    # Call mean of pv_fees and mean of pv_payoffs from function: calculate_present_values
    mean_pv_fees, mean_pv_payoffs, _, _ = calculate_present_values(fee_rate)
    return abs(mean_pv_fees - mean_pv_payoffs)

# Optimization function to perform numerical method with scalar minization
def find_optimal_fee_numerical():
    
    # Display notes
    print("=== NUMERICAL APPROACH ===")
    print("Employing bounded scalar optimization method...")
    
    result = minimize_scalar(
        objective_function,
        bounds=(0.001, 0.10),
        method='bounded',
        options={'xatol': 1e-6}
    )
    
    optimal_fee = result.x
    min_difference = result.fun
    
    print(f"Optimal fee rate: {optimal_fee:.4f} ({optimal_fee*100:.2f}%)")
    print(f"Convergence achieved with difference: ${min_difference:.2f}")
    
    return optimal_fee


# Optimization function to perform iterative method
def find_optimal_fee_iterative():

    # Call global empty list: covergence_data to local
    global convergence_data
    
    # Display notes
    print("\n=== ITERATIVE APPROACH ===")
    print("Implementing the bisection optimization method...")
    
    # Bounds for iteration and maximum iteration time
    low, high = 0.001, 0.10
    iteration = 0
    max_iterations = 50
    
    convergence_data = []  # Reset convergence list locally for the own use of this function
    
    # While loop to iterate the objective function until it reaches the tolerance
    while iteration < max_iterations:
        mid = (low + high) / 2
        difference = objective_function(mid)
        mean_pv_fees, mean_pv_payoffs, _, _ = calculate_present_values(mid)
        
        # Store convergence data for visualization
        convergence_data.append({
            'Iteration': iteration + 1,
            'Fee Rate (%)': mid * 100,
            'Difference ($)': difference,
            'PV Fees ($)': mean_pv_fees,
            'PV Payoffs ($)': mean_pv_payoffs,
            'Convergence': 'Yes' if difference <= TOLERANCE else 'No'
        })
        
        if difference <= TOLERANCE:
            print(f"Convergence achieved after {iteration + 1} iterations")
            print(f"Optimal fee rate: {mid:.4f} ({mid*100:.2f}%)")
            return mid
        
        # Bisection method value update
        if mean_pv_payoffs > mean_pv_fees:
            low = mid
        else:
            high = mid
        
        iteration += 1
    
    final_fee = (low + high) / 2
    return final_fee

# Consolidated data visualization
def visulization_summary(optimal_fee):

    # Generate fund paths for visualization
    sample_paths = simulate_fund_paths(optimal_fee, PATHS)
    
    # Create the main visualization
    fig = plt.figure(figsize=(16, 12))
    
    # Main plot: Fund paths
    ax_main = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
    
    years = np.arange(0, YEARS + 1)
    
    # Plot sample paths with better visibility
    for i in range(sample_paths.shape[0]):
        ax_main.plot(years, sample_paths[i, :], alpha=0.3, color='green', linewidth=1.2)
    
    # Plot average path and confidence intervals using dotted lines
    mean_path = np.mean(sample_paths, axis=0)
    std_path = np.std(sample_paths, axis=0)
    
    ax_main.plot(years, mean_path, color='red', linewidth=3, label='Expected Path', zorder=10)
    ax_main.plot(years, mean_path + std_path, color='red', linewidth=2, linestyle='--', 
                label='+1σ Confidence', alpha=0.8, zorder=9)
    ax_main.plot(years, mean_path - std_path, color='red', linewidth=2, linestyle='--', 
                label='-1σ Confidence', alpha=0.8, zorder=9)
    
    ax_main.set_xlabel('Years', fontsize=8, fontweight='bold')
    ax_main.set_ylabel('Fund Value ($)', fontsize=8, fontweight='bold')
    ax_main.set_title(f'Management Fee: {optimal_fee*100:.2f}% p.a. | Risk-Free Rate: {RF*100:.1f}% | Span: {YEARS} Years|'
                     f'SIGMA: {SIGMA*100:.1f}%', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlim(0, YEARS)
    
    # Add statistical annotations with better positioning
    final_stats = f'Initial: ${W_0:,}\nExpected Terminal: ${mean_path[-1]:,.0f}\nPaths: {sample_paths.shape[0]:,}'
    ax_main.text(0.02, 0.85, final_stats, transform=ax_main.transAxes, 
                verticalalignment='top', fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Convergence table
    ax_table = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    ax_table.axis('off')
    
    # Create convergence DataFrame
    df_convergence = pd.DataFrame(convergence_data)
    
    # Display last 10 iterations for clarity
    display_df = df_convergence.tail(7) if len(df_convergence) > 7 else df_convergence
    
    # Format the table
    table_data = []
    for _, row in display_df.iterrows():
        table_data.append([
            f"{int(row['Iteration'])}",
            f"{row['Fee Rate (%)']:.3f}%",
            f"${row['Difference ($)']:.2f}",
            f"${row['PV Fees ($)']:.2f}",
            f"${row['PV Payoffs ($)']:.2f}",
            row['Convergence']
        ])
    
    headers = ['Iter_num', 'Fee Rate', 'Abs Difference', 'PV Fees', 'PV Payoffs', 'Converged']
    
    table = ax_table.table(cellText=table_data,
                          colLabels=headers,
                          cellLoc='center',
                          loc='center',
                          colWidths=[0.08, 0.12, 0.15, 0.15, 0.15, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    ax_table.set_title('Convergence process', 
                      fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'optimal_fee': optimal_fee,
        'convergence_iterations': len(convergence_data)
    }

def main():
    """
    Main execution function implementing the complete fee optimization workflow.
    """
    print("ASX200 EQUITY FUND MANAGEMENT FEE OPTIMIZATION")
    print("=" * 60)
    
    print(f"\nModel Parameters:")
    print(f"• Initial Fund Value: ${W_0:,}")
    print(f"• Risk-Free Rate: {RF*100:.1f}% p.a.")
    print(f"• SIGMA: {SIGMA*100:.1f}% p.a.")
    print(f"• Investment Horizon: {YEARS} years")
    print(f"• Convergence Tolerance: ${TOLERANCE}")
    print(f"• Monte Carlo Paths: {PATHS:,}")
    
    # Execute both optimization approaches
    optimal_fee_numerical = find_optimal_fee_numerical()
    optimal_fee_iterative = find_optimal_fee_iterative()
    
    # Results comparison and validation
    print(f"\n=== RESULTS ANALYSIS ===")
    print(f"Numerical method: {optimal_fee_numerical*100:.3f}% p.a.")
    print(f"Iterative method: {optimal_fee_iterative*100:.3f}% p.a.")
    print(f"Method Convergence: {abs(optimal_fee_numerical - optimal_fee_iterative)*100:.4f}%")
    
    # Final result selection
    final_fee = round(optimal_fee_iterative * 100, 2) / 100
    print(f"\nOPTIMAL MANAGEMENT FEE: {final_fee*100:.2f}% per annum")
    
    # Generate comprehensive visualization
    results = visulization_summary(final_fee)
    
    print(f"\nOptimization completed successfully.")
    print(f"Convergence achieved in {results['convergence_iterations']} iterations.")

if __name__ == "__main__":
    main()