import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Function to generate correlated Brownian motion
def generate_brownian_motion(n_steps, n_paths, correlation):
    clamped_correlation = max(-1, min(1, correlation))
    z1 = np.random.normal(0, 1, (n_steps, n_paths))
    z2 = clamped_correlation * z1 + np.sqrt(1 - clamped_correlation**2) * np.random.normal(0, 1, (n_steps, n_paths))
    return z1, z2

# Function to simulate stock price paths using the Heston model
def stochastic_volatility_simulation(S0, V0, kappa, theta, sigma_vol, rho, r, T, M, N):
    dt = T / M
    stock_paths = np.zeros((M + 1, N))
    vol_paths = np.zeros((M + 1, N))

    stock_paths[0] = S0
    vol_paths[0] = V0

    z1, z2 = generate_brownian_motion(M, N, rho)

    for t in range(1, M + 1):
        vol_paths[t] = vol_paths[t - 1] + kappa * (theta - vol_paths[t - 1]) * dt + sigma_vol * np.sqrt(np.maximum(vol_paths[t - 1], 0)) * np.sqrt(dt) * z2[t - 1]
        stock_paths[t] = stock_paths[t - 1] * np.exp((r - 0.5 * vol_paths[t - 1]) * dt + np.sqrt(np.maximum(vol_paths[t - 1], 0)) * np.sqrt(dt) * z1[t - 1])

    return stock_paths, vol_paths

# Function to calculate the option price
def calculate_option_price(stock_paths, K, r, T):
    payoff = np.maximum(stock_paths[-1] - K, 0)  # European call option payoff at maturity
    option_price = np.exp(-r * T) * np.mean(payoff)  # Discounted expected payoff
    return option_price

# Streamlit app
st.title("Stochastic Volatility Option Pricing Simulator")

# Sidebar for user inputs with exception handling
st.sidebar.header("Parameters")
try:
    S0 = st.sidebar.number_input("Initial Stock Price (S0)", value=100.0)
    V0 = st.sidebar.number_input("Initial Variance (V0)", value=0.04)
    K = st.sidebar.number_input("Strike Price (K)", value=110.0)
    r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
    T = st.sidebar.number_input("Maturity (T in Years)", value=1.0)
    M = st.sidebar.number_input("Number of Time Steps (M)", value=250)
    N = st.sidebar.number_input("Number of Simulation Paths (N)", value=100, min_value=1, max_value=1000)
    kappa = st.sidebar.number_input("Mean Reversion Speed (kappa)", value=2.0)
    theta = st.sidebar.number_input("Long-run Variance (theta)", value=0.04)
    sigma_vol = st.sidebar.number_input("Volatility of Variance (sigma_vol)", value=0.5)
    rho = st.sidebar.number_input("Correlation (rho)", value=-0.7)

    # Validate inputs
    if not (-1 <= rho <= 1):
        st.sidebar.error("Correlation (rho) must be between -1 and 1.")
    else:
        # Generate the simulation
        if st.button("Run Simulation"):
            stock_paths, vol_paths = stochastic_volatility_simulation(S0, V0, kappa, theta, sigma_vol, rho, r, T, M, N)

            # Plotting with Plotly
            fig = go.Figure()

            time_steps = np.linspace(0, T, M + 1)

            # Plot only a subset of the paths for better visualization
            subset_to_plot = min(N, 20)  # Plot up to 20 paths
            for i in range(subset_to_plot):
                fig.add_trace(go.Scatter3d(
                    x=time_steps,
                    y=stock_paths[:, i],
                    z=vol_paths[:, i],
                    mode='lines',
                    line=dict(width=2),
                    opacity=0.6  # Reduce opacity for clarity
                ))

            # Customize the layout
            fig.update_layout(
                title="3D Visualization of Stock Price Trajectories in the Heston Framework",
                scene=dict(
                    xaxis_title='Time (Years)',
                    yaxis_title='Stock Price',
                    zaxis_title='Variance'
                ),
                width=800,
                height=600
            )

            # Show the interactive plot in Streamlit
            st.plotly_chart(fig)

            # Calculate and display the option price
            option_price = calculate_option_price(stock_paths, K, r, T)
            st.write(f"Option Price: ${option_price:.2f}")

except Exception as e:
    st.sidebar.error(f"An error occurred: {e}")

