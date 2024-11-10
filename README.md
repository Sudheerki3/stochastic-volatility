# Stochastic Volatility Option Pricing Simulator

This project is a **Streamlit** application that simulates stock price paths using the **Heston model** and visualizes them in 3D using **Plotly**. The application also calculates the price of a European call option based on these simulated paths.

## Techniques and Methods Used

### 1. Correlated Brownian Motion
To simulate the stock paths and the volatility process, correlated Brownian motions \( Z_1 \) and \( Z_2 \) are generated. This is controlled by the correlation parameter \( \rho \). The equations for \( Z_2 \) are:

\[
Z_2 = \rho Z_1 + \sqrt{1 - \rho^2} \epsilon,
\]

where \( \epsilon \) is a standard normal random variable independent of \( Z_1 \).

### 2. Heston Model for Stochastic Volatility
The **Heston model** describes the dynamics of the stock price \( S(t) \) and its variance \( V(t) \) as follows:

\[
dS(t) = S(t) \left( r dt + \sqrt{V(t)} dW_1(t) \right),
\]
\[
dV(t) = \kappa (\theta - V(t)) dt + \sigma_{vol} \sqrt{V(t)} dW_2(t),
\]

where:
- \( r \) is the risk-free rate,
- \( \kappa \) is the mean reversion speed,
- \( \theta \) is the long-run variance,
- \( \sigma_{vol} \) is the volatility of variance,
- \( W_1(t) \) and \( W_2(t) \) are Wiener processes with correlation \( \rho \).

### 3. Option Pricing
The European call option payoff at maturity \( T \) is defined as:

\[
\text{Payoff} = \max(S(T) - K, 0),
\]

where \( K \) is the strike price. The option price \( C \) is calculated using:

\[
C = e^{-rT} \mathbb{E}[\text{Payoff}],
\]

where \( \mathbb{E} \) denotes the expected value.

## Implementation Details

### Code Highlights
- **Function to generate correlated Brownian motion:**
  ```python
  def generate_brownian_motion(n_steps, n_paths, correlation):
      clamped_correlation = max(-1, min(1, correlation))
      z1 = np.random.normal(0, 1, (n_steps, n_paths))
      z2 = clamped_correlation * z1 + np.sqrt(1 - clamped_correlation**2) * np.random.normal(0, 1, (n_steps, n_paths))
      return z1, z2
