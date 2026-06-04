# RL Agents for Delta Hedging a USD/PLN European Call Option

Can a reinforcement learning agent learn to hedge a currency option better than Black Scholes? This project builds and compares 15 agents across 5 algorithmic families to find out. It uses real USD/PLN spot data, realistic transaction costs, and a Hidden Markov Model to simulate volatility that switches between regimes.

---

## The Problem

A sell side trader sells a 40 day USD/PLN European call option to a Polish client hedging dollar exposure. The standard approach is Black Scholes delta hedging. It assumes constant volatility and continuous, frictionless rebalancing. In practice neither holds.

USD/PLN volatility clusters and spikes around events like COVID, the Russian invasion of Ukraine, and Liberation Day. Transaction costs make continuous rebalancing prohibitively expensive. A reinforcement learning agent can, in principle, observe both of these and learn a smarter rebalancing policy than a fixed rule.

The core question this project asks is: given access to transaction cost information and changing volatility signals, can an RL agent beat the Black Scholes benchmark?

---

## Data and Volatility Modelling

Daily USD/PLN spot prices are pulled from Yahoo Finance for the period January 2019 to February 2026. This window was chosen deliberately. It contains calm pre COVID periods and genuinely stressed periods, giving the Hidden Markov Model enough contrast to separate the two volatility regimes.

Log returns are used throughout, consistent with the Ito derivation of spot dynamics:

$$d\ln S(t) = \left(r_{PLN} - r_{USD} - \frac{1}{2}\sigma^2\right)dt + \sigma\, dB^*(t)$$

This means the log return over one day is normally distributed. The Black Scholes model assumes the standard deviation of these returns is constant over time. The data shows this is false.

**Figure 1 in the notebook** shows USD/PLN spot and daily log returns from 2019 to 2026. Volatility clearly clusters around stress events.

**Figure 2 in the notebook** overlays the empirical distribution of log returns against the fitted normal distribution. The empirical distribution has fatter tails and a higher peak near zero. This is the signature of a distribution that mixes two regimes: many quiet days and occasional large moves.

**Figure 3 in the notebook** shows rolling volatility computed over 3 day, 21 day, and 252 day windows. If Black Scholes were correct, all three would be flat. They are not.

**Figure 4 in the notebook** shows the autocorrelation of absolute log returns up to 60 day lags. 68 percent of lags are statistically significant at the 5 percent level. Volatility is not only time varying. It is predictable from its own history. This motivates the Hidden Markov Model.

### Hidden Markov Model Calibration

A 2 state HMM is specified by the tuple $\lambda = \{N, M, A, B, \pi\}$ where $N = 2$ is the number of hidden states (calm and stressed), $A = \{a_{ij}\}$ is the state transition matrix, $B = \{b_j(o)\}$ are the emission densities, and $\pi$ is the vector of initial state probabilities.

Because log returns are continuous, each state emits from a Gaussian:

$$b_j(o) = \mathcal{N}(o \mid \mu_j, \sigma_j^2)$$

Calibrating $\lambda$ from data uses the Baum Welch algorithm, an expectation maximisation procedure. The E step computes expected state occupancies $\gamma_t(i)$ and expected transitions $\xi_t(i,j)$ using the forward backward algorithm. The M step re estimates parameters in closed form:

$$\bar{a}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \qquad \bar{\mu}_j = \frac{\sum_{t=1}^{T} \gamma_t(j)\, o_t}{\sum_{t=1}^{T} \gamma_t(j)}, \qquad \bar{\sigma}_j^2 = \frac{\sum_{t=1}^{T} \gamma_t(j)(o_t - \bar{\mu}_j)^2}{\sum_{t=1}^{T} \gamma_t(j)}$$

**Figure 5 in the notebook** shows Baum Welch convergence. The log likelihood increases monotonically and plateaus around iteration 30.

**Figure 6 in the notebook** overlays HMM regime classifications on the spot series. COVID, the Russian invasion of Ukraine, and Trump's elections are correctly flagged as stressed periods.

**Figure 7 in the notebook** shows the two emission distributions alongside the empirical histogram. The mixture distribution closely matches the data, validating the HMM fit.

### Calibrated Parameters

| Statistic | Calm | Stressed |
|---|---|---|
| Daily mean | -0.000123 | 0.000211 |
| Daily std | 0.004698 | 0.009660 |
| Annualised volatility | 7.46% | 15.33% |
| P(stay in regime) | 0.967 | 0.918 |
| P(switch) | 0.033 | 0.082 |
| Average duration (days) | 30.5 | 12.2 |
| Time spent in regime | 74.3% | 25.7% |

| Transition | From Calm | From Stressed |
|---|---|---|
| To Calm | 0.967 | 0.082 |
| To Stressed | 0.033 | 0.918 |

---

## Environment Design

The environment is a finite horizon Markov Decision Process. Each episode represents the 40 day life of a USD/PLN European call option. One time step is one trading day, so $\Delta t = 1/252$.

### Spot Price Simulation

At each step, given the current regime $R_t \in \{\text{calm}, \text{stressed}\}$, the spot evolves as:

$$S_{t+1} = S_t \exp\!\left[\left(\mu_{R_t} - \frac{1}{2}\sigma_{R_t}^2\right)\Delta t + \sigma_{R_t}\sqrt{\Delta t}\,\varepsilon_t\right], \qquad \varepsilon_t \sim \mathcal{N}(0,1)$$

The exponentiated form keeps spot prices strictly positive. The regime transitions according to:

$$P(R_{t+1} = j \mid R_t = i) = A_{ij}$$

**Figures 8 and 9 in the notebook** compare 500 simulated paths from the constant volatility GBM model and the HMM regime switching model against actual USD/PLN data. The HMM produces a tighter fan of paths that better matches the real data.

### Book Value

The trader sells a call and collects a premium $P_0 = c_0 \cdot N$ where $N = \$100{,}000$ is the notional. The book value at each step is:

$$V_t = -c_t \cdot N + P_0 + I^{PLN}_t - C_t + U_t \cdot S_t + I^{USD}_t \cdot S_t - TC^{cum}_t$$

The terms are the mark to market short call liability, the premium, PLN interest earned on the premium, PLN spent buying the hedge, the USD hedge converted to PLN, USD interest earned on the hedge, and cumulative transaction costs.

### Transaction Costs

Transaction costs follow Dewynne, Whalley and Wilmott (1994). This was confirmed as realistic by a conversation with an emerging markets FX trader at Bank of America.

$$TC_t = k_1 + k_2 |\Delta U_t| + k_3 |\Delta U_t| \cdot S_t$$

where $k_1 = k_2 = 0$ and $k_3 = 0.0003$. The dominant cost is the proportional bid ask spread. Most RL hedging literature uses $k_3 \approx 1\%$ to exaggerate agent benefits. This project uses 0.03%, which is realistic for institutional USD/PLN spot trading.

### States, Actions and Rewards

The raw state passed to the agent is:

$$s_t = (t, T, S_t, K, \delta_t, \{S_0, \ldots, S_t\}, r_{PLN}, r_{USD}, k_1, k_2, k_3)$$

From this, six features are engineered: moneyness $S_t/K$, time to expiry fraction $(T-t)/T$, urgency $1/(T-t)$, rolling volatility estimate $\hat{\sigma}_t$, volatility surprise $\hat{\sigma}_t - \sigma_{fixed}$, and hedge misalignment $\delta_t - \delta^{BS}_t$. Each is also binned into one hot indicators to allow the linear models to assign distinct values to distinct regions of the state space.

The reward is the per step change in book value:

$$R_t = V_{t+1} - V_t$$

At expiry the agent unwinds its USD position and incurs a final transaction cost.

### The Call Pricing Formula

The strike is set to spot at inception so the option is at the money. Gamma is highest at the money, meaning delta moves most for a given spot move. This is the hardest case for delta hedging. The Garman Kohlhagen call price is:

$$c(S, \tau) = e^{-r_{USD}\tau} S \, N(d_1) - e^{-r_{PLN}\tau} K \, N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r_{PLN} - r_{USD} + \frac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}}, \qquad d_2 = d_1 - \sigma\sqrt{\tau}$$

The Black Scholes delta is:

$$\delta^{BS} = e^{-r_{USD}\tau} N(d_1)$$

---

## Why This Is a Hard RL Problem

The fundamental difficulty is a high signal to noise ratio. The per step reward $R_t = V_{t+1} - V_t$ is driven mostly by the mark to market change of the call, which is in turn driven by random spot moves. The component the agent controls, the change from rebalancing decisions, is small relative to this noise.

This is made worse by the HMM regime structure. Calm episodes and stressed episodes produce very different return distributions. An agent may receive positive reinforcement for a hedge decision in a calm episode and negative reinforcement for the same decision in a stressed episode. The signal the agent is trying to learn from is genuinely small compared to the episode to episode variance.

This is the core reason most agents in this project fail to converge. It is also why the Black Scholes baseline is so hard to beat at realistic transaction cost levels.

---

## Agents and Results

### Prediction

Before moving to control, Monte Carlo and TD(0) value function approximation are run on both Black Scholes policies. The best alpha schedules are identified by sweeping inverse, power, and log decay schedules across 3 seeds. A log schedule with $c = 20$ is consistently best for TD(0). A power schedule with $c = 50$ is best for MC on the fixed vol policy.

**Table: Alpha schedule comparison for Black Scholes Fixed Vol policy**

| Schedule | MC MSE mean | MC MSE std | TD error mean | TD error std |
|---|---|---|---|---|
| inverse c=10 | 3.69e6 | 267,091 | 505.23 | 19.23 |
| inverse c=50 | 2.68e6 | 260,505 | 1274.46 | 176.52 |
| power c=20 | 2.96e6 | 665,363 | 1364.47 | 63.68 |
| power c=50 | 2.54e6 | 132,880 | 1365.02 | 90.42 |
| log c=20 | 3.08e6 | 415,337 | 373.42 | 47.45 |

**Figures 10 and 11 in the notebook** show the alpha schedule decay curves and convergence across seeds.

### Linear Q Learning

The first control agents use epsilon greedy Q learning with a linear state action value function $\hat{Q}(s,a) = \theta^\top \phi(s,a)$. Actions are discretised into 11 bins over $[0, 1]$. Two variants are tested.

MC on policy control updates with the full episode return:

$$\theta \leftarrow \theta + \alpha \cdot (G_t - \theta^\top \phi_t)\, \phi_t$$

SARSA TD(0) updates with a one step bootstrap target:

$$\theta \leftarrow \theta + \alpha \cdot (R_t + \theta^\top \phi_{t+1} - \theta^\top \phi_t)\, \phi_t$$

The MC agent converges to a near no trade policy. It collects the premium and holds a flat position. This produces a mean P&L close to Black Scholes but a standard deviation 3.6 times larger. The TD agent learns to speculate rather than hedge and also has very high variance. Both agents fail to capture the non linear interaction between moneyness, time to expiry, and transaction cost that defines the optimal rebalancing policy.

**Figures 13 and 14 in the notebook** compare alpha and epsilon schedule performance for both agents.

### DQN

The Q network is a two hidden layer MLP with ReLU activations $(6 \to 64 \to 64 \to 21)$. Training uses experience replay with a buffer of 50,000 transitions, a frozen target network synchronised every 200 episodes, and the Adam optimiser at learning rate $10^{-3}$ over 5,000 episodes.

The Bellman loss is:

$$\mathcal{L}(\theta) = \mathbb{E}_{(S_t, A_t, R_{t+1}, S_{t+1}) \sim \mathcal{D}} \left[\left(R_{t+1} + \gamma \max_a Q_{\theta^-}(S_{t+1}, a) - Q_\theta(S_t, A_t)\right)^2\right]$$

Three versions are trained with progressively redesigned reward shaping. DQN v1 uses an asymmetric penalty on negative rewards only. DQN v2 applies a symmetric quadratic penalty on all rewards to discourage both large profits and large losses, reducing variance. DQN v3 restructures the reward as excess return over a parallel Black Scholes shadow book.

DQN v2 with $\lambda = 5$ is the best performing agent overall. It matches Black Scholes mean P&L while tightening standard deviation to 1918.

**Figures 15, 16 and 17 in the notebook** show training curves and P&L distributions for DQN v1, v2, and v3.

### MCPG

The policy network outputs a Gaussian distribution over actions:

$$A_t \sim \pi_\theta(\cdot \mid S_t) = \mathcal{N}(\mu_\theta(S_t),\, \sigma_\theta(S_t)^2), \qquad \delta^*_t = \text{clip}(A_t, 0, 1)$$

The REINFORCE gradient estimator is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\!\left[\sum_{t=0}^{T-1} G_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)\right]$$

All three MCPG variants fail to converge. The root cause is not the reward structure. It is the REINFORCE estimator itself. Each policy update is scaled by a single noisy episode return $G_0$. With HMM regime switches producing fat tailed return distributions, successive updates push $\theta$ in incoherent directions. v3 adds an exponential moving average baseline to reduce gradient variance:

$$b_{e+1} = 0.99\, b_e + 0.01 \sum_{t=0}^{T-1} R_{t+1}$$

This reduces the magnitude of gradients but does not correct their direction. When episode returns flip sign due to regime switches, the baseline cannot compensate. A state dependent baseline, which is what actor critic provides, is needed.

**Figures 18, 19, 20 and 21 in the notebook** show the training instability across MCPG variants.

### Actor Critic

The actor is the MCPG Gaussian policy. The critic $V_\phi(s)$ shares the same architecture but outputs a scalar. The actor critic losses are:

$$\mathcal{L}_{critic}(\phi) = \frac{1}{T}\sum_{t=0}^{T-1} \left(G_t - V_\phi(S_t)\right)^2$$

$$\mathcal{L}_{actor}(\theta) = -\sum_{t=0}^{T-1} \left(G_t - V_\phi(S_t)\right) \log \pi_\theta(A_t \mid S_t)$$

AC v1 collapses because the critic diverges early in training. Raw episode returns are on the scale of PLN thousands while the critic initialises near zero, producing unstable advantage estimates. AC v2 fixes this by standardising returns within each episode before they enter either loss:

$$\tilde{G}_t = \frac{G_t - \bar{G}}{\text{std}(G) + \varepsilon}$$

This constrains advantage estimates to the range $[-2, 2]$ and prevents early critic divergence. AC v2 converges to a genuine hedging policy that matches Black Scholes mean P&L exactly but with 92 percent higher standard deviation.

**Figures 22 and 23 in the notebook** compare AC v1 and AC v2 training curves and P&L distributions.

### DDPG

Both DDPG variants collapse to a zero hedge policy. The deterministic actor is updated by backpropagating through the critic. The critic initialises with random weights so its early Q estimates are noise. This pushes the sigmoid output of the actor to an extreme where the gradient is near zero and no further learning signal can flow. The standard fix in the literature is an uncertainty aware critic, as proposed by Zheng, He and Yang (2023).

**Figures 24, 25 and 26 in the notebook** show DDPG collapse under both reward structures.

---

## Training Stability and the Non Stationary Environment

A recurring finding across agents is instability caused by the non stationary nature of the environment. Each episode draws a fresh 40 day spot path from the HMM. This means the distribution of rewards shifts between episodes as regime composition changes. Agents trained with decaying alpha schedules eventually assign near zero weight to new experience, which prevents adaptation. Agents trained with decaying epsilon schedules eventually stop exploring, which locks in policies learned under one regime mix that may be wrong for another.

A natural fix for both is to hold alpha and epsilon constant throughout training. A constant alpha schedule treats the problem as a tracking problem: it keeps the agent continuously responsive to new experience rather than converging to a fixed policy based on early training data. A constant epsilon similarly maintains ongoing exploration so the agent can update its policy as the regime mix changes. This was not implemented in this project but is a direct next step given the training instability observed across all agent families.

---

## Final Results

**Figure 27 in the notebook** shows the risk return scatter for all 15 agents. DQN v2 clusters tightly around Black Scholes Fixed. DDPG variants are far to the right at standard deviation 9250.

**Figure 28 in the notebook** shows mean turnover per episode for each agent.

| Policy | Mean P&L | Std P&L | Sharpe | 5th pct | 95th pct | Mean TC | Turnover |
|---|---|---|---|---|---|---|---|
| BS Fixed | 1022 | 1623 | 0.6 | -1949 | 3305 | 307 | 2.8 |
| BS Rolling | 923 | 1946 | 0.5 | -2556 | 3711 | 388 | 3.6 |
| Linear Q MC | 862 | 5840 | 0.1 | -11755 | 6156 | 64 | 0.6 |
| Linear Q TD | 982 | 5787 | 0.2 | -8720 | 9206 | 360 | 3.3 |
| DQN v1 | 1189 | 3488 | 0.3 | -4985 | 6276 | 483 | 4.5 |
| DQN v2 (lambda=2) | 1071 | 2178 | 0.5 | -3244 | 3930 | 297 | 2.8 |
| DQN v2 (lambda=5) | 1027 | 1918 | 0.5 | -2558 | 3731 | 303 | 2.8 |
| DQN v3 | 1067 | 2822 | 0.4 | -4011 | 4994 | 343 | 3.2 |
| MCPG v1 | 2224 | 7883 | 0.3 | -14144 | 7881 | 215 | 2.0 |
| MCPG v2 | 1649 | 6082 | 0.3 | -10523 | 9069 | 216 | 2.0 |
| MCPG v3 | 1670 | 6030 | 0.3 | -9311 | 9085 | 241 | 2.2 |
| AC v1 | 661 | 5144 | 0.1 | -8999 | 5925 | 175 | 1.6 |
| AC v2 | 1022 | 3121 | 0.3 | -4882 | 4909 | 245 | 2.3 |
| DDPG v1 | 276 | 9250 | 0.0 | -20191 | 5927 | 0 | 0.0 |
| DDPG v2 | 276 | 9250 | 0.0 | -20191 | 5927 | 0 | 0.0 |

No RL method strictly dominates Black Scholes fixed vol delta hedging. Value based methods with experience replay are the most robust to regime switching. Policy gradient methods fail due to gradient variance that reward shaping cannot fix. DDPG fails due to a structural critic initialisation problem.

---

## Repository Structure

```
notebooks/
    data_analysis.ipynb         HMM calibration and data exploration
    pred_and_control.ipynb      Environment, all 15 agents, training, evaluation
data/
    USDPLN_spot_data.csv
    HMM_calibrated_params.csv
    HMM_transition_matrix.csv
```

Both notebooks contain all cell outputs and can be read statically. To re run, upload to Google Colab with the data folder intact. The main notebook takes approximately 5 hours to re run from scratch on Colab Pro with High RAM.
