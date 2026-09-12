# Differentiable Multi-Agent Financial Market Model

A **multi-agent stock market simulator** where different AI and rule-based traders interact to generate realistic market behavior.

The project combines neural-network-based traders, momentum and value investors, behavioral factors, and a simulated market to study how individual trading decisions affect overall price movements.

---

## Overview

The simulator models a market with **300 independent trading agents**.

Agents differ in how they make decisions:

* **AI Traders** — use LSTM, GRU, CNN, and MLP models to predict returns.
* **Momentum Traders** — follow recent price trends.
* **Value Traders** — trade based on deviations from estimated fair value.
* **Random Traders** — provide background market activity and liquidity.

Agents are also influenced by factors such as:

* Risk tolerance
* Fear and herding
* Trading aggressiveness
* Portfolio pressure
* Influence from other traders

The goal is to create a market where these different behaviors interact and produce realistic price and trading patterns.

---

## How It Works

The project follows three main stages:

```text
Historical Market Data
        ↓
Train AI Return Predictors
        ↓
Calibrate Trader Behavior
        ↓
Run 300-Agent Market Simulation
        ↓
Analyze Prices, Returns & Trading Behavior
```

### 1. Train AI Traders

Historical stock data is used to train LSTM, GRU, CNN, and MLP models to predict future returns.

### 2. Build & Calibrate Agents

Different types of traders are created with different risk preferences and trading behaviors.

Their parameters are calibrated so that the simulated market behaves more like real market data.

### 3. Run the Market

All 300 agents interact over multiple trading days.

Their combined buying and selling activity determines market prices and generates:

* Price movements
* Trading volume
* Portfolio wealth
* Agent performance
* Market behavior statistics

---

## Agent Types

| Agent Type       | Number | Behavior                                     |
| ---------------- | -----: | -------------------------------------------- |
| AI Traders       |     36 | Predict future returns using neural networks |
| Momentum Traders |    100 | Follow price trends                          |
| Value Traders    |    100 | Trade toward estimated fair value            |
| Random Traders   |     64 | Provide stochastic market activity           |

**Total: 300 agents**

---

## Market Simulation

The project supports two ways of simulating market prices:

* **Price Impact Model** — market prices change according to overall buying and selling pressure.
* **Limit Order Book** — agents submit buy/sell orders that are matched to simulate a more realistic trading environment.

---

## Results

The simulator was evaluated using **5 independent runs**.

| Metric                           |   Mean |    Std |
| -------------------------------- | -----: | -----: |
| Log-price RMSE                   | 0.3787 | 0.0183 |
| Mean Absolute Price Error        | 22.24% |  0.76% |
| Return Distribution KS Statistic | 0.2368 | 0.0020 |

The low variation across independent runs indicates that the simulation produces **consistent results despite stochastic agent behavior**.

---

## Agent Performance

A representative 500-day simulation showed different outcomes across trader types:

| Agent Type       | Mean Return |
| ---------------- | ----------: |
| MLP Traders      |     +166.6% |
| CNN Traders      |     +141.0% |
| LSTM Traders     |      +98.0% |
| GRU Traders      |      +82.0% |
| Momentum Traders |      +45.0% |
| Value Traders    |      +21.0% |
| Random Traders   |       -2.0% |

These differences illustrate how different trading strategies behave under the same simulated market conditions.

---

## Repository Structure

```text
Differentiable-multi-agent-financial-market-model/
│
├── README.md
├── requirements.txt
│
├── train.ipynb
├── gan_calibration.ipynb
├── unified_abm_simulation_v2_independent_mv_random_regularized.ipynb
│
├── data/
│   ├── ADANIPORTS.csv
│   └── ASIANPAINT.csv
│
├── checkpoint/
│   ├── AI model checkpoints
│   ├── trained trading agents
│   └── simulation states
│
├── outputs/
│   └── simulation results
│
└── calibrated_personalities.json
```

---

## Installation

### Requirements

* Python 3.10+
* PyTorch
* CUDA GPU recommended but not required

### Setup

```bash
git clone https://github.com/Maheshpolisetti/Differentiable-multi-agent-financial-market-model.git

cd Differentiable-multi-agent-financial-market-model

python -m venv env

# Linux / macOS
source env/bin/activate

# Windows
.\env\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Running the Project

Run the notebooks in the following order:

```text
1. train.ipynb
        ↓
2. gan_calibration.ipynb
        ↓
3. unified_abm_simulation_v2_independent_mv_random_regularized.ipynb
```

### `train.ipynb`

Trains the neural-network return predictors.

### `gan_calibration.ipynb`

Calibrates the behavioral characteristics of the trading agents.

### `unified_abm_simulation...ipynb`

Runs the complete 300-agent market simulation and generates the final results.

---

## Technologies

* Python
* PyTorch
* NumPy
* Pandas
* Jupyter Notebook
* LSTM / GRU
* 1D CNN
* Multi-Agent Simulation
* Limit Order Book
