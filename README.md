# Evaluate and Detect Attacks in Financial Market with Simulation
This repository contains codes for course project authored by Chenhao Huang, Shlok Khandelwal and Zizhen Lian for 
CMSC498P/798P.

In this project, we study the problem of evaluating and detecting adversarial attacks in financial market by 
using a simulation environment ([ABIDES](https://github.com/abides-sim/abides)).

## Framework/Pipeline
### Files/Folders   
- `config`: folder contains config scripts to use with ABIDES for running experiments with attack agents and 
detection methods.
    + `rmsc03_analysis.py`: 
- `ABIDES_to_Lobster_data.py`: 
- `agents\VulnerableMomentumAgent.py`: Contains trading logic based of Abides momentum agent
- `agents\AttackMomentumAgent.py`: Contains spoofing logic to attack VulnerableMomentumAgent

### Setups
Before running the configs, make sure submodule `abides` is checked out and the environment variable `PYTHONPATH` 
contains the root of the repo `.` and `./abides`.

### Commands

Running the experiments
```bash
# usage
python config/rmsc03_analysis.py -c rmsc03 -t <ticker symbol> -d <historical date> -s <seed> -l <log dir>

# example
python config/rmsc03_analysis.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour_final
```

Proof of deterministic results
```bash
python abides/util/plotting/chart_fundamental.py -f "log/rmsc03_two_hour_new/impact/fundamental_ABM.bz2" -l impact1 \
-f "log/rmsc03_two_hour_final/impact/fundamental_ABM.bz2" -l impact2
```

## Agents

### Vulnerable Momentum Agent

ABIDES already used a momentum agent, however, this agent had a few problems that needed to be addressed. 

#### How it works

The momentum agent that ABIDES implemented calculates the moving average looking back at a window of 20 and 50 days. It then compares the mean price and if the mean price of 20 is greater than the mean price of 50 it will place a buy limit order. If it is the other way around then it will place a sell or sell limit order. Our goal was to incorporate volume into this same trading rule that ABIDES followed, so we calculated the percentage of buy orders to sell orders based on volume. This was done by enabling the Publisher Subscriber communication system within the agent and changing the message body to request up to 10 levels. With this new buy/sell pressure metric we would additionally only trade buy orders if the buy pressure > sell pressure and ask order vice versa while following the same moving average rules from above. 
    
### Attack Agent

The attack agent's goal is to manipulate the market in such a way that it reacts differently, by placing orders on the edge of the orderbook at key moments in the market. 

#### How it works

The attack agent follows a similar design pattern as the agent above, however, the key difference is determining when to place attack orders. Using the same buy/sell pressure metric as above we were able to determine when the buy volume and sell volume were very close to 50%. Doing so enabled us to flood the market with fake orders such that the vulnerable momentum agents would be tricked into thinking the buy pressure is actually a sell pressure or vice versa. This way the attack agents were able to construct orders at critical moments to confuse the new momentum agents.


## Detection method
###Price Compare

###Trading Compare(Bid/Ask)

In order to earn greater profits, traders all want to buy at the Bid price and then sell at the Ask price. As long as we cross-compare the absolute value of the Ask-Bid price of each transaction, we can find the fluctuation of the transaction. So as to discover which transactions are not making money

