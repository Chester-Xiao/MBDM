***Read the document `A_MBDM_results.ipynb` for detailed information.***

We are newly-formed Group 27, with members from former Group 17 (Christiaan Huisman, Dirk Jacobs) and Group 18 (Chester Xiao, Claudia Fan). 

Our structure is a improved ***Multi-objective robust decision making (MORDM)***. 

1. Exploratory Analysis of Policy Scenarios (***Open Exploration***)
   To generate a diverse set of policy scenarios and understand the broad landscape of possible outcomes.
    - Input: Initial set of scenarios and policies.
    - Process: Simulation of different combinations of policies including "Do Nothing", only dike heightening, only RfR, and mixed strategies.
    - Output: Initial insights into the effectiveness of different policies.
2. Optimization of Policy Strategies (***Multi-Objective Evolutionary Algorithm, MOEA***)
   To identify optimal policy strategies that balance multiple objectives.
    - Input: Results from Open Exploration.
    - Process: Optimization using MOEA to find Pareto-optimal solutions balancing multiple objectives like minimising expected damage, expected number of deaths, and costs.
    - Output: A set of candidate solutions representing the best trade-offs between objectives.
3. Scenario Discovery (***PRIM***) and Robustness Assessment (***Maximum Regret***)
   PRIM
    - Input: Candidate solutions from MOEA.
    - Process: Scenario discovery using PRIM to identify and characterise scenarios (conditions/uncertainties) that lead to favourable or unfavourable.
    - Output: Refined set of candidate solutions with identified parameter ranges and vulnerabilities.
   Maximum Regret:
    - Input: Candidate solutions from MOEA.
    - Process: Evaluation of policy robustness by examining the worst-case erformance relative to the best possible performance in each scenario.
    - Output: Calculate maximum regret for each outcome and policy.
4. Integration and Final Selection of Robust Policies
   - Combine results from exploratory analysis, optimization, scenario discovery, and robustness assessment.
   - Choose policies that show the best trade-offs and robustness under a wide range of scenarios.
