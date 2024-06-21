import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import multiprocessing
import pickle

# Import EMA Workbench for exploratory modelling and analysis
from ema_workbench import (Policy, ema_logging, MultiprocessingEvaluator, save_results, Scenario, Constraint,
                            ScalarOutcome, Constant, perform_experiments)

from dike_model_function import DikeNetwork
from problem_formulation import get_model_for_problem_formulation, sum_over, sum_over_time

from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from ema_workbench.em_framework.samplers import sample_uncertainties

from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress, ArchiveLogger, to_problem)
from ema_workbench.em_framework.evaluators import (BaseEvaluator, SequentialEvaluator)
from ema_workbench.analysis import parcoords    # Create parallel coordinate plots

from ema_workbench.util.utilities import (save_results, load_results)
from ema_workbench.analysis import prim
 

print(pd.__version__)       # make sure pandas is version 1.0 or higher
print(nx.__version__)       # make sure networkx is version 2.4 or higher



# Initialisation
ema_logging.log_to_stderr(ema_logging.INFO)                         # log information to the console
dike_model, planning_steps = get_model_for_problem_formulation(3)   # choose PF3 for open exploration   

def get_do_nothing_dict():                                          # baseline policy ("do nothing")
    return {l.name: 0 for l in dike_model.levers}


# Policy definitions
policies = 100



# Run the open exploration
n_scenarios = 200
with MultiprocessingEvaluator(dike_model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios, policies)

experiments, outcomes = results     # unpack the results into "scenarios & policies" and "model outputs"
policies = experiments['policy']    # extract the policy information

data = pd.DataFrame.from_dict(outcomes)
data['policy'] = policies


# save results for later
save_results(results, "s_d.tar.gz")

# Visualisation
plot = sns.pairplot(data, hue='policy', vars=outcomes.keys())
plot.savefig("open_exploration_pf3.png")
plt.show()



# Initialisation
ema_logging.log_to_stderr(ema_logging.INFO)         # Log information to the console
experiments, results = load_results('s_d.tar.gz')   # load results from open exploration

dike_model, planning_steps = get_model_for_problem_formulation(3)   # choose PF3 for open exploration

# Define outcomes with expected ranges //CAN BE CHANGED
dike_model.outcomes = [
    ScalarOutcome('Expected Annual Damage', kind=ScalarOutcome.MINIMIZE, expected_range=(0, 1e6)),
    ScalarOutcome('Total Costs', kind=ScalarOutcome.MINIMIZE, expected_range=(0, 1e9)),
    ScalarOutcome('Reliability', kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1))
]


# Set convergence metrics 
convergence_metrics = [
                        EpsilonProgress(),  # track optimisation progress in terms of how much the solutions
                        ]                   # have improved over time, based on epsilon-dominance


# Define constraints  //CAN BE CHANGED
constraints = [Constraint("max_cost", outcome_names="Expected Evacuation Costs",
                          function=lambda x: max(0, x-1e6))]


# Run MOEA optimisation
workers = min(multiprocessing.cpu_count(),4)
with MultiprocessingEvaluator(dike_model) as evaluator:
# with MultiprocessingEvaluator(dike_model) as evaluator:
    results, convergence = evaluator.optimize(nfe=10000,                        # number of function evaluations, set to 10000
                                              searchover='levers',              # optimise 'levels' (can be 'uncertainties')
                                              epsilons=[0.25]*len(dike_model.outcomes),        # grid resolution for epsilon-dominance
                                              convergence=convergence_metrics,  # list of convergence metrics to track
                                              constraints=constraints          # list of constraints to enforce during optimisation
                                              )          

# Visualise convergence, to assess whether the algorithm has converged (how the solution have been improved over time)
fig, ax = plt.subplots()
ax.plot(convergence.nfe, convergence.epsilon_progress)
ax.set_ylabel('$\epsilon$-progress')
ax.set_xlabel('number of function evaluations')
plt.savefig('convergence_plot.png')
plt.show()


# Visualise results
outcomes = results.loc[:, ['Expected Evacuation Costs', 'RfR Total Costs', 'A.1 Total Costs']]
limits = parcoords.get_limits(outcomes)
axes = parcoords.ParallelAxes(limits)
axes.plot(outcomes)
axes.invert_axis('Expected Evacuation Costs')
plt.show()


# Save the optimization results
with open('optimization_results.pkl', 'wb') as file:
    pickle.dump((results, convergence), file)

print("Optimization results saved to 'optimization_results.pkl'")