'''
A utility class for tracking negotiations and related concepts.

Knows nothing about scoring, rating, ranking, or testing. Knows nothing about other negotiations.

The goal is to provide back to the experiment a dataframe of all proposals
and the utilities thereof for all parties. These proposals should be prepared
for analysis by providing convenient views, such as a dummy view including
issue interactions. Such a view is also used for calculting utilities.

A party without utility based on interactions will simply have 0 as its
coefficients for those interactions (by default). A utilitymodel can be
fully replaced by a custom one, as long as it takes a proposal as its input
and outputs a utility value.
'''

from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, List

import numpy as np
import pandas as pd
from itertools import combinations
from functools import reduce, cache
from gekko import GEKKO
from hyperopt import fmin, tpe, hp
from hyperopt import STATUS_OK, STATUS_FAIL



def generate_dummy_names(variable_name:str, level_names:List, omit_first:bool=False):
    """Generates dummy variable names given a variable and its levels"""
    if len(level_names) == 1:
        if omit_first:
            return
        yield variable_name
    else:
        for level_name in level_names[omit_first:]:
            yield f"{variable_name}.{level_name}"

def generate_interaction_dummy_names(variable_name1:str, level_names1:List, variable_name2:str, level_names2:List, omit_first:bool=False):
    """Generates all combinations of variable dummies for two variables and their levels"""
    omitted_first = False
    for var1_str in generate_dummy_names(variable_name1, level_names1):
        for var2_str in generate_dummy_names(variable_name2, level_names2):
            if omit_first and not omitted_first:
                omitted_first = True
                continue
            yield f"{var1_str}*{var2_str}"

def generate_all_issue_interactions(issues:Dict):
    """Generates all possible interactions of issues (omitting redundant interactions)"""
    for i1, i2 in combinations(issues.values(), 2):
        if (i1.issue_type == "scale") and (i2.issue_type == "scale"):
            yield f"{i1.name}*{i2.name}"
        else:
            if i1.issue_type == "scale":
                i1_vals = [i1.name]
            else:
                i1_vals = i1.values
            if i2.issue_type == "scale":
                i2_vals = [i2.name]
            else:
                i2_vals = i2.values
            for dummy_name in generate_interaction_dummy_names(i1.name, i1_vals, i2.name, i2_vals, omit_first=True):
                yield dummy_name



def generate_from_pseudofactorial_design(issues:Dict, n:int):
    """Generates issue combinations where each issue's total range is covered while combinations are randomized"""
    randvals = dict()
    for issue in issues.values():
        if issue.issue_type == "scale":
            randvals[issue.name] = list(np.linspace(issue.min_val, issue.max_val, n))
        else:
            nr_vals = len(issue.values)
            vals = []
            for i in range(n):
                ix = i % nr_vals
                vals.append(issue.values[ix])
            randvals[issue.name] = vals
        np.random.shuffle(randvals[issue.name])
    for combos in zip(*randvals.values()):
        yield combos



@dataclass
class Negotiation:
    """Class for structuring and evaluating a negotiation"""
    issues: Dict = field(default_factory=dict)
    proposals: Dict = field(default_factory=dict)
    parties: Dict = field(default_factory=dict)

    def __post_init__(self):
        for proposal in self.proposals.values():
            proposal.evaluate_utilities(self.parties)

    def get_issue_df(self) -> pd.DataFrame:
        """Returns the negotiation issues as a pandas DataFrame"""
        df = pd.DataFrame(
            {
                "issue": self.issues.keys(),
                "issue_type": [x.issue_type for x in self.issues.values()],
                "issue_definition": [f"{x.min_val}, {x.max_val}" if x.issue_type == "scale" else str(x.values) for x in self.issues.values()]
            }
        )
        return df.set_index("issue")

    def get_proposal_df(self, dummies:bool=True, interactions:bool=True, include_utilities:bool=True) -> pd.DataFrame:
        """Returns the proposals and any utilities as a pandas DataFrame"""
        if interactions and not dummies:
            raise ValueError("Cannot return interactions without using dummies!")
        data = dict()

        data["proposal"] = []
        for proposal_name, proposal in self.proposals.items():
            data["proposal"].append(proposal_name)

            if not dummies:                    
                for issue_name, issue_value in proposal.issue_levels.items():
                    if not issue_name in data:
                        data[issue_name] = []
                    data[issue_name].append(issue_value)

            else:
                proposal_dummy_evals = proposal.get_dummy_evaluations(interaction_effects=interactions)
                for issue_name, issue_value in proposal_dummy_evals.items():
                    if not issue_name in data:
                        data[issue_name] = []
                    data[issue_name].append(issue_value)

        if include_utilities:
            for party_name in self.parties.keys():
                data[f"{party_name}_utility"] = []
            for proposal_name, proposal in self.proposals.items():
                proposal.evaluate_utilities(self.parties)
                for party_name, party_utility in proposal.utilities.items():
                    data[f"{party_name}_utility"].append(party_utility)
                    
        df = pd.DataFrame(data)
        return df.set_index("proposal")

    def optimize_utility(self, party_name:str = None, interaction_effects:bool=True, minimize:bool=False) -> float:
        """Returns optimized utility as a float. If party_name is not specified, returns total utility. Maximizes utility by default."""
        print(f"Optimizing utility. Party=<{party_name}>. Interaction effects=<{interaction_effects}>. Minimize=<{minimize}>")

        issue_keys = list(self.issues.keys())

        # Define the search space for the multi-start parameters
        space = dict()
        for issue_key in issue_keys:
                issue = self.issues[issue_key]
                if issue.issue_type == "scale":
                    space[f"{issue_key}"] = hp.quniform(f"{issue_key}", issue.min_val, issue.max_val, 0.1)
                else:
                    for dummy in generate_dummy_names(issue_key, level_names=self.issues[issue_key].values, omit_first=False):
                        space[f"{dummy}"] = hp.quniform(f"{dummy}", 0, 1, 1)


        def objective(params):
            m = GEKKO(remote=False)
            
            issue_variables = dict()
            for issue_key in issue_keys:
                issue = self.issues[issue_key]
                if issue.issue_type == "scale":
                    v = m.Var(lb=issue.min_val, ub=issue.max_val, integer=False, name=f"{issue_key}_variable")
                    v.value = params[f"{issue_key}"]
                    issue_variables[issue_key] = v
                else:
                    dummy_vars_temp_list = []
                    for dummy in generate_dummy_names(issue_key, level_names=self.issues[issue_key].values, omit_first=False):
                        v = m.Var(lb=0, ub=1, integer=True, name=f"{dummy}_variable")
                        v.value = params[f"{dummy}"]
                        issue_variables[dummy] = v
                        dummy_vars_temp_list.append(v)
                    var_sum = reduce(lambda x,y: x+y, dummy_vars_temp_list)
                    m.Equation(var_sum==1)

            #Objective
            objective_components = []
            for dummy_issue_name, dummy_issue in issue_variables.items():
                if party_name is not None:
                    if dummy_issue_name in self.parties[party_name].issue_weights:
                        weight = self.parties[party_name].issue_weights[dummy_issue_name]
                    else:
                        weight = 0
                else:
                    if dummy_issue_name in self.parties[list(self.parties.keys())[0]].issue_weights:
                        weight = np.sum([self.parties[party_name].issue_weights[dummy_issue_name] for party_name in self.parties.keys()])
                    else:
                        weight = 0
                term = dummy_issue * weight
                objective_components.append(term)
            if interaction_effects:
                for combined_name in generate_all_issue_interactions(self.issues):
                    dname0, dname1 = combined_name.split("*")
                    diss0 = issue_variables[dname0]
                    diss1 = issue_variables[dname1]

                    if party_name is not None:
                        if combined_name in self.parties[party_name].issue_weights:
                            weight = self.parties[party_name].issue_weights[combined_name]
                        else:
                            weight = 0
                    else:
                        if combined_name in self.parties[list(self.parties.keys())[0]].issue_weights:
                            weight = np.sum([self.parties[party_name].issue_weights[combined_name] for party_name in self.parties.keys()])
                        else:
                            weight = 0
                    term = diss0 * diss1 * weight
                    objective_components.append(term)

            u = m.Intermediate(reduce(lambda x,y: x+y, objective_components))
            if minimize:
                m.Minimize(u)
            else:
                m.Maximize(u)

            #m.options.IMODE = 3 #Real time optimization of steady state problem, which is what we want
            m.options.SOLVER = 1 #APOPT is an MINLP solver, the only one handling mixed integer problems. try 0 to test them all!

            m.solve(disp=False, debug=False)
            
            results = dict()

            obj = m.options.objfcnval
            if m.options.APPSTATUS==1:
                s=STATUS_OK
            else:
                s=STATUS_FAIL
            results["loss"] = obj
            results["status"] = s
            results["u"] = u.value[0]
            
            for issue_name, issue_var in issue_variables.items():
                if "." in issue_name:
                    results[issue_name] = int(issue_var.value.value[0])
                else:
                    results[issue_name] = issue_var.value.value[0]

            m.cleanup()
            return results
        

        best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
        sol = objective(best)

        ret = dict()
        ret["objective"] = sol['u']
        for var in sol:
            if var not in ("u","status","loss"):
                ret[var] = sol[var]
        
        return ret

    def get_utility_range(self, party_name:str = None, interaction_effects:bool=True) -> Tuple[float,float]:
        """Returns minimum and maximum utility as a tuple. If party_name is not specified, returns total utility min and max."""
        if not hasattr(self, "_utility_range"):
            self._utility_range = dict()
        p = (party_name, interaction_effects)
        if p not in self._utility_range:
            min_util = self.optimize_utility(party_name=party_name, interaction_effects=interaction_effects, minimize=True)
            max_util = self.optimize_utility(party_name=party_name, interaction_effects=interaction_effects, minimize=False)
            self._utility_range[p] = (min_util["objective"], max_util["objective"])
        return self._utility_range[p]
        
    def get_batna_goodness(self, party_name:str, interaction_effects:bool=True) -> float:
        """Returns the percent of utility that the BATNA of the party captures. Negative values represent batnas worse than any negotiated deal."""
        utility_range = self.get_utility_range(party_name=party_name, interaction_effects=interaction_effects)
        range_length = utility_range[1] - utility_range[0]
        u = self.parties[party_name].batna
        u_diff = u - utility_range[0]
        return u_diff / range_length
    
    @staticmethod
    def new_random_negotiation(nr_issues:int=3, nr_proposals:int=20, interaction_effects:bool=True) -> 'Negotiation':
        """Class instatiation method providing a random negotiation set up based on parameters"""
        random_negotiation = Negotiation()

        for i in range(nr_issues):
            if i == 0: #First issue must be scale!
                random_issue = Issue.new_random_issue(issue_name=f"Issue{i}", issue_type="scale")
            else:
                random_issue = Issue.new_random_issue(issue_name=f"Issue{i}")
            random_negotiation.issues[random_issue.name] = random_issue

        for j, issue_values in enumerate(generate_from_pseudofactorial_design(random_negotiation.issues, n=nr_proposals)):
            issue_levels = dict(zip(random_negotiation.issues.keys(), issue_values))
            random_proposal = Proposal(name=f"Proposal{j}", issues=random_negotiation.issues, issue_levels=issue_levels)
            random_negotiation.proposals[random_proposal.name] = random_proposal
        #Old below! Was used before we wanted a factorial design
        #for j in range(nr_proposals):
        #    random_proposal = Proposal.new_random_proposal(proposal_name=f"Proposal{j}", negotiation=random_negotiation)
        #    random_negotiation.proposals[random_proposal.name] = random_proposal

        p1 = Party(name="Party1")
        p1.randomize_weights(random_negotiation.issues, interaction_effects=interaction_effects)
        random_negotiation.parties[p1.name] = p1
        p2 = Party(name="Party2")
        p2.randomize_weights(random_negotiation.issues, interaction_effects=interaction_effects)
        random_negotiation.parties[p2.name] = p2
        p1.randomize_batna_within_utility_range(random_negotiation, interaction_effects=interaction_effects) #This is done AFTER adding parties and weights because we need them for optimization
        p2.randomize_batna_within_utility_range(random_negotiation, interaction_effects=interaction_effects)
        return random_negotiation


@dataclass
class Issue:
    """Class for tracking a negotiation issue and its defining properties"""
    name: str
    issue_type: str
    definition: Union[ Tuple[float,float], Tuple[Union[str, int, float]] ]

    def __post_init__(self):
        if self.issue_type == "scale":
            self.min_val, self.max_val = min(self.definition), max(self.definition)
        else:
            self.values = self.definition
            self.N_values = len(self.definition)
    
    def digitize_value(self, value:Union[str, int, float]) -> Union[float, int]:
        """Evaluates the issue at value and returns a corresponding numerical value.
        
        Scalars are returned as float evaluations of the value.
        Nominals are returned as zero-indexed integers.
        """
        if self.issue_type == "scale":
            return float(value)
        if value not in self.values:
            raise ValueError(f"{value} not in {self.values}")
        i = self.values.index(value)
        return i
    
    def dummy_evaluate_value(self, value:Union[str, int, float], omit_first:bool=False) -> Dict:
        """Evaluates the provided value against dummies for this issue"""
        return_dict = dict()
        if not hasattr(self, "values"):
            return_dict[self.name] = value
        else:
            for val, lvl_name in zip(self.values[omit_first:], generate_dummy_names(self.name, self.values, omit_first=omit_first)):
                return_dict[lvl_name] = (val == value)
        return return_dict
    
    def get_random_value(self) -> Union[str, int, float]:
        """Generates a uniformly distributed random value for this issue"""
        if self.issue_type == "nominal":
            return np.random.choice(self.values)
        return np.random.uniform(self.min_val, self.max_val)
    
    @staticmethod
    def new_random_issue(issue_name:str, issue_type:str=None) -> 'Issue':
        """Creates a new randomized issue instance"""
        if issue_type is None:
            t = np.random.choice(["scale", "nominal"], p=[0.7, 0.3])
        else:
            t = issue_type
        if t == "scale":
            minval = np.random.uniform(-1, 0)
            #maxval = np.random.uniform(0, 1)
            maxval = minval + 1 #We make width equal to that of one dummy variable.
            return Issue(name=issue_name, issue_type=t, definition=(minval,maxval))
        nr_levels = np.random.randint(2,5)
        levels = tuple([f"lvl{i}" for i in range(nr_levels)])
        return Issue(name=issue_name, issue_type=t, definition=levels)


@dataclass
class Proposal:
    """Class for maintaining a set of values over a set of issues. And of utilities."""
    name: str
    issues: Dict = field(default_factory=dict)
    issue_levels: Dict = field(default_factory=dict)
    utilities: Dict = field(default_factory=dict)

    def __post_init__(self):
        pass

    def evaluate_utilities(self, parties:Dict) -> None:
        """Evaluates utilities for all parties provided and saves them in this object's utilities"""
        for party in parties.values():
            self.utilities[party.name] = party.evaluate_utility(self)
    
    def get_dummy_evaluations(self, interaction_effects:bool=True) -> Dict:
        """Evaluates all issues on their dummy variable form"""
        dummy_evaluation_dict = dict()
        for issue_name, issue in self.issues.items():
            if issue.issue_type == "scale":
                dummy_evaluation_dict[issue_name] = issue.digitize_value(self.issue_levels[issue_name])
            else:
                for lvl_name, truth_value in issue.dummy_evaluate_value(self.issue_levels[issue_name], omit_first=True).items():
                    dummy_evaluation_dict[lvl_name] = truth_value
        if interaction_effects:
            temp_dummy_evaluation_dict = dict()
            for issue_name, issue in self.issues.items():
                if issue.issue_type == "scale":
                    temp_dummy_evaluation_dict[issue_name] = issue.digitize_value(self.issue_levels[issue_name])
                else:
                    for lvl_name, truth_value in issue.dummy_evaluate_value(self.issue_levels[issue_name], omit_first=False).items():
                        temp_dummy_evaluation_dict[lvl_name] = truth_value

            for interaction_name in generate_all_issue_interactions(self.issues):
                v1, v2 = interaction_name.split("*")
                dummy_evaluation_dict[interaction_name] = temp_dummy_evaluation_dict[v1]*temp_dummy_evaluation_dict[v2]
        return dummy_evaluation_dict

    def get_optimality(self, negotiation:Negotiation, party_name:str = None, interaction_effects:bool=True) -> float:
        """Evaluates the optimality as a percent of possible utility capture. No party_name returns total utility."""
        utility_range = negotiation.get_utility_range(party_name=party_name, interaction_effects=interaction_effects)
        range_length = utility_range[1] - utility_range[0]
        self.evaluate_utilities(negotiation.parties)
        if party_name is not None:
            u = self.utilities[party_name]
        else:
            u = np.sum(list(self.utilities.values()))
        u_diff = u - utility_range[0]
        return u_diff / range_length
    
    @staticmethod
    def new_from_dummy_issues(proposal_name: str, negotiation: Negotiation, dummy_issues: Dict) -> "Proposal":
        """Creates a new proposal instance based on dummy issues and levels."""
        per_issue_levels = list((k, v) for k, v in dummy_issues.items() if "*" not in k)
        original_levels = dict()
        for name, val in per_issue_levels:
            if "." not in name:
                original_levels[name] = val
            else:
                base_name = name.split(".")[0]
                if base_name not in original_levels:
                    original_levels[base_name] = negotiation.issues[base_name].values[0]
                if val == True:
                    original_levels[base_name] = name.split(".")[1]
        new_proposal = Proposal(name=proposal_name, issues=negotiation.issues, issue_levels=original_levels)
        return new_proposal

    @staticmethod
    def new_random_proposal(proposal_name:str, negotiation: Negotiation) -> "Proposal":
        """Creates a new randomized proposal instance based on issues in negotiation."""
        issue_names_and_values = dict(zip(negotiation.issues.keys(), (x.get_random_value() for x in negotiation.issues.values())))
        new_proposal = Proposal(name=proposal_name, issues=negotiation.issues, issue_levels=issue_names_and_values)
        return new_proposal


@dataclass
class Party:
    name: str
    issue_weights: Dict = field(default_factory=dict)
    batna:float = 0

    def randomize_weights(self, issues:Dict, interaction_effects:bool=True) -> None:
        """Randomizes the weights of the utility function associated with this Party"""
        for issue_name, issue in issues.items():
            if issue.issue_type == "scale":
                self.issue_weights[issue_name] = np.random.uniform(low=-1, high=1)
            else:
                for dummy_name in generate_dummy_names(issue_name, issue.values, omit_first=True):
                    self.issue_weights[dummy_name] = np.random.uniform(low=-1, high=1)
        if interaction_effects:
            for interaction_name in generate_all_issue_interactions(issues):
                self.issue_weights[interaction_name] = np.random.uniform(low=-1, high=1)

    def evaluate_utility(self, proposal:Proposal, interaction_effects:bool=True) -> float:
        """Evaluates the utility of a proposal based on the issue weights of this Party"""
        utility = 0
        proposal_dummy_evals = proposal.get_dummy_evaluations(interaction_effects=interaction_effects)
        for issue_name, weight in self.issue_weights.items():
            utility_contribution = weight*proposal_dummy_evals[issue_name]
            utility += utility_contribution
        return utility

    def randomize_batna_within_utility_range(self, negotiation:Negotiation, interaction_effects:bool=True) -> None:
        """Randomizes the batna to some value within the utility range +/- 10 %"""
        min_util, max_util = negotiation.get_utility_range(party_name=self.name, interaction_effects=interaction_effects)
        util_range = max_util - min_util
        self.batna = np.random.uniform(min_util - util_range*0.1, max_util + util_range*0.1)

        


def gekko_test():
    issue0 = Issue(name="Issue0", issue_type="scale", definition=(-0.9593553291234116, 0.04064467087658841))
    issue1 = Issue(name="Issue1", issue_type="scale", definition=(-0.2296904226173454, 0.7703095773826546))
    party1 = Party(name='Party1', issue_weights={'Issue0': 0.9298295514426353, 'Issue1': 0.13201307383634653, 'Issue0*Issue1': -0.3050387870868745}, batna=-0.01303174707802457)
    party2 = Party(name='Party2', issue_weights={'Issue0': -0.6429371485361639, 'Issue1': 0.11629976308237633, 'Issue0*Issue1': -0.7848308022753774}, batna=0.031574528829476145)
    neg = Negotiation(
        issues = {
            "Issue0": issue0,
            "Issue1": issue1
        },
        parties = {
            "Party1": party1,
            "Party2": party2
        }
    )
    print(neg.optimize_utility())


    optimal_proposal = Proposal(
        "optim",
        issues = {
            "Issue0": issue0,
            "Issue1": issue1
        },
        issue_levels = {
            'Issue0': -0.9593553291234116,
            'Issue1': 0.77030957738
        }
    )
    optimal_proposal.evaluate_utilities(neg.parties)
    print(sum(optimal_proposal.utilities.values()))


    m = GEKKO(remote=False)
    # Define objective
    def obj(x1, x2):
        return x1*0.11566 + x2*0.74956 + x1*x2*(-0.53452)

    # Define variables
    x1 = m.Var(lb=-0.95936, ub=0.04064)
    x2 = m.Var(lb=-0.22969, ub=0.77031)

    u = m.Intermediate(obj(x1,x2))
    m.Maximize(u)

    # Solve the problem
    m.options.SOLVER = 'APOPT'
    m.solve(disp=False)

    # Print the results
    print(f'x1: {x1.value[0]}')
    print(f'x2: {x2.value[0]}')
    print(f'u: {u.value[0]}')




if __name__ == "__main__":
    n = Negotiation.new_random_negotiation(
        nr_issues=5,
        nr_proposals=10,
        interaction_effects=True
    )
    print(n.get_issue_df())
    print(n.get_proposal_df())