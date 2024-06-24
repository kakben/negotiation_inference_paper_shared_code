"""
Contains the Experiment class and the Trial class.

The purpose is to create and analyze a negotiation and the rating thereof.
Each party will be asked to score or rate proposals.
Each party will also have a rating error for this.
We will:
1. Create negotiation based on experiment parameters. DONE
2. Move from true utilities to a score or rating. DONE
    This may include inserting a "do nothing" alternative and scrambling the batna
    value of a party to get its score/rank.
3. Run MCMC to estimate parameters. DONE

4. Then use the estimations of multiple
    experiments to say something about uncertainty of these parameters based on
    experiment settings.
5. Draw parameter values from their posterior distributions. Then optimize to find
    the best, or at least a very good, proposal for both parties. Check its
    acceptability (vs batna) for both parties. Also check how much per-party and
    total value it captures as compared to a randomized proposal. DONE

1-3 together are a Trial.
Repeated trials and 4+5 are administered by the Experiment.
Notebook testing will use Trial to make steps 1 and 2. Then we manually specify
and develop step 3. DONE

Finally, use notebook to run experiment. Track progress etc. Save progess gradually.
"""



from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, List
from functools import reduce, cmp_to_key
from itertools import combinations, product
from time import time
import datetime
import numpy as np
import pymc as pm
import json
import pandas as pd

import negotiation



@dataclass
class Trial:
    """A class to contain a negotiation, a scoring thereof,
    and estimation of a PyMC model to infer the parameters."""
    nr_issues: int = 3
    nr_proposals: int = 8
    interaction_effects_in_negotiation: bool = True
    interaction_effects_in_model: bool = True
    include_batna_in_rating: bool = True
    rater_error: float = 0.02
    use_scoring: bool = True #if false, use pairwise comparison
    #Pairwise requires us to decide how many comparisons have been made!
    #I need to create some kind of orthogonal design here maybe?
    nr_test_deals: int = 1000

    def __post_init__(self):
        self.negotiation = negotiation.Negotiation.new_random_negotiation(
            nr_issues=self.nr_issues,
            nr_proposals=self.nr_proposals,
            interaction_effects=self.interaction_effects_in_negotiation
        )
        self.negotiation_issues = self.negotiation.get_issue_df()
        self.negotiation_data = self.negotiation.get_proposal_df(
            interactions=self.interaction_effects_in_negotiation,
            include_utilities=True
        )
        if self.use_scoring:
            self.calculate_scores()
        else:
            raise NotImplementedError("Pairwise comparison not implemented yet")
    
    def calculate_scores(self) -> None:
        """Prepares a scores_df and, for each party, a batna score.
        All based on utilities and rater_error."""
        for party_name in self.negotiation.parties:
            utility_column_name = f"{party_name}_utility"
            utility_values = self.negotiation_data[utility_column_name].values
            party_batna = self.negotiation.parties[party_name].batna

            minval = min(min(utility_values), party_batna)
            maxval = max(max(utility_values), party_batna)
            value_range = maxval - minval
            noisy_utility_values = np.random.normal(
                loc=utility_values,
                scale=value_range*self.rater_error
            )
            noisy_batna = np.random.normal(
                loc=party_batna,
                scale=value_range*self.rater_error
            )

            minval = min(min(noisy_utility_values), noisy_batna)
            maxval = max(max(noisy_utility_values), noisy_batna)
            value_range = maxval - minval
            scaling_factor = 100 / value_range
            if not hasattr(self, "utility_to_score_params"):
                self.utility_to_score_params = dict()
            self.utility_to_score_params[party_name] = (minval, maxval, value_range, scaling_factor)
            proposal_scores = tuple([self.utility_to_score(party_name, x) for x in noisy_utility_values])
            self.negotiation_data[f"{party_name}_score"] = proposal_scores
            if self.include_batna_in_rating:
                if not hasattr(self, "party_batna_scores"):
                    self.party_batna_scores = dict()
                batna_score = self.utility_to_score(party_name, noisy_batna)
                self.party_batna_scores[party_name] = batna_score
        
    def utility_to_score(self, party_name: str, utility: float) -> float:
        """Converts a utility to a corresponding score for a given party"""
        p = self.utility_to_score_params[party_name]
        return (utility - p[0])*p[3]

    def score_to_utility(self, party_name: str, score: float) -> float:
        """Converts a score back to a corresponding utility for a given party"""
        p = self.utility_to_score_params[party_name]
        return (score / p[3]) + p[0]
    
    def build_scoring_model(self) -> None:
        """Builds and stores a scoring model for the negotiation"""
        print(f"Building PyMC model.")
        self.model = pm.Model()

        with self.model:
            ### Observed predictors ###
            data = dict()
            for data_column_name in self.negotiation_data.columns:
                if not data_column_name.startswith("Party"):
                    nn = f"{data_column_name}_data"
                    mutable_data = pm.MutableData(nn, self.negotiation_data[data_column_name])
                    data[nn] = mutable_data
            
            ### Outcome ###
            for party_name in self.negotiation.parties:
                nn = f"{party_name}_score"
                party_score = pm.MutableData(nn, self.negotiation_data[f"{party_name}_score"])
                data[nn] = party_score

            ### Priors ###
            NORMALS_STD = 100

            priors = dict()
            for party_name in self.negotiation.parties:
                priors[party_name] = dict()
                for data_column_name in self.negotiation_data.columns:
                    if (not data_column_name.startswith("Party")) and \
                        (self.interaction_effects_in_model or ("*" not in data_column_name)):
                        nn = f"{party_name}_{data_column_name}_beta"
                        beta = pm.Normal(nn, mu=0, sigma=NORMALS_STD)
                        priors[party_name][nn] = beta

                nn = f"{party_name}_rating_error"
                rating_error_prior = pm.HalfNormal(nn, 20)
                priors[party_name][nn] = rating_error_prior

                nn = f"{party_name}_intercept"
                intercept = pm.Normal(nn, mu=0, sigma=NORMALS_STD)
                priors[party_name][nn] = intercept

            ### Linear model and likelihood ###
            score_components = dict()
            estimated_score_dict = dict()
            for party_name in self.negotiation.parties:
                score_components[party_name] = []

                #Adding beta components
                for data_column_name in self.negotiation_data.columns:
                    if (not data_column_name.startswith("Party")) and \
                        (self.interaction_effects_in_model or ("*" not in data_column_name)):
                        component_data = data[f"{data_column_name}_data"]
                        component_prior = priors[party_name][f"{party_name}_{data_column_name}_beta"]
                        component = pm.math.dot(component_prior, component_data)
                        score_components[party_name].append(component)

                #Adding intercept
                score_components[party_name].append(priors[party_name][f"{party_name}_intercept"])

                #Summing up
                estimated_score = pm.Deterministic(
                    f"{party_name}_estimated_score",
                    reduce(lambda a,b: a+b, score_components[party_name])
                )
                estimated_score_dict[f"{party_name}_estimated_score"] = estimated_score

                #Likelihood
                observed = pm.Normal(
                    f"{party_name}_observed_score",
                    mu = estimated_score,
                    sigma = priors[party_name][f"{party_name}_rating_error"],
                    observed=data[f"{party_name}_score"]
                )
    
            summed_scores = pm.Deterministic("summed_estimated_scores", reduce(lambda a,b: a+b, estimated_score_dict.values()))
    
    def infer_model_and_get_results(self) -> Dict:
        """Estimate the posteriors, suggest good deal, and evaluate results."""
        if not hasattr(self, "model"):
            raise RuntimeError("Cannot estimate a model that has not been specified.")
        
        t1 = time()
        print("Fitting model.")
        with self.model:
            idata = pm.sample()

        print("Evaluating results.")
        temporary_deals = []
        for i in range(self.nr_test_deals):
            random_new_deal = negotiation.Proposal.new_random_proposal(f"random_prop_{i}", self.negotiation)
            temporary_deals.append(random_new_deal.get_dummy_evaluations(interaction_effects=self.interaction_effects_in_negotiation))
        test_deals = dict()
        for tmp_deal in temporary_deals:
            for item in tmp_deal:
                if f"{item}_data" not in test_deals:
                    test_deals[f"{item}_data"] = []
                test_deals[f"{item}_data"].append( tmp_deal[item] )
        for data_column_name, data_column in test_deals.items():
            test_deals[data_column_name] = np.array(data_column)

        with self.model:
            pm.set_data(test_deals)
            test_deal_posterior_predictive = pm.sample_posterior_predictive(
                idata, var_names=["Party1_estimated_score", "Party2_estimated_score", "summed_estimated_scores"]
            )
        
        # Function for distilling out the inferior deals
        def d1_superior_to_d2(party1_deal1, party1_deal2, party2_deal1, party2_deal2):
            p1d1_wins = (party1_deal1 > party1_deal2)
            d1_better_for_p1 = p1d1_wins.sum() / len(p1d1_wins)
            p2d1_wins = (party2_deal1 > party2_deal2)
            d1_better_for_p2 = p2d1_wins.sum() / len(p2d1_wins)
            return (d1_better_for_p1 > 0.5 and d1_better_for_p2 > 0.5)

        inferior = set()

        for i, j in combinations(range(self.nr_test_deals), 2):
            if i not in inferior and j not in inferior:
                party1_deal1 = test_deal_posterior_predictive.posterior_predictive["Party1_estimated_score"][:,:,i].to_numpy().flatten()
                party1_deal2 = test_deal_posterior_predictive.posterior_predictive["Party1_estimated_score"][:,:,j].to_numpy().flatten()
                party2_deal1 = test_deal_posterior_predictive.posterior_predictive["Party2_estimated_score"][:,:,i].to_numpy().flatten()
                party2_deal2 = test_deal_posterior_predictive.posterior_predictive["Party2_estimated_score"][:,:,j].to_numpy().flatten()
                if d1_superior_to_d2(party1_deal1, party1_deal2, party2_deal1, party2_deal2):
                    inferior.add(j)
                elif d1_superior_to_d2(party1_deal2, party1_deal1, party2_deal2, party2_deal1):
                    inferior.add(i)

        good_candidates = set(range(self.nr_test_deals)) - inferior

        # Function for distilling deals better than batnas for both parties
        def deal_better_than_batnas(deal_nr, significance=1.0):
            p1_util = test_deal_posterior_predictive.posterior_predictive["Party1_estimated_score"][:,:,deal_nr].to_numpy().flatten()
            p2_util = test_deal_posterior_predictive.posterior_predictive["Party2_estimated_score"][:,:,deal_nr].to_numpy().flatten()
            p1_batna_score = self.party_batna_scores["Party1"]
            p2_batna_score = self.party_batna_scores["Party2"]
            p1_wins = (p1_util > p1_batna_score)
            p1_better_proportion = p1_wins.sum() / len(p1_wins)
            p2_wins = (p2_util > p2_batna_score)
            p2_better_proportion = p2_wins.sum() / len(p2_wins)
            return (p1_better_proportion > significance and p2_better_proportion > significance)

        better_candidates = []
        significance = 1.02
        while len(better_candidates) == 0:
            significance -= 0.02
            better_candidates = list(filter(lambda x: deal_better_than_batnas(x, significance), list(good_candidates)))
        
        # Function for ordering list of deals in terms of total value (if a has more total value than b more than 50 % of cases, a goes higher, else b)
        def d1_higher_total_value_to_d2(d1, d2):
            total_value_d1 = test_deal_posterior_predictive.posterior_predictive["summed_estimated_scores"][:,:,d1].to_numpy().flatten()
            total_value_d2 = test_deal_posterior_predictive.posterior_predictive["summed_estimated_scores"][:,:,d2].to_numpy().flatten()
            d1wins = (total_value_d1 > total_value_d2)
            d1better_proportion = d1wins.sum() / len(d1wins)
            return d1better_proportion - 0.5
        
        sorted_candidates = sorted(list(better_candidates), key=cmp_to_key(d1_higher_total_value_to_d2))
        best_candidate = sorted_candidates[-1]
        best_candidate_deal = temporary_deals[best_candidate]
        best_p = negotiation.Proposal.new_from_dummy_issues("best_candidate_proposal", self.negotiation, best_candidate_deal)
        best_p.evaluate_utilities(self.negotiation.parties)
        t2 = time()

        # Function for evaluating proposal fitness (a fit proposal is one that is acceptable by all)
        def proposal_beats_all_batnas(proposal):
            for party_name in self.negotiation.parties:
                if self.party_batna_scores[party_name] >= self.utility_to_score(party_name, proposal.utilities[party_name]):
                    return False
            return True
        
        # Saving fitness metrics
        results = dict()
        results["trial_timestamp"] = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        results["is_acceptable"] = proposal_beats_all_batnas(best_p)
        results["P1_batna_score"] = self.party_batna_scores["Party1"]
        results["P2_batna_score"] = self.party_batna_scores["Party2"]
        results["P1_batna_utility"] = self.negotiation.parties["Party1"].batna
        results["P2_batna_utility"] = self.negotiation.parties["Party2"].batna
        results["P1_utility"] = best_p.utilities["Party1"]
        results["P2_utility"] = best_p.utilities["Party2"]
        results["total_utility"] = results["P1_utility"] + results["P2_utility"]
        results["P1_optimality"] = best_p.get_optimality(self.negotiation, "Party1")
        results["P2_optimality"] = best_p.get_optimality(self.negotiation, "Party2")
        results["total_optimality"] = best_p.get_optimality(self.negotiation)
        results["P1_batna_goodness"] = self.negotiation.get_batna_goodness("Party1")
        results["P2_batna_goodness"] = self.negotiation.get_batna_goodness("Party2")
        lower, upper = self.negotiation.get_utility_range("Party1")
        results["P1_utility_lower_bound"] = lower
        results["P1_utility_upper_bound"] = upper
        results["P1_utility_range"] = upper - lower
        lower, upper = self.negotiation.get_utility_range("Party2")
        results["P2_utility_lower_bound"] = lower
        results["P2_utility_upper_bound"] = upper
        results["P2_utility_range"] = upper - lower
        lower, upper = self.negotiation.get_utility_range()
        results["total_utility_lower_bound"] = lower
        results["total_utility_upper_bound"] = upper
        results["total_utility_range"] = upper - lower

        results["seconds_elapsed"] = t2 - t1
        results["nr_issues"] = self.nr_issues
        results["nr_issue_dummy_vars"] = len([x for x in self.negotiation_data.columns if not x.startswith("Party")])
        results["interactions_in_model"] = self.interaction_effects_in_model
        results["interactions_in_negotiation"] = self.interaction_effects_in_negotiation
        results["nr_scored_proposals"] = self.nr_proposals
        results["rater_error"] = self.rater_error
        results["nr_test_deals_for_suggesting"] = self.nr_test_deals

        divergent = idata["sample_stats"]["diverging"].to_numpy().flatten()
        divfrac = divergent.sum() / len(divergent)
        results["pymc_divergence_fraction"] = divfrac

        return results
    

@dataclass
class Experiment:
    runs_per_setting: int = 1
    settings_and_levels: Dict = field(default_factory=dict)
    results: List = field(default_factory=list)

    def run(self):
        nr_trials = self.runs_per_setting * reduce(lambda x,y: x*y, [len(v) for v in self.settings_and_levels.values()])
        print(f"Running {nr_trials} trials.")
        i = 1
        for settings_tuple in product(*list(self.settings_and_levels.values())):
            for _ in range(self.runs_per_setting):
                keywords = dict(zip(list(self.settings_and_levels.keys()), settings_tuple))
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Trial {i} with keywords", keywords)
                trial = Trial(**keywords)
                trial.build_scoring_model()
                res = trial.infer_model_and_get_results()
                self.results.append(res)
                del trial #Just making sure we free up memory between trials
                i += 1
    
    def generate_trial_batch_file(self, filename:str) -> None:
        """Uses own settings to generate a batch file which can later be consumed gradually"""
        dump_strings = []
        for _ in range(self.runs_per_setting):
            for settings_tuple in product(*list(self.settings_and_levels.values())):
                keywords = dict(zip(list(self.settings_and_levels.keys()), settings_tuple))
                dump_string = json.dumps(keywords)
                dump_strings.append(dump_string)
        with open(filename, "w") as f:
            f.write("\n".join(dump_strings))
    
    @staticmethod
    def run_trials_from_file_and_save(batch_filename:str, results_filename:str, max_runs_before_stop:int) -> None:
        """Consumes trial settings from batch file and runs trials. Saves results to results file."""
        print(f"Running {max_runs_before_stop} trials, or less if batch file becomes empty.")
        i = 1
        batch_empty = False
        while i < max_runs_before_stop+1 and not batch_empty:
            with open(batch_filename, "r") as f:
                batch_strings = f.readlines()
            if len(batch_strings) == 0:
                batch_empty = True
                continue
            job = batch_strings[0]
            with open(batch_filename, "w") as f:
                f.writelines(batch_strings[1:])
            keywords = json.loads(job)
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Trial {i}/{max_runs_before_stop} with keywords", keywords)
            trial = Trial(**keywords)
            trial.build_scoring_model()
            res = json.dumps(trial.infer_model_and_get_results())
            with open(results_filename, "a") as f:
                f.write(res + "\n")
            del trial #Just making sure we free up memory between trials
            i += 1
        print("Done!")
    
    @staticmethod
    def read_test_results_into_df(results_filename:str) -> pd.DataFrame:
        """Reads test results from file and returns them as a DataFrame"""
        with open(results_filename, "r") as f:
            results = [json.loads(x) for x in f.readlines()]
        return pd.DataFrame(results)


if __name__ == "__main__":



    x = Experiment(
        runs_per_setting=1,
        settings_and_levels={
            "nr_issues": [2],
            "nr_proposals": [25],
            "interaction_effects_in_negotiation": [False,True],
            "interaction_effects_in_model": [False,True]
            }
        )
    
    x.generate_trial_batch_file("test_batch.json")
    Experiment.run_trials_from_file_and_save("test_batch.json", "test_results.json", 100)
    print(Experiment.read_test_results_into_df("test_results.json"))



    '''    x.run()
    df = pd.DataFrame(x.results)
    df.to_excel("my_first_run.xlsx")
    print(df)'''

    """
    Other things to track!
    - Ratio between weights estimated vs real. Explains how well we captured reality.
    - How some issues are more important than others, either because the weights are large or because the ranges are broad. Or both.
    """