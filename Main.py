#!/usr/bin/env python3

from SVMPredictingAgent import SVMPredictingAgent

if __name__ == "__main__":
    # Initializing the agent
    prediction_agent = SVMPredictingAgent("poly", "training_data.xlsx", "balanced_test.xlsx")
    # Using prediction by patient on the given file ("balanced_test.xlsx")
    prediction_agent.split_by_patient(3640, prob_param=True, gamma_scale=False)
