#!/usr/bin/env python3
import sys
import logging
import matplotlib.pyplot as pyplot

from SVMPredictingAgent import SVMPredictingAgent

if __name__ == "__main__":
    prediction_agent = SVMPredictingAgent("poly", "training_data.xlsx", "balanced_test.xlsx")
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Testing various C parameters gives different accuracy ratios.
    list_c = []
    list_accuracy = []
    prediction_agent.split_by_patient(3620)
    #for i in range(3620, 3640, 5):
    #    prediction_agent.k_fold_stratified_test(i, 96)
