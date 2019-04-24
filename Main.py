#!/usr/bin/env python3
import sys
import logging
import matplotlib.pyplot as pyplot

from SVMPredictingAgent import SVMPredictingAgent

if __name__ == "__main__":
    prediction_agent = SVMPredictingAgent("poly", "balanced_training.xlsx", "balanced_test.xlsx")
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # Testing various C parameters gives different accuracy ratios.
    list_c = []
    list_accuracy = []
    #for i in range(2000, 4000, 100):
    prediction_agent.split_by_patient(3420)
    #for i in range(3620, 3640, 5):
    #prediction_agent.pca_trans_plot(3620)
    #prediction_agent.k_fold_stratified_test(3620, 10)
