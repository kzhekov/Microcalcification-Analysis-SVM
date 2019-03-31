#!/usr/bin/env python3
from collections import Counter
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import numpy
import pandas
import logging
import sys
import matplotlib.pyplot as pyplot


class SVMPredictingAgent:

    def __init__(self, kernel, training_file, test_file):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.training_file = training_file
        self.test_file = test_file
        self.kernel = kernel
        self.training_samples, self.training_classes,\
            self.training_labels, self.patients_list_training = self.parse_data(training_file)
        self.prediction_samples, self.prediction_classes,\
            self.prediction_labels, self.patients_list_prediction = self.parse_data(test_file)
        self.micro_svm = svm.SVC()

    def predict(self, c_param):
        """
        Public predict method that uses the prediction samples given during the agent initialization.
        :param c_param:
        :return:
        """
        self.__predict(c_param, self.prediction_samples, self.prediction_classes)

    def split_by_patient(self, c_param):
        """
        Splits the predictions by patient.
        :param c_param:
        :return:
        """
        patient_array = [[] for i in range(int(self.patients_list_training[-1][0]))]
        print("Number of patients: ", patient_array.__len__())
        for i in range(len(self.patients_list_training)):
            patient_array[int(self.patients_list_training[i][0])-1].append(self.patients_list_training[i][1:])
        patient_array = numpy.array(patient_array)
        counter = 0
        for patient_data in patient_array:
            patient_data = numpy.array(patient_data)
            counter += 1
            print("Current patient:", counter, "has the following number of characteristics:", patient_data.shape)
            print("Current patient:", counter, "predicted with accuracy:", self.__predict(c_param, patient_data, [0 if i < 47 else 1 for i in range(len(patient_data))]))

    def __predict(self, c_param, prediction_samples, prediction_classes):
        """
        Predicts the class of the given data using an SVM.
        :param c_param: The error compensation
        :param prediction_samples: Data given as input to the function.
        :param: prediction_classes: The expected results from the predictions.
        :return: The accuracy of the predictions.
        """
        logging.debug("Starting predictions of data.")
        if self.kernel == "poly":
            self.micro_svm = self.init_poly_svm(c_param)
        elif self.kernel == "linear":
            self.micro_svm = self.init_linear_svm(c_param)

        predictions = self.micro_svm.predict(prediction_samples)
        #print(predictions)
        accuracy = self.compute_accuracy(prediction_classes, predictions)

        logging.debug("End of predictions.")
        return accuracy

    def init_linear_svm(self, c_param):
        """
            Initializes and trains an SVM to predict whether described microcalcifications in input data are benign or
            malicious.
            :param c_param: The error compensation parameter
            :return: The trained SVM to be used for predictions.
            """
        logging.debug("Initializing linear SVM from: " + self.training_file)

        micro_svm = svm.SVC(kernel="linear", gamma="auto", C=c_param)
        micro_svm.fit(self.training_samples, self.training_classes)

        logging.debug("End of SVM initialization.")
        return micro_svm

    def init_poly_svm(self, c_param, dgr=2):
        """
        Initializes and trains an SVM to predict whether described microcalcifications in input data are benign or
        malicious.
        :return: The trained SVM to be used for predictions.
        """
        logging.debug("Initializing SVM from: " + self.training_file)

        micro_svm = svm.SVC(kernel="poly", degree=dgr, gamma="auto", C=c_param)
        micro_svm.fit(self.training_samples, self.training_classes)

        logging.debug("End of SVM initialization.")
        return micro_svm

    def parse_data(self, filename):
        """
        Parses data from an .xlsx file into a matrix that can be fed to the SVM for training or predictions.
        The data has to be in the following format:
        Patient_ID|... all the data ...|Class
        :param filename: The name of the .xlsx file.
        :return: A triple (data matrix, class matrix, data labels). The data labels are the column names.
        """
        logging.debug("Starting .xlsx parsing.")

        data = pandas.read_excel(filename)
        patient_numbers = data.values[:, :-1]
        data_classes = data.values[:, -1]  # Take last column for class labels (0 = benign, 1 = malignant)
        data_samples = data.values[:, 1:-1]  # Remove first column (patient number) and last column (result)
        data_labels = list(data.columns)  # Labels of all columns
        data_samples = self.preprocess_data(data_samples)

        logging.debug("End of parsing.")
        return data_samples, data_classes, data_labels, patient_numbers

    def k_fold_stratified_test(self, c_param, k):
        """
        Uses the training data set to test the network's accuracy. Cross-validation is a statistical method used to
        estimate the skill of machine learning models.The procedure has a single parameter called k that refers to the
        number of groups that a given data sample is to be split into. It is stratified because it ensures that that
        each fold has the same proportion of observations with regard to the class outcome value.
        :param c_param: The error compensation parameter.
        :param k: The number of folds for the stratified test.
        :return: Saves a graph of the accuracy and mean accuracy of the test for each fold.
        """

        logging.debug("Starting stratified K-fold test with data from: " + self.training_file)

        skf = StratifiedKFold(n_splits=k, shuffle=True)
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for a, b in (skf.split(self.training_samples, self.training_classes)):
            X_train.append(self.training_samples[a])
            y_train.append(self.training_classes[a])
            X_test.append(self.training_samples[b])
            y_test.append(self.training_classes[b])

        X_train = numpy.array(X_train)
        y_train = numpy.array(y_train)
        X_test = numpy.array(X_test)
        y_test = numpy.array(y_test)

        list_c = []
        list_accuracy = []

        for i in range(5):
            if self.kernel == "poly":
                self.micro_svm = svm.SVC(kernel="poly", degree=2, gamma="auto", C=c_param)
                self.micro_svm.fit(X_train[i], y_train[i])
            elif self.kernel == "linear":
                self.micro_svm = svm.SVC(kernel="linear", gamma="auto", C=c_param)
                self.micro_svm.fit(X_train[i], y_train[i])
            list_c.append(i)
            accuracy = self.__predict(c_param, X_test[i], y_test[i])
            print(c_param, " error with accuracy: ", accuracy)
            list_accuracy.append(accuracy)

        k_fold_mean = [numpy.mean(list_accuracy)]*len(list_c)
        print(k_fold_mean)
        fig, ax = pyplot.subplots()
        ax.plot(list_c, list_accuracy, label="Data", marker='o')
        ax.plot(list_c, k_fold_mean, label="Mean", linestyle='--')
        ax.legend(loc='upper right')
        ax.set(xlabel='Fold', ylabel='Accuracy',
               title=('Accuracy for K-fold test with 2-poly kernel and PN, C=' + str(c_param)))
        ax.grid()
        ax.text(0, 0.7, ("Mean = "+str(k_fold_mean[0])))
        fig.savefig(('10 K-fold test with 2-poly kernel and PN, C=' + str(c_param)))
        #pyplot.show()

    @staticmethod
    def preprocess_data(data_matrix):
        """
        Normalizes the data before the SVM can process it. This is a necessary step, without which the SVM never reaches
        the end of its training or prediction.
        :param data_matrix: The matrix with the data to be normalized
        :return: The matrix with normalized data
        """
        logging.debug("Starting data normalization.")

        scaler = MinMaxScaler()  # Used to normalize data between 0 and 1
        new_matrix = scaler.fit_transform(data_matrix)

        logging.debug("End of data normalization.")
        return new_matrix

    @staticmethod
    def compute_accuracy(facts, predictions):
        """
        Computes the accuracy between a given list of facts (class labels from data matrix) and the SVM predictions about
        the same data. This is used on data that is unknown to the SVM, i.e. not used in the training set.
        :param facts: The class labels that are known to be true.
        :param predictions: The predictions obtained from the SVM.
        :return: The accuracy ratio of the predictions.
        """
        logging.debug("Computing accuracy.")
        prediction_counter = Counter(predictions)
        return (sum(min(n, prediction_counter[l]) for l, n in Counter(facts).items())) / (facts.__len__())
