#!/usr/bin/env python3
import logging
import sys
from collections import Counter

import matplotlib.pyplot as pyplot
import numpy
import pandas
import pandas as pd
from scipy import interp
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


class SVMPredictingAgent:
    """
    Class used to initialize various SVMs and use their prediction abilities to help diagnose patients based on data
    extracted from 3D microcalcifications present in their breast tissue. Initialized with the desired SVM kernel,
    the .xlsx file containing the training data and the .xlsx file containing the patient's data.
    """

    def __init__(self, kernel, training_file, test_file):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        self.training_file = training_file
        self.test_file = test_file
        self.kernel = kernel
        self.training_samples, self.training_classes, \
            self.training_labels, self.patients_list_training = self.parse_data(training_file)
        self.prediction_samples, self.prediction_classes, \
            self.prediction_labels, self.patients_list_prediction = self.parse_data(test_file)
        self.micro_svm = svm.SVC()

    def predict(self, c_param):
        """
        Public predict method that uses the prediction samples given during the agent initialization.
        :param c_param:
        :return: The accuracy of the prediction
        """
        return self.__predict(c_param, self.prediction_samples, self.prediction_classes)

    def split_by_patient(self, c_param, prob_param=False, gamma_scale=False):
        """
        Splits the prediction dataset given by the user by patient, and gives a prediction for every patient.
        Gives the accuracy of the prediction for testing purposes or average benign/malignant probabilities for
        each patient.
        :param gamma_scale: Whether to use scaling gamma parameter or the automatic one.
        :param prob_param: Whether or not to calculate probabilities instead of directly classifying the micros
        :param c_param: The error compensation parameter.
        """
        if self.kernel == "poly":
            self.micro_svm = self.init_poly_svm(c_param, prob=True, gamma_scale=gamma_scale)
        elif self.kernel == "linear":
            self.micro_svm = self.init_linear_svm(c_param, prob=True)
        patient_array = [[] for i in range(
            int(self.patients_list_prediction[-1][0]) - int(self.patients_list_prediction[0][0]) + 1)]
        patient_class = [[] for i in range(
            int(self.patients_list_prediction[-1][0]) - int(self.patients_list_prediction[0][0]) + 1)]
        for i in range(len(self.patients_list_prediction)):
            to_add = int(self.patients_list_prediction[i][0]) - int(self.patients_list_prediction[0][0])
            patient_array[to_add].append(self.patients_list_prediction[i][1:])
            patient_class[to_add].append(self.prediction_classes[i])
        patient_array = numpy.array(patient_array)
        patient_class = numpy.array(patient_class)
        counter = 0
        predictions_accuracy = []

        if prob_param is False:
            for patient_data in patient_array:
                patient_data = numpy.array(patient_data)
                counter += 1
                print("Current patient data:", counter, "has the following shape:", patient_data.shape)
                print("Current patient classes:", counter, "has the following values:", patient_class[counter - 1])
                current_prediction = self.__predict(c_param, patient_data,
                                                    [patient_class[counter - 1][0] for i in range(len(patient_data))],
                                                    self.micro_svm)
                predictions_accuracy.append(current_prediction)
                print("Current patient:", counter, "predicted with accuracy:", current_prediction)

            predictions_accuracy_mean = [numpy.mean(predictions_accuracy)] * len(predictions_accuracy)
            print(predictions_accuracy_mean[0])
            fig, ax = pyplot.subplots()
            ax.plot(range(counter), predictions_accuracy, label="Patient Predictions Accuracy", marker='o')
            ax.plot(range(counter), predictions_accuracy_mean, label="Mean Prediction Accuracy", linestyle='--')
            ax.legend(loc='upper right')
            ax.set(xlabel='Fold', ylabel='Accuracy',
                   title=('Accuracy test per patient with 3-' + self.kernel + ' kernel, C=' + str(c_param)))
            ax.grid()
            ax.text(0, 0.7, ("Mean = " + str(predictions_accuracy_mean[0])))
            print("Saving file :", ('Accuracy test per patient with ' + self.kernel + ' kernel, C=' + str(c_param)))
            fig.savefig(('Accuracy test per patient with 3-' + self.kernel + ' kernel, C=' + str(c_param)))

        else:
            r = range(1, patient_array.__len__() + 1)
            print("r=", r)
            green_bars = []
            orange_bars = []
            text_report = []
            for patient_data in patient_array:
                patient_data = numpy.array(patient_data)
                counter += 1
                # print("Current patient data:", counter, "has the following shape:", patient_data.shape)
                # print("Current patient classes:", counter, "has the following values:", patient_class[counter-1])
                current_prediction = numpy.array(self.micro_svm.predict_proba(patient_data))
                predictions_accuracy.append(current_prediction)
                green_bars.append(current_prediction.mean(axis=0)[0])
                orange_bars.append(current_prediction.mean(axis=0)[1])
                predict_tuple = (float('%.3f' % (current_prediction.mean(axis=0)[0])),
                                 float('%.3f' % (current_prediction.mean(axis=0)[1])))
                # Create diagnosis
                if predict_tuple[0] >= 0.82:
                    diag = "Patient healthy, no biopsy needed."
                elif predict_tuple[0] >= 0.71:
                    diag = "Patient probably healthy, low necessity of biopsy."
                elif predict_tuple[1] >= 0.75:
                    diag = "Patient has cancer, biopsy needed."
                elif predict_tuple[1] >= 0.6:
                    diag = "Patient probably has cancer, high necessity of biopsy."
                else:
                    diag = "Diagnosis inconclusive, biopsy recommended."

                text_report.append((predict_tuple[0], predict_tuple[1], diag))

            predictions_accuracy = numpy.array(predictions_accuracy)
            raw_data = {'greenBars': green_bars, 'orangeBars': orange_bars}
            df = pd.DataFrame(raw_data)
            totals = [i + j for i, j in zip(df['greenBars'], df['orangeBars'])]
            greenBars = [i / j * 100 for i, j in zip(df['greenBars'], totals)]
            orangeBars = [i / j * 100 for i, j in zip(df['orangeBars'], totals)]
            # Plotting the patients' results
            fig = pyplot.figure(figsize=(r.__len__() / 2.2, 8))
            fig.suptitle("Patient predictions and diagnosis", fontsize=13)
            ax = pyplot.subplot(121, title="Patient results graph")
            barWidth = 0.85
            # Create green Bars
            grB = pyplot.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Benign level")
            # Create orange Bars
            orB = pyplot.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth,
                             label="Malignant level")
            # Create blue Bars
            for i in r:
                mic = pyplot.plot([i for x in range(len(predictions_accuracy[i - 1]))],
                                  predictions_accuracy[i - 1][:][:, 0] * 100,
                                  '.', color="gray", markersize=2, label="Microcalcification")[0]

            # Custom x axis
            pyplot.xlabel("Patient Predictions")
            pyplot.xticks(r)
            pyplot.table(cellText=text_report, loc="right", colWidths=[0.1, 0.1, 0.8],
                         rowLabels=r)
            ax.legend(handles=[mic, grB, orB], loc="lower left")
            fig.tight_layout()
            # Show graphic
            pyplot.show()

    def __predict(self, c_param, prediction_samples, prediction_classes, krnl=None):
        """
        Predicts the class of the given data using an SVM.
        :param c_param: The error compensation
        :param prediction_samples: Data given as input to the function.
        :param: prediction_classes: The expected results from the predictions.
        :return: The accuracy of the predictions.
        """
        logging.debug("Starting predictions of data.")
        if krnl is None:
            if self.kernel == "poly":
                self.micro_svm = self.init_poly_svm(c_param)
            elif self.kernel == "linear":
                self.micro_svm = self.init_linear_svm(c_param)

        predictions = self.micro_svm.predict(prediction_samples)
        accuracy = self.compute_accuracy(prediction_classes, predictions)

        logging.debug("End of predictions.")
        return accuracy

    def init_linear_svm(self, c_param, prob=False):
        """
            Initializes and trains an SVM to predict whether described microcalcifications in input data are benign or
            malicious.
            :param prob: Whether to classify the data with probabilities, or assign classes directly.
            :param c_param: The error compensation parameter
            :return: The trained SVM to be used for predictions.
            """
        logging.debug("Initializing linear SVM from: " + self.training_file)

        micro_svm = svm.SVC(kernel="linear", gamma="auto", C=c_param, probability=prob)
        micro_svm.fit(self.training_samples, self.training_classes)

        logging.debug("End of SVM initialization.")
        return micro_svm

    def init_poly_svm(self, c_param, prob=False, dgr=2, gamma_scale=False):
        """
        Initializes and trains an SVM to predict whether described microcalcifications in input data are benign or
        malicious.
        :return: The trained SVM to be used for predictions.
        """
        logging.debug("Initializing SVM from: " + self.training_file)
        if gamma_scale:
            micro_svm = svm.SVC(kernel="poly", degree=dgr, gamma="scale", C=c_param, probability=prob)
        else:
            micro_svm = svm.SVC(kernel="poly", degree=dgr, gamma="auto", C=c_param, probability=prob)
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
        patient_numbers[:, 1:] = data_samples

        logging.debug("End of parsing.")
        return data_samples, data_classes, data_labels, patient_numbers

    def pca_trans_plot(self, c_param):

        def make_meshgrid(x, y, h=.02):
            """
            Create a mesh of points to plot in.
            :param x: The data to base x-axis meshgrid on.
            :param y: The data to base y-axis meshgrid on.
            :param h: The stepsize for meshgrid (optional).
            :return : (xx, yy) ndarrays
            """
            x_min, x_max = x.min() - 1, x.max() + 1
            y_min, y_max = y.min() - 1, y.max() + 1
            xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                                    numpy.arange(y_min, y_max, h))
            return xx, yy

        def plot_contours(ax, clf, xx, yy, **params):
            """
            Plot the decision boundaries for a classifier.

            Parameters
            ----------
            :param ax: matplotlib axes object
            :param clf: a classifier
            :param xx: meshgrid ndarray
            :param yy: meshgrid ndarray
            :param params: dictionary of parameters to pass to the contourf function (optional).
            :return : out, the response of the contourf function.
            """
            Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            out = ax.contourf(xx, yy, Z, **params)
            return out

        # Use feature extraction to get the features that influence the most the distribution of the dataset.
        logging.debug("Starting PCA feature extraction.")
        pca = PCA(n_components=2).fit(self.training_samples)
        X = pca.transform(self.training_samples)
        y = self.training_classes
        logging.debug("End of PCA feature extraction.")

        # We create an instance of SVM and fit out data. No scaling needed for the support vectors plot
        logging.debug("Initialising SVM with 2D PCA matrix.")
        models = (svm.SVC(kernel='linear', C=c_param),
                  svm.SVC(kernel='poly', degree=2, C=c_param))
        models = (clf.fit(X, y) for clf in models)
        logging.debug("End of SVM initialization with 2D PCA matrix.")

        # Title for the plots
        titles = ('SVC with linear kernel',
                  'SVC with 2-polynomial kernel')

        # Set-up 2x1 grid for plotting linear and 2-poly kernel.
        fig, sub = pyplot.subplots(2, 1)
        pyplot.subplots_adjust(wspace=0.8, hspace=0.8)

        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)

        for clf, title, ax in zip(models, titles, sub.flatten()):
            plot_contours(ax, clf, xx, yy,
                          cmap=pyplot.cm.coolwarm, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=pyplot.cm.coolwarm, s=20, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('PCA Parameter 1')
            ax.set_ylabel('PCA Parameter 2')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)

        pyplot.show()

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

        skf = StratifiedKFold(n_splits=k, shuffle=False)
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

        for i in range(k):
            if self.kernel == "poly":
                self.micro_svm = svm.SVC(kernel="poly", degree=2, gamma="scale", C=c_param)
                self.micro_svm.fit(X_train[i], y_train[i])
            elif self.kernel == "linear":
                self.micro_svm = svm.SVC(kernel="linear", gamma="auto", C=c_param)
                self.micro_svm.fit(X_train[i], y_train[i])
            list_c.append(i)
            accuracy = self.__predict(c_param, X_test[i], y_test[i], self.micro_svm)
            print(c_param, " error with accuracy: ", accuracy)
            list_accuracy.append(accuracy)

        k_fold_mean = [numpy.mean(list_accuracy)] * len(list_c)
        print(k_fold_mean[0])

        fig, ax = pyplot.subplots()
        ax.plot(list_c, list_accuracy, label="Data", marker='o')
        ax.plot(list_c, k_fold_mean, label="Mean", linestyle='--')
        ax.legend(loc='upper right')
        ax.set(xlabel='Fold', ylabel='Accuracy',
               title=('Accuracy for ' + (str(k) + '-fold test with 2-poly kernel, scaling gamma, C=' + str(c_param))))
        ax.grid()
        ax.text(0, 0.8, ("Mean = " + str(k_fold_mean[0])))
        fig.savefig((str(k) + '-fold test with 2-poly kernel, scaling gamma, C=' + str(c_param)))

    def k_fold_ROC_test(self, c_param, k):
        """
        Uses the training data set to create a ROC curve using cross-validation. The procedure has a single parameter
        called k that refers to the number of groups that a given data sample is to be split into.
        It is stratified because it ensures that that each fold has the same proportion of observations with regard to
        the class outcome value.
        :param c_param: The error compensation parameter.
        :param k: The number of folds for the stratified test.
        :return: Saves a graph of the ROC curves and AUC of each fold, as well an average.
        """

        logging.debug("Creating ROC curve using K-fold test with data from: " + self.training_file)

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

        tprs = []
        aucs = []
        mean_fpr = numpy.linspace(0, 1, 100)

        for i in range(k):
            if self.kernel == "poly":
                self.micro_svm = svm.SVC(kernel="poly", degree=2, gamma="auto", C=c_param, probability=True)
                self.micro_svm.fit(X_train[i], y_train[i])
            elif self.kernel == "linear":
                self.micro_svm = svm.SVC(kernel="linear", gamma="auto", C=c_param, probability=True)
                self.micro_svm.fit(X_train[i], y_train[i])
            list_c.append(i)
            probas_ = self.micro_svm.predict_proba(X_test[i])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test[i], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            pyplot.plot(fpr, tpr, lw=1, alpha=0.3,
                        label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        pyplot.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                    label='Chance', alpha=.8)

        mean_tpr = numpy.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = numpy.std(aucs)
        pyplot.plot(mean_fpr, mean_tpr, color='b',
                    label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                    lw=2, alpha=.8)

        std_tpr = numpy.std(tprs, axis=0)
        tprs_upper = numpy.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = numpy.maximum(mean_tpr - std_tpr, 0)
        pyplot.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                            label=r'$\pm$ 1 std. dev.')

        pyplot.xlim([-0.05, 1.05])
        pyplot.ylim([-0.05, 1.05])
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.title('SVM receiver operating characteristic curve')
        pyplot.legend(loc="lower right")
        # pyplot.show()
        pyplot.savefig((str(k) + '-fold ROC with 3-poly kernel and PN, C=' + str(c_param)))

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
