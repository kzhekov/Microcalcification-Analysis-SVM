# PROJECT DESCRIPTION
The goal of the project is to implement a machine-learning assisted algorithm that helps detect whether a patient has breast cancer or not, based on the analysis of characteristics extracted from the 3D models of microcalcifications present in the patient’s breast tissue.
# IMPLEMENTATION
For the realization of the project, an SVM was chosen as the machine-learning method used to classify the microcalcifications. The model was trained and tested with the given data set of 3562 microcalcifications, each having 150 characteristics. 
# DIAGNOSING THE PATIENT
Once the algorithm was implemented and well-tested, the program diagnoses the patient using the same characteristics as the ones used in the training data set. The diagnosis is based on the number of microcalcifications detected as malignant, and the SVM’s prediction performance in the various tests.
A visual representation is shown, representing the percentages of chance that a patient has cancer, based on the average probability classifications of his microcalcifications.
# TESTING THE ALGORITHM
To obtain an algorithm that is to be trusted, various tests are used in order to maximize the test’s accuracy. The tests were realised using a patient-by-patient basis, a 10-fold stratified test as well as a 96-fold stratified test (simulating a patient-by-patient test).
Various kernels (linear, 2-poly, 3-poly) and SVM parameters were extensively tested and graphed, in order to find the optimal implementation for the classification.
Particular attention was given to the fact that in disease diagnosing sensitivity is the extent to which actual positives are not overlooked. In this case, what is needed is a highly sensitive test that rarely overlooks an actual positive (for example, showing "nothing bad" despite something bad existing).
# IMPLEMENTATION
The program is implemented as an agent which parses the given .xlsx files when initialized. The agent can then use various functions to classify microcalcifications and diagnose the patients. The functions are documented, but here is roughly what the program can do:
•	Takes an array of data containing microcalcification characteristics and classifies it into benign or malignant.
•	Initializes a linear or polynomial support vector machine that can classify directly or with probabilities
•	Parse the data from the given .xlsx files in order to be used in training and classification
•	Plot the SVM decision boundaries in a 2D plane using PCA feature extraction, but results are disappointing since there is no good way of projecting the data on a 2D plane using only the most important feature
•	Plot a K-fold stratified test on the training data, with a varying error-compensation parameter and number of folds.
•	Plot the ROC curve on a K-fold stratified test of the training data with a varying error-compensation parameter and number of folds
•	Preprocess the data given in the form of .xlsx by normalizing it
•	Compute the accuracy of a prediction based on the known classes
•	Test the accuracy for the predictions of the testing data set given as a parameter to the agent
•	Plot and give a diagnosis for every patient found in the .xlsx file, displaying it at the end
