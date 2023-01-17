# External libraries (URL):
#	- Pytorch (https://pytorch.org/docs/stable/torch.html)
#	- Numpy (https://numpy.org/)
#	- NLTK (https://www.nltk.org/api/nltk.html)
#	- Matplotlib (https://matplotlib.org/stable/api/index) 
#	- Pandas (https://pandas.pydata.org/docs/) 
#	- Scikit Learn (https://scikit-learn.org/stable/modules/classes.html) 
#	- Imblearn (https://imbalanced-learn.org/stable/)
# Publicly available code (URL + how many lines modified, added):
#	- https://discuss.pytorch.org/t/how-to-plot-train-and-validation-accuracy-graph/105524/2, modified 5 of the 8 used lines of the first code snippet to graph our train and validation accuracy graph.
#   - https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/, used the last 8 lines of the last code snippet for our training loop.
#   - https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html, used the model class as a reference when building the LSTM-RNN portion of the LSTM-RNN+FFN.
#   - https://www.kaggle.com/code/arunmohan003/sentiment-analysis-using-lstm-pytorch/notebook. We used this publically available notebook as a basis for our LSTM-RNN + Feed Forward Network. From this notebook, we used a modified version of the 69 line training loop, and used /rewrote a modified version of the 44 line Model.
#   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html. Modified and added 2 lines of code to enable our results to be displayed in a confusion matrix.
#   - https://pytorch.org/tutorials/beginner/saving_loading_models.html, Used and Modified ~4-5 lines of code to allow our model to be saved for use later on or in other jupyter notebooks.
# Team’s code (number of lines, what the file do):
#	- CompareModels.ipynb (1269 lines - to compare performance of model when being trained on 4 different datasets)
#	######  The next 4 files are clones of each other with different training datasets as inputs and different models and onehot dictionaries as outputs
#	- Final_Model_on_Imbalanced_Combo_Dataset.ipynb (1371 lines - experiment and trained model on imbalanced combined dataset to get parameters for best accuracy and loss value)
#	- Final_Model_on_OSF_Dataset.ipynb (1389 lines - experiment and trained model on OSF dataset to get parameters for best accuracy and loss value)
#	- Final_Model_on_Oversampled_Combo_Dataset.ipynb (1399 lines - experiment and trained model on oversampled combined dataset to get parameters for  best accuracy and loss value)
#	- Final_Model_on_Undersampled_Combo_Dataset.ipynb (1380 lines - experiment and trained model on undersampled combined dataset to get parameters for  best accuracy and loss value)
#	- baselineModels.ipynb (945 lines - baseline models implementation and experimentation)
#	- RNN.py (143 lines - initial skeleton RNN for later development) 
#	- prepDataset.ipynb (593 lines - data preprocessing)
