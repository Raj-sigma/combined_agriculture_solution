# combined_agriculture_solution
Combined Agriculture Solution is a combined app aimed to help farmers to increase crop productivity, deal with unseen circumstances and increase profit. It brings the advancement of technology and gathered scientific information to local farmer using artificial intelligence. It uses random forest algorithm, feed forward neural network and convolutional neural network along with transfer learning to make good prediction. The UI is made using flask for easy interaction with the users.

# Problem 
There is no mechanism for farmers to know which crop will give best yield, which fertiliser will be best for the soil, on which day should we irrigate the soil to get get best result, how to identify crop disease symptoms etc. 

# Classification and Regression Model
FeedForward Neural Network: I uses 3 layer (no of neurons vary) feedforward neural network (this is the best parameter I found considering overfitting) with Relu activation function. For training regression models I used mseloss function (mean squared loss function) and for classification I used cross entropy loss function. For optimiser I preferred using Adam optimizer as it uses various concept like momentum, resistance etc to reach global minima faster than standard SGD optimizer.

Random Forest Algorithm: The model is implemented using sklearn inbuild function for random forest. I varied number of trees and their max depth as per dataset to get better result. For classification i used sklearn.ensemble.RandomForestClassifier and for regression i used sklearn.ensemble.RandomForestRegressor. I used R2 score on testing dataset to evaluate performance of my model.

Ensemble of Model: I combined both models to get final result. I found that random forest model outperform fnn so i decided to take weighted average of both prediction giving high weightage to random forest model.

# Convolutional Neural Network
Initially I build my own cnn model ( 3 * ( cnn layer -> relu -> max pool ) + fnn ) but its performance is not good. Eventually I shifted to transfer learning and after experimenting with many models i finally found resnet18 with best output.
Here I faced many challenges. Eventually I used a unique method to train models. I train the model till it become slight overfitted then I do some transformation on dataset like horizontal flipping, rotation by 60*, etc and train on new dataset till it overfits on them. I repeated it with many transformation and eventually it show better performance on original dataset (on testing at end) 

# Datasets and their Preprocessing
Datasets: https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset
https://www.kaggle.com/datasets/khanalkiran/price-receivedton-for-crops-for-10-year 
https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield 
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset 
https://www.kaggle.com/datasets/gdabhishek/fertilizer-prediction 
https://www.kaggle.com/datasets/pusainstitute/cropirrigationscheduling 
https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification 
https://www.kaggle.com/datasets/tarundalal/dangerous-insects-dataset/data  .

Dataset Processing: I created a automatic python code that do data processing for csv file. It find all the labels in it and do label encoding and return me a dictionary of label encoding for future reference.
For image and cnn I did many processing including transforming images, merging two sources to increase dataset etc. 

# Implementation Workflow
I mainly created three models (regression, classification and cnn ) and then modified them to various work by training them on different datasets and doing hyperparameter tuning.

# How to use it
It is made using flask and aimed to be more userfriendly. To use it navigate to combined agriculture solution > User Interface folder and in this folder select the model which you want to use. 
Then open the respective folder in command shell and run following commands:
$env:FLASK_APP = 'app.py' (to set variable of flask app to app.py which is the flask app file)
flask run (to run flask app)
This will run the flask app and you can view it on local browser by opening localhost in it
