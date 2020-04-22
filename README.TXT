# Final_Proj

To run program make sure you have all the packages installed on your computer: 
- json, sklearn, string, numpy, re, pandas
- line 13, change the name of your training json file (with labels) to match the read in json file
  -- EX: ours is called 'data_train.json'
- line 56, change the name of the json file you want test (without labels) to match the read in json file
  -- EX: ours is called 'data_test_wo_label.json'
  
Run using Python3

OpenJSON():
  This function will simply open a json file and convert directly into a pandas dataframe using the pandas library
  
CleanText(Dataframe):
  This function takes in a pandas dataframe and removes urls, newlines, and digits, then returns edited Dataframe
  
CalculateAccuracy(predicted_label, true_label)):
  This function takes in our predicted labels and actual labels as numpy arrays. Each corresponding element is compared and returns the accuracy.
 
 main():
  This function is where we train on the reviews from 'data_train.json' and edited to train on the incoming test file 'data_test_wo_label.json'. We use CountVectorizer from sklearn library to remove stop words, accented words, 
  transform text to lowercase, tokenize and train on that text.
