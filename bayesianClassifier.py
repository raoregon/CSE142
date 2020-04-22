import pandas as pd
import numpy as np
import re, string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def Open_JSON():
    print("\n\n\n#######################################################################")
    print("\t\t\t Opening data_train.json\t")
    print("#######################################################################\n\n\n")
    return pd.read_json(r'data_train.json')

def Clean_Text(DataFrame):
    print("\n\n\n#######################################################################")
    print("\t\t\t Cleaning Data-newlines,URLs,transforming accented words\t")
    print("#######################################################################\n\n\n")
    # convert to all lowercase
    DataFrame = DataFrame.applymap(lambda s: s.lower() if type(s) == str else s)
    """
    remove stop words
    """
    # remove strings that begin with http
    DataFrame['text'] = DataFrame['text'].str.replace(r'http\S+', '')
    # remove punctuation
    DataFrame['text'] = DataFrame['text'].str.replace(r'[^\w\s]+', '')
    # remove numbers
    DataFrame['text'] = DataFrame['text'].str.replace(r'\d+', '')
    # remove newlines
    DataFrame['text'] = DataFrame['text'].str.replace(r'\n', '')

    return DataFrame

def Remove_Unnecessary_Data(DataFrame):
    print("\n\n\n#######################################################################")
    print("\t\t\t Removing Unnecessary Data\t")
    print("#######################################################################\n\n\n")

    DataFrame = DataFrame[['text', 'stars']]
    return DataFrame

def main():

    # read in json data into a pandas Dataset
    Data = Open_JSON()

    # strip out the other columns so its only 'text' and 'stars'
    Data = Remove_Unnecessary_Data(Data)

    # preprocess the data, first drop down to all lowercase, then remove punctuation
    # then remove numbers
    Data = Clean_Text(Data)

    # create a secondary array that we will test after training
    csvTest = Data['text'].to_numpy()

    # these are the stars that we will train
    """
    args = [1.0, 2.0, 3.0, 4.0, 5.0]
    """
    ###### REMOVE THIS LINE WHEN TRAINING ALL 5
    args = [4.0]

    # an array that we will keep the tested dataframes for each star
    finalDataFrames = []

    for stars in args:
        print("\n\n\n#######################################################################")
        print("\t\tProcessing reviews with ", stars, " stars\t")
        print("#######################################################################\n\n\n")

        # isolate the labeled star from the rest of the Dataset
        # EX: if processing 1.0 star, we set currentStar to all of the 1 star reviews
        # then we set restStarVS to all of the other star'd reviews
        currentStar = Data.loc[Data['stars'] == stars, 'text'].copy().reset_index(drop=True)
        restStarVS = Data.loc[Data['stars'] != stars, 'text'].copy().reset_index(drop=True)

        # add a classifier label, 1 for 1 star reviews, 0 for rest star reviews
        currentStar = pd.concat([pd.DataFrame(currentStar), pd.DataFrame(np.ones(currentStar.shape),
                                                                         columns=['class'])], 1)
        restStarVS = pd.concat([pd.DataFrame(restStarVS), pd.DataFrame(np.zeros(
            restStarVS.shape), columns=['class'])], 1)

        # remove reviews that are more than 500 characters long, this helps with training accuracy
        long_reviews = currentStar.loc[currentStar['text'].str.len() > 500].index
        currentStar.drop(long_reviews, inplace=True)

        long_reviews = restStarVS.loc[restStarVS['text'].str.len() > 500].index
        restStarVS.drop(long_reviews, inplace=True)

        # randomize the dataset, then reduce 'restStarVS' to the same size as currentStar so that
        # our classifier has an even distribution of data to work with
        np.random.seed(42)
        rand = np.random.permutation(restStarVS.shape[0])
        restStarVS = restStarVS.iloc[rand[:currentStar.shape[0]]].reset_index(drop=True)

        # concatenate positive and negative reviews into one Dataframe
        DataNew = pd.concat([restStarVS, currentStar]).sample(frac=1).reset_index(drop=True)

        # split data into train and test set, however, this test set is based off of the already
        # labled reviews so that we can measure accuracy
        X_train, X_test, y_train, y_test = train_test_split(DataNew['text'].values,
                                                            DataNew['class'].values, test_size=0.2,
                                                            random_state=42,shuffle=True)

        # tokenize the 'text'
        print("\n\n\n#######################################################################")
        print("\t\tTokenizing data\t")
        print("#######################################################################\n\n\n")

        re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        def tokenize(s): return re_tok.sub(r' \1 ', s).split()

        # creating bag of words
        vect = CountVectorizer(tokenizer=tokenize)

        # training/vectorizing the data: our classifier checks which words are used in each review.
        #  It then sets up a set of words that it knows are commonly found in 'currentStar' reviews.
        # EX: if 1.0 is the current star, it checks which words are most common in 1 star reviews
        # then it compares the current unknown label review and sees if the words are commonly found
        # If yes, then the review is labeled as a 1, if no, then a 0.
        tf_train = vect.fit_transform(X_train)
        tf_test = vect.transform(X_test)

        # Lastly, we vectorize the final test csv
        tf_csv_test = vect.transform(csvTest)

        # Multinomial naive Bayes implementation
        p = tf_train[y_train == 1].sum(0) + 1
        q = tf_train[y_train == 0].sum(0) + 1
        r = np.log((p / p.sum()) / (q / q.sum()))
        b = np.log(len(p) / len(q))

        pre_preds = tf_test @ r.T + b

        # our trained classifier: classifies each review
        preds = pre_preds.T > 0

        # using our trained classifier, we classify the final test csv
        csv_preds = tf_csv_test @ r.T + b
        predsFinal = csv_preds.T > 0

        # the array is a 1 x n dimensional array, to print to a csv, we Transpose it to a n x 1
        # array
        csvArray = predsFinal.T

        # check the accuracy of the current star training
        acc = (preds == y_test).mean()
        print("Accuracy: {:.4f}".format(acc))

        # append the final test csv results to a list
        finalDataFrames.append(csvArray)

    # convert all of our "final test csv" results to integers (instead of true/false)

    #REMOVE THIS LINE TO TRAIN ALL 5
    fourS = finalDataFrames[0]
    """
    fourS = finalDataFrames[3]
    """
    fourS = fourS.astype(int)
    """
    twoS = finalDataFrames[1]
    twoS = twoS.astype(int)
    threeS = finalDataFrames[2]
    threeS = threeS.astype(int)
    oneS = finalDataFrames[0]
    oneS = oneS.astype(int)
    fiveS = finalDataFrames[4]
    fiveS = fiveS.astype(int)
    """

    # create the final array that will be printed to our csv "predictionsOutput"
    testArray= []


    # take the worst accuracy classifier and set its values first in the array
    # then we change the values using our next best classifier until finally our best classifier
    # changes the results in the array
    for index,stars in enumerate(fourS):
        if stars == [0]:
            testArray.append(0)
        else:
            testArray.append(4)
    """
    for index,stars in enumerate(threeS):
        if stars == [1]:
            testArray[index] = 3
    for index,stars in enumerate(twoS):
        if stars == [1]:
            testArray[index] = 2
    for index,stars in enumerate(fiveS):
        if stars == [1]:
            testArray[index] = 5
    for index,stars in enumerate(oneS):
        if stars == [1]:
            testArray[index] = 1
    """
    # print the array to a csv.
    df = pd.DataFrame(testArray, columns=["Predictions"])
    df.to_csv('predictionsOutput.csv', index=False)


main()