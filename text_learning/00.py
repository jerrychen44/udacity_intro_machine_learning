import os,sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def create_bag_of_words():

    ##################
    # Example: create the bag of words
    ###################

    vectorizer=CountVectorizer()


    string1="hi Katie the self driving car will be late Best Sebastian"
    string2="Hi Sebastian the machine learning class will be great great great Best Katie"
    string3="Hi Katie the machine learning class will be most excellent"

    email_list=[string1,string2,string3]


    # fit first, then transformer
    # fit() will get the all token sets
    bag_of_words=vectorizer.fit(email_list)
    print(dir(bag_of_words))
    print(vectorizer.vocabulary_)
    '''
    {'driving': 4, 'self': 14, 'excellent': 5, 'great': 6, 'be': 0, 'late': 9, 'best': 1, 'hi': 7, 'katie': 8,
    'most': 12, 'learning': 10, 'car': 2, 'machine': 11, 'will': 16, 'sebastian': 13, 'class': 3, 'the': 15}
    '''
    #transform(), help  you to count the frequence of each string
    bag_of_words=vectorizer.transform(email_list)

    print(bag_of_words)
    '''
      (text data index, bag of word column index)  frequence
      (1, 0)    1
      (1, 1)    1
      (1, 3)    1
      (1, 6)    3
      (1, 7)    1
      (1, 8)    1
      (1, 10)    1
      (1, 11)    1
      (1, 13)    1
      (1, 15)    1
      (1, 16)    1
      '''

    # get the "great" column-th number in bag of word
    print(vectorizer.vocabulary_.get("great"))
    '''6'''
    # which means the bag_of_words table likes..table

    '''
            be  car  class  driving  excellent  great.....string1
    string2  1  1    0      1             0       1
    '''

    print(len(vectorizer.get_feature_names()))
    print(vectorizer.get_feature_names())
    print(vectorizer.get_feature_names()[6])
    '''
    17
    ['be', 'best', 'car', 'class', 'driving', 'excellent', 'great', 'hi', 'katie', 'late', 'learning', 'machine', 'most', 'sebastian', 'self', 'the', 'will']
    great
    '''

    return 0



def remove_stop_words():

    ######################
    # deal with low information: remove the stop words
    #######################
    '''
    # if you use nltk first , need to download all-corpora first
    import nltk
    nltk.download()
    '''
    from nltk.corpus import stopwords
    stop_word_set = stopwords.words("english")
    print(len(stop_word_set),stop_word_set[0:10])
    '''153 ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']'''
    return 0

def stemming_example():

    ##################
    # Example: Stemming (you should do the stemming before create the bag of the words)
    ###################


    #######################
    # Stemming the similary word.
    #################
    # example
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")

    print(stemmer.stem("responsiveness"))
    '''respons'''
    print(stemmer.stem("responsivity"))
    '''respons'''
    print(stemmer.stem("unresponsive"))
    '''unrespons'''
    return 0


def test1():
    #stemming_example()
    #remove_stop_words()

    #create_bag_of_words()

    # quick note
    # ti = term frequency =(like) bag of words (use the number as the weight)
    # idf= invers document frequency




    return 0


def test2():




    string1="kitty is cat"
    string2="kitty is not dog"
    string3="hello kitty is not cat"

    input_list=[string1,string2,string3]




    vectorizer=CountVectorizer()

    # fit() will get the all token sets
    bag_of_words=vectorizer.fit(input_list)
    print((bag_of_words))

    print(vectorizer.vocabulary_)


    bag_of_words_transed=vectorizer.transform(input_list)
    print(bag_of_words_transed)



    tfidtransfor=TfidfTransformer()
    tfidf_fit=tfidtransfor.fit(bag_of_words_transed)
    print(tfidf_fit)
    tfidf_traned=tfidf_fit.transform(bag_of_words_transed)
    print(tfidf_traned)

    ###################
    # if you have new input
    ###################
    print("=====================Get New Input")
    new_txet=["jerry love dog more then kitty, and he will say hello to you"]
    #use old vocabulary dic
    vectorizer_new=CountVectorizer(vocabulary=vectorizer.vocabulary_)
    #bag_of_words_new=vectorizer_new.fit(new_txet)
    #print((bag_of_words_new))
    #print(bag_of_words_new.vocabulary_)

    bag_of_words_transed_new=vectorizer_new.transform(new_txet)
    print(bag_of_words_transed_new)
    print(bag_of_words_transed_new.indices)
    print(bag_of_words_transed_new.indptr)
    print(vectorizer_new.vocabulary_)
    for item in bag_of_words_transed_new.indices:
        print(vectorizer_new.get_feature_names()[item])
        #print(vectorizer_new.get_feature_names()[4])
    # Done , reuse the old bag_of_word to encode the new input



    tfidtransfor_new=TfidfTransformer()
    #use old bag of words to encode a new input
    tfidf_fit_new=tfidtransfor_new.fit(bag_of_words_transed)
    #print(dir(tfidf_fit_new))
    #print(tfidf_fit_new.get_params())
    tfidf_traned_new=tfidf_fit_new.transform(bag_of_words_transed_new)
    #print(dir(tfidf_traned_new))
    print(tfidf_traned_new.toarray()[0])
    return 0

test2()








