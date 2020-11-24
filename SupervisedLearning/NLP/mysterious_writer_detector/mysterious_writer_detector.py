# In this program, we use writing samples of three writers and using Bag of Words, decide to which of them a
# new text belongs.

# Importing text samples. Note that samples are string originally, then we split them by ". ". Therefore, lists
# of string parts are imported below.

from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setting up the combined list of writing samples.

writers_docs = goldman_docs + henson_docs + wu_docs

# Setting up labels for your three writers.

writers_labels = [1] * 154 + [2] * 141 + [3] * 166

# print(goldman_docs[5])
# print(henson_docs[4])
# print(wu_docs[4])

# This is the sample text for which we want to detect the most likely writer among the three.

mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

# Creating a Bag of Words vectorizer

bow_vectorizer = CountVectorizer()

# Defining a vector of writers' works, that is, strings transformed into numeric values.

writers_vectors = bow_vectorizer.fit_transform(writers_docs)

# Define a vector of the new text. 

mystery_vector = bow_vectorizer.transform([mystery_postcard])

# Here we define a Naive Bayes classifer. We have so far transformed the samples and the new text into numeric values.
# Now we can use a classifier to find out how likely is for the new text to belong to each writer.

friends_classifier = MultinomialNB()

# Training the classifier.

friends_classifier.fit(writers_vectors, writers_labels)

# Predicting the text.

predictions = friends_classifier.predict(mystery_vector)
predictions_proba = friends_classifier.predict_proba(mystery_vector)

mystery_friend = predictions[0]

print("Probability for the text to belong to 1: {}\nProbability for the text to belong to 2: {}\nProbability for the text to belong to 3: {}\n".format(round((predictions_proba[0][0]*100), 3), round((predictions_proba[0][1]*100), 3), round((predictions_proba[0][2]*100), 3)))
print("The postcard was from {}!".format(mystery_friend))

print('\nThanks for reviewing')

# Thanks for reviewing
