# In this program, we use a Naive Bayes classifier to create a model for distinguishing emails based on their topics.

from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Loading the data. Note that emails are stored in a list called emails.data and labels can be found in the list emails.target. Labels are numbers, and their corresponding names can be found at emails.target_names.

emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'])

# Making the training and test sets.

train_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'train', shuffle = True, random_state = 108)

test_emails = fetch_20newsgroups(categories = ['rec.sport.baseball', 'rec.sport.hockey'], subset = 'test', shuffle = True, random_state = 108)

# Now we want to transform these emails into lists of words counts.

counter = CountVectorizer()
counter.fit(train_emails.data + test_emails.data)

train_counts = counter.transform(train_emails.data)
test_counts = counter.transform(test_emails.data)

# Alright. Now is the time to make and train a MultinomialNB classifier.

clf = MultinomialNB()
clf.fit(train_counts, train_emails.target)

print(clf.score(train_counts, train_emails.target))

# As of the 0.9974 accuracy, we can conclude that our model does a good job distinguishing between baseball and hockey emails. Note that, at the beginning, we chose two email datasets of baseball and hockey, each email in each category having its corresponding label.

print('\nThanks for reviewing')

# Thanks for reviewing
