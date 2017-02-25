# corpus
doc = [
    'this app is good!',
    'i love your app',
    'i love this. good!',
    'This app is horrible.',
    'I feel tired use your app.',
    'this is my enemy'

]

# class
label = [
    'positive',
    'positive',
    'positive',
    'negative',
    'negative',
    'negative'
]

def line(x):
    print(x*100)


from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from term_weighting import TfRf_Vectorizer

#instansiasi objek Scikit-learn
vectorizerInt = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(norm=None)
tfrf_vectorizer = TfRf_Vectorizer(norm=None)

bow = vectorizerInt.fit_transform(doc)

print "Feature"
line("-")
print(vectorizerInt.get_feature_names())
print "\nFeature Size : ", len(vectorizerInt.get_feature_names())


line("=")
print "BOW representation (Frequency):"
line("-")
pprint(bow.toarray())

line("=")
bow = tfidf_vectorizer.fit_transform(doc)
print "BOW representation (TF-IDF):"
line("-")
pprint(bow.toarray())

line("=")
bow = tfrf_vectorizer.fit_transform(doc, label)
print "BOW representation (TFRF):"
line("-")
pprint(bow.toarray())