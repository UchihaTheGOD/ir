from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk
from nltk.corpus import stopwords
import numpy as np
from numpy.linalg import norm

train_set = ["The sky is blue.", "The sun is bright."]
test_set = ["The sun in the sky is bright."]

nltk.download('stopwords')
stopWords = stopwords.words('english')

vectorizer = CountVectorizer(stop_words=stopWords)
transformer = TfidfTransformer()

trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
testVectorizerArray = vectorizer.transform(test_set).toarray()

print('Fit Vectorizer to train set:', trainVectorizerArray)
print('Transform Vectorizer to test set:', testVectorizerArray)

cx = lambda a, b: round(np.inner(a, b) / (norm(a) * norm(b)), 3)

for train_vec in trainVectorizerArray:
    for test_vec in testVectorizerArray:
        cosine = cx(train_vec, test_vec)
        print("Cosine Similarity:", cosine)

transformer.fit(trainVectorizerArray)
print("TF-IDF (Train):", transformer.transform(trainVectorizerArray).toarray())

transformer.fit(testVectorizerArray)
tfidf = transformer.transform(testVectorizerArray)
print("TF-IDF (Test):", tfidf.todense())
