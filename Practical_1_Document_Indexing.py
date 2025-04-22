import nltk
from nltk.corpus import stopwords

# Sample documents
document1 = "The quick brown fox jumped over the lazy dog"
document2 = "The lazy dog slept in the sun"

# Download stopwords
nltk.download('stopwords')
stopWords = stopwords.words('english')

# Tokenize and normalize
tokens1 = document1.lower().split()
tokens2 = document2.lower().split()

# Unique terms
terms = list(set(tokens1 + tokens2))

# Create inverted index and term frequencies
inverted_index = {}
occ_num_doc1 = {}
occ_num_doc2 = {}

for term in terms:
    if term in stopWords:
        continue
    documents = []
    if term in tokens1:
        documents.append("Document 1")
        occ_num_doc1[term] = tokens1.count(term)
    if term in tokens2:
        documents.append("Document 2")
        occ_num_doc2[term] = tokens2.count(term)
    inverted_index[term] = documents

# Output the inverted index with term frequencies
for term, documents in inverted_index.items():
    print(term, "->", end=" ")
    for doc in documents:
        if doc == "Document 1":
            print(f"{doc} ({occ_num_doc1.get(term, 0)}),", end=" ")
        else:
            print(f"{doc} ({occ_num_doc2.get(term, 0)}),", end=" ")
    print()
