
# coding: utf-8

## Cosine Similarity for Vector Space Models and TF-IDF

###### Example script adapted from Pyevolve: http://pyevolve.sourceforge.net/wordpress/?s=cosine+similarity

#### Define set of example documents:

# In[2]:

documents = (
"A man lives in Montana",
"Interesting people live in Montana",
"Montana has mountains",
"The man walked his dog near the mountain"
)


#### Instantiate the Sklearn TF-IDF Vectorizer and transform our documents into the TF-IDF matrix:

# In[4]:

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print tfidf_matrix.shape


#### We now have the TF-IDF matrix (tfidf_matrix) for each document (the number of rows of the matrix) with 15 tf-idf terms (the number of columns from the matrix), we can calculate the Cosine Similarity between the first document (“A man lives in Montana”) with each of the other documents of the set:

# In[3]:

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)


#### The tfidf_matrix[0:1] is the Scipy operation to get the first row of the sparse matrix and the resulting array is the Cosine Similarity between the first document with all documents in the set. Note that the first value of the array is 1.0 because it is the Cosine Similarity between the first document with itself. Also note that due to the presence of similar words on the second document (“Interesting people live in Montana”), it achieved a better score.

#### Continue: Check the angle between the first and third documents:

# In[7]:

import math
# This was already calculated on the previous step, so we just use the value
cos_sim = 0.16128176
angle_in_radians = math.acos(cos_sim)
print math.degrees(angle_in_radians)


#### The angle of ~80.7 radians is the angle between the first and the third document of our document set.
