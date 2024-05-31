import numpy as np
import string


# Method 1
m = int(input())

allSentences = list()
allWords = list()

for i in range(m):
    sentence = input()
    sentence = sentence.lower().strip()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentenceWords = sentence.split()
    allSentences.append(sentenceWords)
    for word in sentenceWords:
        if allWords.count(word) == 0:
            allWords.append(word)
n = len(allWords)

querySentence = input()
querySentence = querySentence.lower().strip()
querySentence = querySentence.translate(str.maketrans("", "", string.punctuation))
querySentenceWords = querySentence.split()

wordsFrequencyMatrix = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        wordsFrequencyMatrix[i, j] = allSentences[i].count(allWords[j]) / len(allSentences[i])

wordsCommonnessVector = np.zeros(n)
for j in range(n):
    numOfSentences = 0
    hasTheWord = False
    for i in range(m):
        if wordsFrequencyMatrix[i, j] > 0:
            hasTheWord = True
            numOfSentences += 1
            wordsCommonnessVector[j] = np.log(m / numOfSentences)

sentencesRepresentationMatrix = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        sentencesRepresentationMatrix[i][j] = wordsFrequencyMatrix[i][j] * wordsCommonnessVector[j]

querySentenceFrequencyVector = np.zeros(n)
for j in range(n):
    querySentenceFrequencyVector[j] = querySentenceWords.count(allWords[j]) / len(querySentenceWords)

querySentenceRepresentationVector = querySentenceFrequencyVector * wordsCommonnessVector

maxDotProduct = -10
mostSimilarSentenceIndex = 0
for i in range(m):
    if np.dot(sentencesRepresentationMatrix[i], querySentenceRepresentationVector) > maxDotProduct:
        maxDotProduct = np.dot(sentencesRepresentationMatrix[i], querySentenceRepresentationVector)
        mostSimilarSentenceIndex = i

print(mostSimilarSentenceIndex + 1)


# Method 2
a, b, d = map(int, input().split())
n = a + b

words = []
sentences = []

for _ in range(n):
    sentence = input().split()
    sentences.append(sentence)
    words.extend(sentence)

words = np.unique(words)

matrix_A = np.empty((n, words.size))

for i in range(n):
    for j in range(words.size):
        matrix_A[i, j] = sentences[i].count(words[j])

U, s, V = np.linalg.svd(matrix_A, full_matrices=False)

approx_A = U[:, :d] @ np.diag(s[:d]) @ V[:d, :]

approx_A_a = approx_A[:a]
approx_A_b = approx_A[a:]

dot_products = np.dot(approx_A_b, approx_A_a.T)
norms_a = np.linalg.norm(approx_A_a, axis=1)
norms_b = np.linalg.norm(approx_A_b, axis=1)

pairwise_similarities = dot_products / np.outer(norms_b, norms_a)

best_match_indices = np.argmax(pairwise_similarities, axis=1)

for index in best_match_indices:
    print(index)
