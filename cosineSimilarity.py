import math
from collections import Counter


def build_sentence_vector(sentence1, sentence2):
    iterable1 = sentence1.split()
    iterable2 = sentence2.split()


    counter1 = Counter(iterable1)
    counter2 = Counter(iterable2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2


def cosineSimilarity(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2) )
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)


fuzzy_summary_file = open("docs/summaryObasanjo.txt","r")
fuzzy_content = fuzzy_summary_file.read()

arewa_summary_file = open("docs/arewaSummary.txt","r")
arewa_content = arewa_summary_file.read()

sun_summary_file = open("docs/sun_summary.txt","r")
sun_content = sun_summary_file.read()

original_article_file = open("doc.txt","r")
original_content = original_article_file.read()





v1, v2 = build_sentence_vector(fuzzy_content, arewa_content)
res1 = cosineSimilarity(v1, v2)

print("Cosine similarity between fuzzy & Arewa content : ", res1)

v3, v4 = build_sentence_vector(fuzzy_content, sun_content)
res2 = cosineSimilarity(v3, v4)
print("Cosine similarity between fuzzy & sun content : ", res2)


v5, v6 = build_sentence_vector(arewa_content, sun_content)
res3 = cosineSimilarity(v5, v6)
print("Cosine similarity arewa content & sun content : ", res3)

v7, v8 = build_sentence_vector(original_content, fuzzy_content)
res4 = cosineSimilarity(v7, v8)
print("Cosine similarity between original content & fuzzy result : ", res4)

v9, v10 = build_sentence_vector(original_content, sun_content)
res5 = cosineSimilarity(v9, v10)
print("Cosine similarity between original content & sun content : ", res5)

v11, v12 = build_sentence_vector(original_content, arewa_content)
res6 = cosineSimilarity(v11, v12)
print("Cosine similarity original content & arewa content : ", res6)
