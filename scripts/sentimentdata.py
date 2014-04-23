import re
import numpy as np

single_terms = []
scores = []

avg_scores = []
only_root = re.compile(r'([0-4] [A-Za-z,.-][A-Za-z,.-]*)')


# Collect all 1-25 sentiment scores for single words
with open('../stanfordSentimentTreebankRaw/sentlex_exp12.txt') as f:
    for line in f:
        terms = line.strip().lower().split(',')
        tokens = terms[1].split(' ')
        if len(tokens) == 1:
            single_terms.append((terms[0], tokens[0]))

with open('../stanfordSentimentTreebankRaw/rawscores_exp12.txt') as f:
    for line in f:
        terms = line.strip().split(',')
        scores.append(terms[1:])


# Find all single words (leaf nodes) from training tree and their 0-4 averaged sentiment scores
with open('../trees/train.txt') as f:
    for line in f:
        matches = re.findall(only_root, line)
        avg_scores.append(matches)

# with open('../trees/dev.txt') as f:
#     for line in f:
#         matches = re.findall(only_root, line)
#         avg_scores.append(matches)

# with open('../trees/test.txt') as f:
#     for line in f:
#         matches = re.findall(only_root, line)
#         avg_scores.append(matches)

avg_scores = [score for ascores in avg_scores for score in ascores]
avg_scores = list(set(avg_scores))

# Map each leaf-node word from training tree with its average score
sentiments = {} 
for score in avg_scores:
    score = score.split(" ")
    sentiments[score[1].lower()] = score[0]


data_terms = [] # All terms found in the training tree
# Write all terms found in both the treebank and training tree to 'present' and 
# all remaining terms to 'missing'
with open('../data/present', 'w') as f_out1, open('../data/missing', 'w') as f_out2:
    for term in single_terms:
        # print term[1], scores[int(term[0])], sentiments.get(term[1], 'None')
        word = term[1]
        sentiment_scores = scores[int(term[0])]
        avg_score = sentiments.get(word, 'None')
        if avg_score != 'None':
            data_terms.append((term[0], term[1]))
            f_out1.write(' '.join(map(str, [word, sentiment_scores, avg_score, '\n'])))
        else:
            f_out2.write(' '.join(map(str, [word, sentiment_scores, '\n'])))

# Write all words with high variance sentiments to use for word-sense disambiguation
with open('../data/datafile', 'w') as f_out:
    for term in data_terms:
        word = term[1]
        sentiment_scores = map(int, scores[int(term[0])])
        var = np.var(sentiment_scores)
        if var > 2:
            f_out.write(' '.join(map(str, [term[1], sentiment_scores, var, '\n'])))
