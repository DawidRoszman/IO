from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from text2emotion import get_emotion

text = []
with open("positive.txt", "r", encoding="utf8") as f:
    text.append(f.read())

with open("negative.txt", "r", encoding="utf8") as f:
    text.append(f.read())

sentences = []
for line in text:
    sentences.extend(tokenize.sent_tokenize(line))

for sentence in sentences:
    sid = SentimentIntensityAnalyzer()
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print("{0}: {1}, ".format(k, ss[k]), end="")
    print("\n")

for sentence in sentences:
    print(sentence)
    emotions = get_emotion(sentence)
    print("Emotions detected:")
    for emotion, score in emotions.items():
        print(f"{emotion}: {score:.2f}")
    print()
