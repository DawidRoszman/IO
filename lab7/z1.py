import string
from matplotlib import pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud

with open("article.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

word_tokens = word_tokenize(text)
print(f"Token words: {len(word_tokens)}")

stop_words = stopwords.words("english")
filtered_tokens = [w for w in word_tokens if not w.lower() in stop_words]
print(f"Filtered Tokens: {len(filtered_tokens)}")

stop_words.extend(string.punctuation)
stop_words.extend(["’", "—", "“", "”"])
refiltered_tokens = [w for w in word_tokens if not w.lower() in stop_words]
print(f"Refiltered Tokens: {len(refiltered_tokens)}")

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w, pos="a") for w in refiltered_tokens]
print(f"Lemmatized Tokens: {len(lemmatized_tokens)}")

stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(w) for w in refiltered_tokens]
print(f"Stemmed Tokens: {len(stemmed_tokens)}")

stemmed_lemmed_tokens = [lemmatizer.lemmatize(w, pos="a") for w in stemmed_tokens]
lemmed_stemmed_tokens = [stemmer.stem(w) for w in lemmatized_tokens]

wordcloud = WordCloud(
    width=800, height=400, background_color="white", collocations=False
).generate(" ".join(lemmatized_tokens))
wordcloud.to_file("wordcloud-lemma.png")

word_freq = {}
for word in lemmatized_tokens:
    word_freq[word] = word_freq.get(word, 0) + 1
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
words, frequencies = zip(*top_words)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words")
plt.xticks(rotation=45)
plt.savefig("word-bag-plot-lemma.png")

wordcloud = WordCloud(
    width=800, height=400, background_color="white", collocations=False
).generate(" ".join(stemmed_tokens))
wordcloud.to_file("wordcloud-stem.png")

word_freq = {}
for word in stemmed_tokens:
    word_freq[word] = word_freq.get(word, 0) + 1
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
words, frequencies = zip(*top_words)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words")
plt.xticks(rotation=45)
plt.savefig("word-bag-plot-stemmed.png")

wordcloud = WordCloud(
    width=800, height=400, background_color="white", collocations=False
).generate(" ".join(stemmed_lemmed_tokens))
wordcloud.to_file("wordcloud-stemmed-lemmed.png")

word_freq = {}
for word in stemmed_lemmed_tokens:
    word_freq[word] = word_freq.get(word, 0) + 1
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
words, frequencies = zip(*top_words)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words")
plt.xticks(rotation=45)
plt.savefig("word-bag-plot-stemmed-lemmed.png")

wordcloud = WordCloud(
    width=800, height=400, background_color="white", collocations=False
).generate(" ".join(stemmed_lemmed_tokens))
wordcloud.to_file("wordcloud-lemmed-stemmed.png")

word_freq = {}
for word in stemmed_lemmed_tokens:
    word_freq[word] = word_freq.get(word, 0) + 1
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
words, frequencies = zip(*top_words)

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 10 Most Frequent Words")
plt.xticks(rotation=45)
plt.savefig("word-bag-plot-lemmed-stemmed.png")
