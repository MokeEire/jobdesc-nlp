# Resources:
#   -   Text Analysis in Python: https://medium.com/towards-artificial-intelligence/text-mining-in-python-steps-and-examples-78b3f8fd913b
#   -   Text Analysis overview: https://monkeylearn.com/text-analysis/
#   -   https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f
#   -   https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386
#   -
#   -   https://www.machinelearningplus.com/nlp/lemmatization-examples-python/#wordnetlemmatizerwithappropriatepostag


# Importing necessary libraries
import pandas as pd
import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import re
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 5)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 300)


## Functions -----------------------------------------------------------------------------------------------------------
# Text cleaning function, shamelessly stolen from: https://github.com/datanizing/reddit-selfposts-blog
def clean(s):
    s = re.sub(r'((\n)|(\r))', " ", s) # replace newline characters and \r whatever it is (another line break char?) with spaces
    s = re.sub(r'\r(?=[A-Z].)', "", s) # remove \r when it is next to a word
    s = re.sub(r'/', " ", s) # replace forward slashes with spaces
    s = re.sub(r'\-', " ", s) # replace dashes with spaces (I will be forever cursed for not accounting for the em dash)
    no_punct = "".join([c.lower() for c in s if c not in string.punctuation]) # remove punctuation

    return no_punct

# Function to remove stopwords from a list of words
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

# Function to lemmatize strings from a list of words
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
# Lemmatizing reduces words to their root form
def word_lemmatizer(text):
    lem_text = [lemmatizer.lemmatize(word = i, pos= get_wordnet_pos(i)) for i in text]

    return (lem_text)

# Function for finding the most commonly used words in job desc
def top_words(cleaned_desc, n = 3):
    # Count word freq
    freq = pd.Series(cleaned_desc.split()).value_counts()

    # Select top 3 words
    top_n = freq[:n].index.to_list()
    return(top_n)

# Function for creating masked wordcloud
# Found here: https://amueller.github.io/word_cloud/auto_examples/masked.html
def makeImage(text, img):
    # Need to get a mask image
    mask = np.array(Image.open(img))

    wc = WordCloud(background_color="white", max_words=1000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()



# Read in data
jobapps_file = 'data/jobapps.csv'
jobapps_df = pd.read_csv(jobapps_file)

# Isolate job description text
job_desc = jobapps_df[['description']]

# Cleaning
job_desc_clean = job_desc.assign(desc_clean = job_desc.description.apply(clean))

## Tokenizing
# Instantiate Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# Add tokenized column
job_desc_clean['desc_tokenized'] = job_desc_clean.desc_clean.apply(lambda x: tokenizer.tokenize(x))

# Remove stop words
job_desc_clean['desc_clean_nostop'] = job_desc_clean['desc_clean'].apply(lambda x: " ".join(x for x in x.split() if x not in stopwords.words('english')))
# Add tokenized column w/o stop words
job_desc_clean['desc_tokenized_nostop'] = job_desc_clean.desc_tokenized.apply(lambda x: remove_stopwords(x))

# finding the frequency distinct in the tokens
# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability import FreqDist
job_desc_freq = [FreqDist(desc) for desc in job_desc_clean.desc_tokenized_nostop]
job_desc_freq

# To find the frequency of top 10 words
desc_most_common = [fdist.most_common(10) for fdist in job_desc_freq]
desc_most_common



## Lemmatization
# Importing Lemmatizer library from nltk

lemmatizer = WordNetLemmatizer()


# Add lemmatized column
job_desc_clean['desc_lemmatized'] = job_desc_clean.desc_tokenized_nostop.apply(lambda x: word_lemmatizer(x))

# Count word frequencies
freq = pd.Series(' '.join(job_desc_clean['desc_clean']).split()).value_counts()[:10]

# Count word freq w/o stop words
freq_nostop = pd.Series(' '.join(job_desc_clean['desc_clean_nostop']).split()).value_counts()

freq_lemma = pd.Series(' '.join(job_desc_clean['desc_lemmatized'].apply(lambda x: ' '.join(x))).split()).value_counts()

# Select top words for each
job_desc_clean['top_words'] = job_desc_clean.desc_clean_nostop.apply(lambda x: top_words(x, 5))
job_desc_clean['top_words_lemma'] = job_desc_clean.desc_lemmatized.apply(lambda x: top_words(' '.join(x), 5))

# Join back to the job data to see each position's most common terms
jobapps_df.iloc[:,0:2].join(job_desc_clean.top_words)





# Wordcloud
## Convert word frequencies to dictionary
dict_for_wc = freq_lemma.to_dict()

# plot the WordCloud image
makeImage(dict_for_wc, 'charlie_black.png')






