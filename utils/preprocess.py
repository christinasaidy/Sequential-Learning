import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch 
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk
from nltk.tokenize import word_tokenize

def format_text_regex(text):

    # ^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%.\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%\+.~#?&\/=]*)$

    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE) #clean all URLs
    text = re.sub(r'\<a href', ' ', text) #clean html style URL
    text = re.sub(r'&amp;', '', text) #remove &amp; chars
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text) #remove special characters
    text = re.sub(r'<br />', ' ', text) #remove html style <br>
    text = re.sub(r'\'', ' ', text)
    return text

def cloud(word, title, colormap='viridis', background_color='white'):
    wordcloud = WordCloud(width=800, height=400, background_color=background_color, colormap=colormap, random_state=42).generate(word)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()


def remove_stopwords(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    return " ".join(text)

contractions = {
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def decontract_words(text):
    text = text.split()
    new_text = []
    for word in text:
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    return " ".join(new_text)


def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in tokens]

def encode_and_pad(text, word_to_idx, max_length):
    encoded = [word_to_idx.get(word, 1) for word in text]  # 1 unknown token
    return encoded + [0] * (max_length - len(encoded))  # 0 padding


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:         
        return wordnet.NOUN

def lemmatize_passage(text):
    lemmatizer = WordNetLemmatizer()
    # tokens = nltk.WordPunctTokenizer().tokenize(text)
    pos_tags = pos_tag(text, tagset=None, lang='eng')
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return lemmatized_words

def tokenize(text):
    return word_tokenize(text.lower())

def vectorize(Text, max_length, word_to_idx):
    vocab = {word for word in Text}  
    encoded = [word_to_idx.get(word, 1) for word in Text]  # 1 unknown token
    return encoded + [0] * (max_length - len(encoded))  # 0 padding


def preprocess_text(text):
    text = format_text_regex(text)
    text = decontract_words(text)
    text = remove_stopwords(text)
    tokens = tokenize(text)
    tokens = stem_words(tokens)
    return tokens