import re
import nltk
from nltk.corpus import stopwords


class TextPreprocessing():
    def __init__(self, remove_stopwords, remove_punctuation):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
    
    def to_lowercase(self, text):
        return text.lower() 
    
    def replace_double_quote(self, match):
        text = match.group()
        
        return text.replace("20", "")

    def regex_pattern_removal(self, x): 
        
        """
        Data cleaning and curating of TREC dataset.
        Input : x - raw text
        Output : sentece - cleaned text
        """
        
        sentence = self.to_lowercase(x)
        sentence = re.sub("\d{10}", "doc_id", sentence) # remove doc_id prefix
        sentence = re.sub("[a-zA-Z0-9]{32}$", "doc_id", sentence) # remove doc_id suffix
        sentence = re.sub("[a-zA-Z0-9]{32}-\d{1}$", "doc_id", sentence) # remove doc_id suffix
        sentence = re.sub("[a-zA-Z0-9]{32}-\d{2}", "doc_id", sentence) # remove doc_id suffix
        regex = re.compile("20[a-zA-Z]+")
        sentence = regex.sub(self.replace_double_quote, sentence) # remove "20" before words
        sentence = re.sub("&amp;", "", sentence) # remove &amp; 
        sentence = re.sub("&amp", "", sentence) # remove &amp 
        sentence = re.sub("nbsp", "", sentence) # remove nbsp
        sentence = re.sub("blq", "", sentence) # remove blq
        sentence = re.sub("[^\x00-\x7f]", "", sentence) # remove hex character
        sentence = re.sub("(\\t)", "", sentence) # remove escape characters
        sentence = re.sub("(\\n)", "", sentence) # remove escape characters
        sentence = re.sub("(__+)", "", sentence) # remove underscore if it occurs more than one time consecutively
        sentence = re.sub("(--+)", "", sentence) # remove dash if it occurs more than one time consecutively
        sentence = re.sub("(\.\.+)", "", sentence) # remove . if it occurs more than one time consecutively
        sentence = re.sub("((https*:\/*)([^\/\s]+))(.[^\s]+)", "", sentence) # remove urls
        sentence = re.sub("http\S+", "", sentence) # remove urls
        sentence = re.sub("www\S+", "", sentence) # remove urls
        sentence = re.sub("http", "", sentence) # remove http 
        sentence = re.sub("b&gt", "", sentence) # remove b&gt
        
        #deleting stopwords from sentence
        if self.remove_stopwords:
            sentence = [i for i in sentence.split() if i not in stopwords.words('english')]
            sentence = " ".join(sentence)
        
        #deleting punctuation from sentence
        if self.remove_punctuation:
            sentence = re.sub("[^\w\s]", "", sentence)
            sentence = re.sub("  ", " ", sentence)
        
        return sentence