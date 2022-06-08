import spacy


class Vocabulary(): 
    def __init__(self, min_freq, vocab_size):
        # Initialize special tokens
        self.itos = {0:"<pad>", 1:"<sos>", 2:"<eos>", 3:"<unk>"}
        self.stoi = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
        
        self.min_freq = min_freq                                     # Minimum frequency needed to include a token in the vocabulary
        self.vocab_size = vocab_size                                 # Vocabulary maximum size
        self.nlp = spacy.load("en_core_web_sm")                      # spacy english tokenizer  
        
    def __len__(self):
        """ return the length of the vocabulary """
        return len(self.itos) 
        
    def tokenizer_eng(self, text):
        """ tokenize and lowercase sentences """
        return [tok.text.lower() for tok in self.nlp.tokenizer(str(text))] 
    
    def build_vocabulary(self, sentence_list):
        """
        Build dataset vocabulary from input texts by adding token to vocabulary diictionnaries
        input : sentence_list - Text List from training data
        """
        # keep track of word frequency (necessary to filter with min_freq)
        frequencies = {} 
        idx = len(self.itos)
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1     
                else:
                    frequencies[word] += 1
                    
                # add word to vocabulary when its frequency == min_freq
                if frequencies[word] == self.min_freq:  
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
                # break when vocabulary reaches vocab_size
                if len(self.itos) == self.vocab_size: 
                    break
    
    def numericalize(self, text):
        """transform text into list of position index in vocabulary"""
        tokenized_text = self.tokenizer_eng(text)
        
        # Numericalize if word in vocabulary, otherwise set to <unk> token
        return [self.stoi[token] if token in self.stoi else self.stoi["<unk>"] for token in tokenized_text] 
    
    def invert_numericalize_extended(self, oovs, oovs_updated, idx_list):
        """ 
        Return the words corresponding to indices from our vocabulary 
        Input : oovs - list of out of vocabulary words for individual texts
                oovs_updated - list of out of vocabulary words cumulated for updates
                idx_list - numeric text - list of indexed words
        Output : list of tokens/words 
        """
        
        # Getting extended vocabulary for out of vocabulary words
        itos_extended = {key:value for (key,value) in enumerate(self.extend_vocabulary(oovs))}
        itos_extended_updated = {key:value for (key,value) in enumerate(self.extend_vocabulary(oovs_updated))}
        
        # 
        itos_list = []
        for idx in idx_list:
            if idx in itos_extended:
                itos_list.append(itos_extended[idx])
            else:
                itos_list.append(itos_extended_updated[idx])
            
        return itos_list
    
    def oov_ids_extended_vocab(self, text):
        """ 
        Allow the copy mechanism to point to oov words positionally 
        Input : text - list of text to transform into list of index
        Output : ids - list of words index positions in extended vocabulary (OOVs words have an index created)
                 oovs - list of Out of vocabulary words 
        """
        ids = []
        oovs = []
        
        #Going through text
        for token in self.tokenizer_eng(text):
            #Index tokens
            token_id = self.numericalize(token)[0]
            unk_id = self.stoi["<unk>"]
            #if token is OOV we add token to oovs list and create an additional index in vocabulary and index token in list
            if token_id == unk_id:
                if token not in oovs:
                    oovs.append(token)
                ids.append(len(self.itos) + oovs.index(token))
            else:
                ids.append(token_id)
                
        return ids, oovs
    
    def extend_vocabulary(self, oovs):
        """
        Extending vocabulary by creating additional ids in dicitonary if words in not in original dictionnary
        """
        extended_vocab = list(self.itos.values()) + list(oovs)
        
        return extended_vocab
