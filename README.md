Language Models Using N-gram
As with all statistical models, the true data generating process is unknown to us, so all we can do is estimate the probabilities of sentences. For example, one might estimate the probability of a sentence as simply the product of the empirical probabilities (i.e., the number of times a word is observed in a dataset divided by the number of words in that dataset). In the above example, we may have:

Using this simple statistic equation, I will create a model that generates human-understandable sentence, N-gram model.

Definition of N-gram
N-gram is a sequence of the N-words. a 2-gram (bigram) is a two word sequence of words like "give me" or "broken vessels" and a 3-gram (trigram) is a three word-sequence of words such as "give me money" or "need broken vessels".

With the equation given above, I will estimate the probability of the last word of an n-gram given the previous words and use it to generate sentence.

Project Catalog
Part 1: Preparing the Corpus
Part 2: Tokenizing the Corpus
Part 3: Creating N-gram Model
Part 4: Testing N-gram Model
Import Libraries
import pandas as pd
import numpy as np
import os
import re
import requests
import time
Part 1: Preparing the Corpus

I'll use the requests module to download the "Plain Text UTF-8" text of a public domain book from Project Gutenberg and prepare it for analysis in later questions. For instance, the book Beowulf's "Plain Text UTF-8" URL is here, which can be accessed by clicking the "Plain Text UTF-8" link here.

# Function to get the content of a book from url through HTTP request
def get_book(url):
    text = requests.get(url).text
    title = re.findall(r'Title: ([A-Za-z ]+)', text)[0].upper()
    pattern = r'\*{3} START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK [\r\n \w]+ \*{3}((?s).*)\*{3} END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK [\r\n \w]+ \*{3}'
    content = re.findall(pattern, text)[0]
    return re.sub(r'\r\n', '\n', content)
# Testing the function with a book called The Great Gatsby
great_gatsby = get_book('https://www.gutenberg.org/cache/epub/64317/pg64317.txt')

great_gatsby[:1000]
'\n\n\t\t\t   The Great Gatsby\n\t\t\t\t  by\n\t\t\t F. Scott Fitzgerald\n\n\n                           Table of Contents\n\nI\nII\nIII\nIV\nV\nVI\nVII\nVIII\nIX\n\n\n                              Once again\n                                  to\n                                 Zelda\n\n  Then wear the gold hat, if that will move her;\n  If you can bounce high, bounce for her too,\n  Till she cry “Lover, gold-hatted, high-bouncing lover,\n  I must have you!”\n\n  Thomas Parke d’Invilliers\n\n\n                                  I\n\nIn my younger and more vulnerable years my father gave me some advice\nthat I’ve been turning over in my mind ever since.\n\n“Whenever you feel like criticizing anyone,” he told me, “just\nremember that all the people in this world haven’t had the advantages\nthat you’ve had.”\n\nHe didn’t say any more, but we’ve always been unusually communicative\nin a reserved way, and I understood that he meant a great deal more\nthan that. In consequence, I’m inclined to reserve all judgements, a\nhabit that has opened up'
Part 2: Tokenizing the Corpus

Now, tokenize the text by implementing the function tokenize, which takes in a string, book_string, and returns a list of the tokens (words, numbers, and all punctuation) in the book such that:

The start of every paragraph is represented in the list with the single character '\x02' (standing for START).
The end of every paragraph is represented in the list with the single character '\x03' (standing for STOP).
Tokens include no whitespace.
Two or more newlines count as a paragraph break, and whitespace (e.g. multiple newlines) between two paragraphs of text do not appear as tokens.
All punctuation marks count as tokens, even if they are uncommon (e.g. '@', '+', and '%' are all valid tokens).
For example, consider the following excerpt. (The first sentence is at the end of a larger paragraph, and the second sentence is at the start of a longer paragraph.)

...
My phone's dead.

I didn't get your call!!
...
Tokenizes to:

[...
'My', 'phone', "'", 's', 'dead', '.', '\x03', '\x02', 'I', 'didn', "'", 't', 'get', 'your', 'call', '!', '!'
...]
# Tokenize the given book text
def tokenize(book_string):
    book_string = '\x02'+book_string.strip()+'\x03'
    book_string = re.sub('^\n{2,}', '\x02', book_string)
    book_string = re.sub('\n{2,}$', '\x03', book_string)
    book_string = re.sub('\n{2,}', '\x03\x02', book_string)
    pattern = r'[A-Za-z]+|[^\s\d\w]|\x03|\x02'
    return re.findall(pattern, book_string)
# Testing on The Great Gatsby
tokenized = tokenize(great_gatsby)
np.array(tokenized)[:100]
array(['\x02', 'The', 'Great', 'Gatsby', 'by', 'F', '.', 'Scott',
       'Fitzgerald', '\x03', '\x02', 'Table', 'of', 'Contents', '\x03',
       '\x02', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX',
       '\x03', '\x02', 'Once', 'again', 'to', 'Zelda', '\x03', '\x02',
       'Then', 'wear', 'the', 'gold', 'hat', ',', 'if', 'that', 'will',
       'move', 'her', ';', 'If', 'you', 'can', 'bounce', 'high', ',',
       'bounce', 'for', 'her', 'too', ',', 'Till', 'she', 'cry', '“',
       'Lover', ',', 'gold', '-', 'hatted', ',', 'high', '-', 'bouncing',
       'lover', ',', 'I', 'must', 'have', 'you', '!', '”', '\x03', '\x02',
       'Thomas', 'Parke', 'd', '’', 'Invilliers', '\x03', '\x02', 'I',
       '\x03', '\x02', 'In', 'my', 'younger', 'and', 'more', 'vulnerable',
       'years', 'my', 'father', 'gave', 'me'], dtype='<U17')
Part 3: Creating N-Gram Model
Sentences are built from tokens, and the likelihood that a token occurs where it does depends on the tokens before it. This points to using conditional probability to compute 
. That is, we can write:

Using chain rule for probabilities.

Example:

'when I drink Coke I smile'
The probability that it occurs, according the the chain rule, is

That is, the probability that the sentence occurs is the product of the probability that each subsequent token follows the tokens that came before. For example, the probability 
 is likely pretty high, as Coke is something that you drink. The probability 
 is likely low, because pizza is not something that you drink.

Side Note 1: Uniform Language Models
A uniform language model is one in which each unique token is equally likely to appear in any position, unconditional of any other information. In other words, in a uniform language model, the probability assigned to each token is 1 over the total number of unique tokens in the corpus.

>>> corpus = 'when I eat pizza, I smile, but when I drink Coke, my stomach hurts'
>>> tokenize(corpus)
['\x02', 'when', 'I', 'eat', 'pizza', ',', 'I', 'smile', ',', 'but', 'when', 'I', 'drink', 'Coke', ',', 'my', 'stomach', 'hurts', '\x03']
The example corpus above has 14 unique tokens. This means that I'd have 
 
, 
 
, and so on. Specifically, in this example, the Series that train returns should contain the following values:

Token	Probability
'\x02'	
 
'when'	
 
'I'	
 
'eat'	
 
'pizza'	
 
','	
 
'smile'	
 
'but'	
 
'drink'	
 
'Coke'	
 
'my'	
 
'stomach'	
 
'hurts'	
 
'\x03'	
 
Unifrom Class:
The __init__ constructor: when you instantiate an LM object, I pass in the "training corpus" on which my model will be trained. The train method uses that data to create a model which is saved in the mdl attribute.
The train method takes in a list of tokens and outputs a language model. This language model is represented as a Series, whose index consists of tokens and whose values are the probabilities that the tokens occur.
The probability method takes in a sequence of tokens and returns the probability that this sequence occurs under the language model.
The sample method takes in a positive integer M and generates a string made up of M tokens using the language model. This method generates random sentences!
# Uniform Language Model
class UniformLM(object):


    def __init__(self, tokens):

        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        unique_tokens = pd.Series(tokens).unique()
        return pd.Series(np.full(len(unique_tokens), 1/len(unique_tokens)), index=unique_tokens)
    
    def probability(self, words):
        try:
            words = list(words)
            prob = np.prod(self.mdl.loc[words])
        except KeyError:
            prob = 0
        return prob 
        
    def sample(self, M):

        return ' '.join(self.mdl.sample(M, replace=True).index)
# Testing Uniform Language Model on The Great Gatsby
uniform = UniformLM(tokenized)
uniform.sample(100)
'wealth rendered orchestras rest interminable Can property insisted cry boarder avoiding peculiarly Were Aren indian real Invilliers twinkle aunts opening sleeves Love settle breeze capes or divan closer Maintenon extraordinary smart sport drew sister measure movements beds hallway facet roughly introduce action in Read loneliness successful ring Wondering limit choking headed divine stored M spread thinks Nothing boisterously bastards squawk fragilely hue murmur carved reminded dirty blankly objects unusual gone special overhanging causing tattoo herding married sneakers thinning hesitated fingers Bird comprehended bosom Dewars wing Henry Ulysses Doubtless exciting effect undeserted waltz waiter bid which Buchanan general Daisy Meyer extreme'
Side Note 2: Uni-Gram Model
A unigram language model is one in which the probability assigned to a token is equal to the proportion of tokens in the corpus that are equal to said token. That is, the probability distribution associated with a unigram language model is just the empirical distribution of tokens in the corpus.

Let's understand how probabilities are assigned to tokens using our example corpus from before.

>>> corpus = 'when I eat pizza, I smile, but when I drink Coke, my stomach hurts'
>>> tokenize(corpus)
['\x02', 'when', 'I', 'eat', 'pizza', ',', 'I', 'smile', ',', 'but', 'when', 'I', 'drink', 'Coke', ',', 'my', 'stomach', 'hurts', '\x03']
Here, there are 19 total tokens. 3 of them are equal to 'I', so 
 
. Here, the Series that train returns should contain the following values:

Token	Probability
'\x02'	
 
'when'	
 
'I'	
 
'eat'	
 
'pizza'	
 
','	
 
'smile'	
 
'but'	
 
'drink'	
 
'Coke'	
 
'my'	
 
'stomach'	
 
'hurts'	
 
'\x03'	
 
As before, the probability method should take in a tuple and return its probability, using the probabilities stored in mdl. For instance, suppose the input tuple is ('when', 'I', 'drink', 'Coke', 'I', 'smile'). Then,

 
 
 
 
 
 
# Creates Uni-Gram Language Model
class UnigramLM(object):
    
    def __init__(self, tokens):

        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        tokens = pd.Series(tokens)
        return tokens.value_counts().apply(lambda x: x/len(tokens))

    def probability(self, words):
        try:
            words = list(words)
            prob = np.prod(self.mdl.loc[words])
        except KeyError:
            prob = 0
        return prob
        
    def sample(self, M):
        return ' '.join(self.mdl.sample(M, replace=True).index)
# Testing Uni-Gram Language Model on The Great Gatsby

unigram = UnigramLM(tokenized)
unigram.sample(100)
'skins bounded now eyelashes straggled As temper walked Across before wisp polite keyed Gravely wanting understanding paper invent McKee cheekbone Montana remains amount Hasn heavens Whereupon grotesque page gardener spoke fianc some peered Very attired wasn cordials valued corridors Lewis desk unobtrusively gets spires burst adventurous is becomes feigned firm dream pungent unreal switch uncertainty delighted clever clog suffered enter kissed actually cries away cluster world brushed neared Jaqueline incomparable alive mysteries demoniac tells kind without funny bust several parents scales eastward twin crouching under pouring inquest Port trying cigarettes abandon guts lieutenant enjoined invariably clenched torpedoes fondled substitute formless'
Creating N-Gram Model
The N-Gram language model relies on the assumption that only nearby tokens matter. Specifically, it assumes that the probability that a token occurs depends only on the previous 
 tokens, rather than all previous tokens. That is:

In an N-Gram language model, there is a hyperparameter that we get to choose when creating the model, 
. For any 
, the resulting N-Gram model looks at the previous 
 tokens when computing probabilities. (Note that the unigram model you built in Question 4 is really an N-Gram model with 
, since it looked at 0 previous tokens when computing probabilities.)

Both when working with a training corpus and when implementing the probability method to compute the probabilities of other sentences, I use "chunks" of 
 tokens at a time.

Definition: The N-Grams of a text are a list of tuples containing sliding windows of length 
.

For instance, the trigrams in the sentence 'when I drink Coke I smile' are:

[('when', 'I', 'drink'), ('I', 'drink', 'Coke'), ('drink', 'Coke', 'I'), ('Coke', 'I', 'smile')]

Computing N-Gram Probabilities
Notice in our trigram model above, I computed 
 as being the product of several conditional probabilities. These conditional probabilities are the result of training our N-Gram model on a training corpus.

To train an N-Gram model, I compute a conditional probability for every 
-token sequence in the corpus. For instance, for every 3-token sequence 
, I must compute 
. To do so, I use:

 
where 
 is the number of occurrences of the trigram sequence 
 in the training corpus and 
 is the number of occurrences of the bigram sequence 
 in the training corpus. (Technical note: the probabilities that I compute using the ratios of counts are estimates of the true conditional probabilities of N-Grams in the population of corpuses from which our corpus was drawn.)

In general, for any 
, conditional probabilities are computed by dividing the counts of N-Grams by the counts of the (N-1)-Grams they follow.


The NGramLM Class
The NGramLM class contains a few extra methods and attributes beyond those of UniformLM and UnigramLM:

Instantiating NGramLM requires both a list of tokens and a positive integer N, specifying the N in N-grams. This parameter is stored in an attribute N.
The NGramLM class has a method create_ngrams that takes in a list of tokens and returns a list of N-Grams (recall from above, an N-Gram is a tuple of length N). This list of N-Grams is then passed to the train method to train the N-Gram model.
While the train method still creates a language model (in this case, an N-Gram model) and stores it in the mdl attribute, this model is most naturally stored as a DataFrame. This DataFrame will have three columns:
'ngram', containing the N-Grams found in the text.
'n1gram', containing the (N-1)-Grams upon which the N-Grams in ngram are built.
'prob', containing the probabilities of each N-Gram in ngram.
The NGramLM class has an attribute prev_mdl that stores an (N-1)-Gram language model over the same corpus (which in turn will store an (N-2)-Gram language model over the same corpus, and so on). This is necessary to compute the probability that a word occurs at the start of a text.
N-Gram LM consists of probabilities of the form

Which can be estimated by:

 
for every N-Gram that occurs in the corpus. To illustrate, consider again the following example corpus:

>>> corpus = 'when I eat pizza, I smile, but when I drink Coke, my stomach hurts'
>>> tokens = tokenize(corpus)
>>> tokens
['\x02', 'when', 'I', 'eat', 'pizza', ',', 'I', 'smile', ',', 'but', 'when', 'I', 'drink', 'Coke', ',', 'my', 'stomach', 'hurts', '\x03']
>>> pizza_model = NGrams(3, tokens)
Here, pizza_model.train must compute 
, 
, 
, and so on, until 
.

To compute 
, I find the number of occurrences of 'when I eat' in the training corpus, and divide it by the number of occurrences of 'when I' in the training corpus. 'when I eat' occurred exactly once in the training corpus, while 'when I' occurred twice, so,

 
 
To store the conditional probabilities of all N-Grams, I use a DataFrame with three columns, like so:

ngram	n1gram	prob
0	(when, I, drink)	(when, I)	0.5
1	(when, I, eat)	(when, I)	0.5
2	(,, but, when)	(,, but)	1.0
3	(,, I, smile)	(,, I)	1.0
4	(I, smile, ,)	(I, smile)	1.0
5	(,, my, stomach)	(,, my)	1.0
6	(but, when, I)	(but, when)	1.0
7	(, when, I)	(, when)	1.0
8	(stomach, hurts, )	(stomach, hurts)	1.0
9	(Coke, ,, my)	(Coke, ,)	1.0
10	(eat, pizza, ,)	(eat, pizza)	1.0
11	(I, drink, Coke)	(I, drink)	1.0
12	(my, stomach, hurts)	(my, stomach)	1.0
13	(pizza, ,, I)	(pizza, ,)	1.0
14	(I, eat, pizza)	(I, eat)	1.0
15	(drink, Coke, ,)	(drink, Coke)	1.0
16	(smile, ,, but)	(smile, ,)	1.0
The row at position 1 in the above table shows that the probability of the trigram ('when', 'I', 'eat') conditioned on the bigram ('when', 'I') is 0.5, as we computed above. Note that many of the above conditional probabilities are equal to 1 because many trigrams and their corresponding bigrams each appeared only once, and 
 
. Note that '\x02' and '\x03' appear as spaces above, such as in row 7.

class NGramLM(object):
    
    def __init__(self, N, tokens):
        self.N = N
        ngrams = self.create_ngrams(tokens)
        self.tok = tokens
        self.ngrams = ngrams
        self.mdl = self.train(ngrams)
        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        result = []
        right_pointer = self.N
        left_pointer = 0
        
        while right_pointer <= len(tokens):
            result.append(tuple(tokens[left_pointer:right_pointer]))
            left_pointer += 1
            right_pointer += 1
        
        return result
    
    def create_ngrams_N(self, n, tokens):
        res = []
        right = n
        left = 0

        while right <= len(tokens):
            res.append(tuple(tokens[left:right]))
            left += 1
            right += 1
        
        return res

    def train(self, ngrams):
        ser_ngram = pd.Series(ngrams, name="ngram")
        n1gram_create = self.create_ngrams_N(self.N-1, self.tok)
        ser_n1gram = pd.Series(n1gram_create, name="n1gram")
        merge_ngram_n1gram = pd.merge(ser_ngram, ser_n1gram, left_index=True, right_index=True)
        denom_col = merge_ngram_n1gram.groupby("n1gram").transform('count')
        numer_col = merge_ngram_n1gram.groupby("ngram").transform('count')
        merge_ngram_n1gram["numerator"] = numer_col
        merge_ngram_n1gram["denominator"] = denom_col
        merge_ngram_n1gram["prob"] = merge_ngram_n1gram["numerator"] / merge_ngram_n1gram["denominator"]
        
        return merge_ngram_n1gram.drop(columns=["numerator", "denominator"]).drop_duplicates(keep='first')
    
    def probability(self, words):
        input_words = ' '.join(words)
        token_words = ' '.join(self.tok)

        if input_words not in token_words:
            return 0
        
        final_prob = 1
        
        initial_prob = []
        for i in range(self.N-1, 0, -1):
            if i == 1:
                initial_prob.append(words[0:i][0])
            else:
                initial_prob.append(words[0:i])
        
        list_grams = self.create_ngrams(words)
        
        for gram in list_grams: 
            if gram not in list(self.mdl['ngram'].values):
                return 0
            else:
                final_prob *= self.mdl[self.mdl['ngram'] == gram]['prob'].iloc[0]
                
        current = self
        
        for gram in initial_prob:
            current = current.prev_mdl
            current_table = current.mdl
            if isinstance(gram, str):
                final_prob *= pd.DataFrame(current_table).loc[gram].iloc[0]
            else:
                final_prob *= current_table[current_table["ngram"] == gram]["prob"].iloc[0]
       
        return final_prob
        
    def sample(self, M):
        sample_words = ['\x02']
        
        for i in range(1, self.N):
            current = self.N - i
            prev = self
            
            for i in range(current, 1, -1):
                prev = prev.prev_mdl
            
            probability = prev.mdl[np.where(prev.mdl['n1gram'].apply(str) == str(tuple(sample_words)), True, False)].drop_duplicates(keep='first')
            
            if probability.shape[0] == 0 or probability.shape[1] == 0:
                sample_words.append('\x03')
            else:
                smalls = np.random.choice(probability['ngram'], p=probability['prob'])
                sample_words.append(smalls[-1])
               
        for j in range(self.N, M+1):
            probability = prev.mdl[np.where(prev.mdl['n1gram'].apply(str) == str(tuple(sample_words[-self.N+1:])), True, False)].drop_duplicates(keep='first')
            
            if probability.shape[0] == 0 or probability.shape[1] == 0:
                sample_words.append('\x03')
            else:
                words = np.random.choice(probability['ngram'], p=probability['prob'])
                sample_words.append(np.random.choice(probability['ngram'], p=probability['prob'])[-1])
               
        if sample_words[-1] != '\x03':
            sample_words[-1] = '\x03'
            
        return ' '.join(sample_words)
Part 4: Testing N-Gram Model

#Initialize N-Gram Language Model on The Great Gatsby
ngram = NGramLM(5, tokenized)
# In List Form
ngram_list = ngram.create_ngrams(tokenized)
ngram_list[:10]
[('\x02', 'The', 'Great', 'Gatsby', 'by'),
 ('The', 'Great', 'Gatsby', 'by', 'F'),
 ('Great', 'Gatsby', 'by', 'F', '.'),
 ('Gatsby', 'by', 'F', '.', 'Scott'),
 ('by', 'F', '.', 'Scott', 'Fitzgerald'),
 ('F', '.', 'Scott', 'Fitzgerald', '\x03'),
 ('.', 'Scott', 'Fitzgerald', '\x03', '\x02'),
 ('Scott', 'Fitzgerald', '\x03', '\x02', 'Table'),
 ('Fitzgerald', '\x03', '\x02', 'Table', 'of'),
 ('\x03', '\x02', 'Table', 'of', 'Contents')]
# In Dataframe Form
ngram_df = ngram.mdl
ngram_df.head()
ngram	n1gram	prob
0	(, The, Great, Gatsby, by)	(, The, Great, Gatsby)	1.0
1	(The, Great, Gatsby, by, F)	(The, Great, Gatsby, by)	1.0
2	(Great, Gatsby, by, F, .)	(Great, Gatsby, by, F)	1.0
3	(Gatsby, by, F, ., Scott)	(Gatsby, by, F, .)	1.0
4	(by, F, ., Scott, Fitzgerald)	(by, F, ., Scott)	1.0
# Genereate Sample Sentence using N-Gram Language Model on The Great Gatsby
ngram_sample = ngram.sample(200) 
ngram_sample
