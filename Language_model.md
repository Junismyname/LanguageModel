# Language Models Using N-gram

As with all statistical models, the true data generating process is unknown to us, so all we can do is **estimate** the probabilities of sentences. For example, one might estimate the probability of a sentence as simply the product of the empirical probabilities (i.e., the number of times a word is observed in a dataset divided by the number of words in that dataset). In the above example, we may have:

$$P(\text{when I drink Coke I smile}) = P(\text{when}) \cdot P(\text{I}) \cdot P(\text{drink}) \cdot P(\text{Coke}) \cdot P(\text{I}) \cdot P(\text{smile})$$

Using this simple statistic equation, I will create a model that generates human-understandable sentence, N-gram model.

## Definition of N-gram
N-gram is a sequence of the N-words. a 2-gram (bigram) is a two word sequence of words like "give me" or "broken vessels" and a 3-gram (trigram) is a three word-sequence of words such as "give me money" or "need broken vessels". <br/>

With the equation given above, I will estimate the probability of the last word of an n-gram given the previous words and use it to generate sentence. 

## Project Catalog

- [Part 1: Preparing the Corpus](#part1)
- [Part 2: Tokenizing the Corpus](#part2)
- [Part 3: Creating N-gram Model](#part3)
- [Part 4: Testing N-gram Model](#part4)

## Import Libraries


```python
import pandas as pd
import numpy as np
import os
import re
import requests
import time
```

## Part 1: Preparing the Corpus 
<a name='part1'></a>

I'll use the `requests` module to download the "Plain Text UTF-8" text of a public domain book from [Project Gutenberg](https://www.gutenberg.org/) and prepare it for analysis in later questions. For instance, the book Beowulf's "Plain Text UTF-8" URL is [here](https://www.gutenberg.org/ebooks/16328.txt.utf-8), which can be accessed by clicking the "Plain Text UTF-8" link [here](https://www.gutenberg.org/ebooks/16328). 


```python
# Function to get the content of a book from url through HTTP request
def get_book(url):
    text = requests.get(url).text
    title = re.findall(r'Title: ([A-Za-z ]+)', text)[0].upper()
    pattern = r'\*{3} START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK [\r\n \w]+ \*{3}((?s).*)\*{3} END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK [\r\n \w]+ \*{3}'
    content = re.findall(pattern, text)[0]
    return re.sub(r'\r\n', '\n', content)
```


```python
# Testing the function with a book called The Great Gatsby
great_gatsby = get_book('https://www.gutenberg.org/cache/epub/64317/pg64317.txt')

great_gatsby[:1000]
```

    C:\Users\lucas\AppData\Local\Temp\ipykernel_11192\3267087960.py:6: DeprecationWarning: Flags not at the start of the expression '\\*{3} START OF (?:TH' (truncated) but at position 69
      content = re.findall(pattern, text)[0]
    




    '\n\n\t\t\t   The Great Gatsby\n\t\t\t\t  by\n\t\t\t F. Scott Fitzgerald\n\n\n                           Table of Contents\n\nI\nII\nIII\nIV\nV\nVI\nVII\nVIII\nIX\n\n\n                              Once again\n                                  to\n                                 Zelda\n\n  Then wear the gold hat, if that will move her;\n  If you can bounce high, bounce for her too,\n  Till she cry “Lover, gold-hatted, high-bouncing lover,\n  I must have you!”\n\n  Thomas Parke d’Invilliers\n\n\n                                  I\n\nIn my younger and more vulnerable years my father gave me some advice\nthat I’ve been turning over in my mind ever since.\n\n“Whenever you feel like criticizing anyone,” he told me, “just\nremember that all the people in this world haven’t had the advantages\nthat you’ve had.”\n\nHe didn’t say any more, but we’ve always been unusually communicative\nin a reserved way, and I understood that he meant a great deal more\nthan that. In consequence, I’m inclined to reserve all judgements, a\nhabit that has opened up'



## Part 2: Tokenizing the Corpus
<a name='part2'></a>

Now, **tokenize** the text by implementing the function `tokenize`, which takes in a string, `book_string`, and returns a **list of the tokens** (words, numbers, and all punctuation) in the book such that:

* The start of every paragraph is represented in the list with the single character `'\x02'` (standing for START).
* The end of every paragraph is represented in the list with the single character `'\x03'` (standing for STOP).
* Tokens include *no* whitespace.
* Two or more newlines count as a paragraph break, and whitespace (e.g. multiple newlines) between two paragraphs of text do not appear as tokens.
* All punctuation marks count as tokens, even if they are uncommon (e.g. `'@'`, `'+'`, and `'%'` are all valid tokens).

For example, consider the following excerpt. (The first sentence is at the end of a larger paragraph, and the second sentence is at the start of a longer paragraph.)
```
...
My phone's dead.

I didn't get your call!!
...
```
Tokenizes to:
```py
[...
'My', 'phone', "'", 's', 'dead', '.', '\x03', '\x02', 'I', 'didn', "'", 't', 'get', 'your', 'call', '!', '!'
...]
```


```python
# Tokenize the given book text
def tokenize(book_string):
    book_string = '\x02'+book_string.strip()+'\x03'
    book_string = re.sub('^\n{2,}', '\x02', book_string)
    book_string = re.sub('\n{2,}$', '\x03', book_string)
    book_string = re.sub('\n{2,}', '\x03\x02', book_string)
    pattern = r'[A-Za-z]+|[^\s\d\w]|\x03|\x02'
    return re.findall(pattern, book_string)
```


```python
# Testing on The Great Gatsby
tokenized = tokenize(great_gatsby)
np.array(tokenized)[:100]
```




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



## Part 3: Creating N-Gram Model
<a name='part3'></a>
Sentences are built from tokens, and the likelihood that a token occurs where it does depends on the tokens before it. This points to using **conditional probability** to compute $P(w)$. That is, we can write:

$$
P(w) = P(w_1,\ldots,w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdot\ldots\cdot P(w_n|w_1,\ldots,w_{n-1})
$$  
Using **chain rule** for probabilities.

**Example:** 

<center><code>'when I drink Coke I smile'</code></center>
    
The probability that it occurs, according the the chain rule, is

$$
P(\text{when}) \cdot P(\text{I | when}) \cdot P(\text{drink | when I})\cdot P(\text{Coke | when I drink}) \cdot P(\text{I | when I drink Coke}) \cdot P(\text{smile | when I drink Coke I})
$$

That is, the probability that the sentence occurs is the product of the probability that each subsequent token follows the tokens that came before. For example, the probability $P(\text{Coke | when I drink})$ is likely pretty high, as Coke is something that you drink. The probability $P(\text{pizza | when I drink})$ is likely low, because pizza is not something that you drink.



### Side Note 1: Uniform Language Models

A uniform language model is one in which each **unique** token is equally likely to appear in any position, unconditional of any other information. In other words, in a uniform language model, the probability assigned to each token is **1 over the total number of unique tokens in the corpus**.


```py
>>> corpus = 'when I eat pizza, I smile, but when I drink Coke, my stomach hurts'
>>> tokenize(corpus)
['\x02', 'when', 'I', 'eat', 'pizza', ',', 'I', 'smile', ',', 'but', 'when', 'I', 'drink', 'Coke', ',', 'my', 'stomach', 'hurts', '\x03']
```

The example corpus above has 14 **unique** tokens. This means that I'd have $P(\text{\x02}) = \frac{1}{14}$, $P(\text{when}) = \frac{1}{14}$, and so on. Specifically, in this example, **the Series that `train` returns should contain the following values**:

| Token | Probability |
| --- | --- |
| `'\x02'` | $\frac{1}{14}$ |
| `'when'` | $\frac{1}{14}$ |
| `'I'` | $\frac{1}{14}$ |
| `'eat'` | $\frac{1}{14}$ |
| `'pizza'` | $\frac{1}{14}$ |
| `','` | $\frac{1}{14}$ |
| `'smile'` | $\frac{1}{14}$ |
| `'but'` | $\frac{1}{14}$ |
| `'drink'` | $\frac{1}{14}$ |
| `'Coke'` | $\frac{1}{14}$ |
| `'my'` | $\frac{1}{14}$ |
| `'stomach'` | $\frac{1}{14}$ |
| `'hurts'` | $\frac{1}{14}$ |
| `'\x03'` | $\frac{1}{14}$ |

#### Unifrom Class:

* The `__init__` constructor: when you instantiate an LM object, I pass in the "training corpus" on which my model will be trained. The `train` method uses that data to create a model which is saved in the `mdl` attribute. 
* The `train` method takes in a list of tokens and outputs a language model. **This language model is represented as a `Series`, whose index consists of tokens and whose values are the probabilities that the tokens occur.** 
* The `probability` method takes in a sequence of tokens and returns the probability that this sequence occurs under the language model.
* The `sample` method takes in a positive integer `M` and generates a string made up of `M` tokens using the language model. **This method generates random sentences!**


```python
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
```


```python
# Testing Uniform Language Model on The Great Gatsby
uniform = UniformLM(tokenized)
uniform.sample(100)
```




    'worse Son chilled scales Ulysses English darkening Chester races interested cleaned Ins mischief Library roofs lunch brush soothing immediate unquestionable See veranda For disposed traded expostulation blankly identical safer gaudiness promoted indirectly abroad mattered trolley thrown longer confessed department disposed leaned salmon mounted practice issued Jongs vain into inexperience pool rudely bench completed deplorably voice sofa anaemic along Cross lingeringly identical hauteur gaudily sumptuous glanced thunder hatted wanderers virgins \x03 husbands Lots triumphantly inappropriate pretences did immediate colourless Phone stored delight hundreds grotesque need created sounded fish barnyard confounding afternoons bumming seek weakness conscientious wed courses stained packed cars hoped'



### Side Note 2: Uni-Gram Model
A unigram language model is one in which the **probability assigned to a token is equal to the proportion of tokens in the corpus that are equal to said token**. That is, the probability distribution associated with a unigram language model is just the empirical distribution of tokens in the corpus. 

Let's understand how probabilities are assigned to tokens using our example corpus from before.

```py
>>> corpus = 'when I eat pizza, I smile, but when I drink Coke, my stomach hurts'
>>> tokenize(corpus)
['\x02', 'when', 'I', 'eat', 'pizza', ',', 'I', 'smile', ',', 'but', 'when', 'I', 'drink', 'Coke', ',', 'my', 'stomach', 'hurts', '\x03']
```

Here, there are 19 total tokens. 3 of them are equal to `'I'`, so $P(\text{I}) = \frac{3}{19}$. Here, the Series that `train` returns should contain the following values:

| Token | Probability |
| --- | --- |
| `'\x02'` | $\frac{1}{19}$ |
| `'when'` | $\frac{2}{19}$ |
| `'I'` | $\frac{3}{19}$ |
| `'eat'` | $\frac{1}{19}$ |
| `'pizza'` | $\frac{1}{19}$ |
| `','` | $\frac{3}{19}$ |
| `'smile'` | $\frac{1}{19}$ |
| `'but'` | $\frac{1}{19}$ |
| `'drink'` | $\frac{1}{19}$ |
| `'Coke'` | $\frac{1}{19}$ |
| `'my'` | $\frac{1}{19}$ |
| `'stomach'` | $\frac{1}{19}$ |
| `'hurts'` | $\frac{1}{19}$ |
| `'\x03'` | $\frac{1}{19}$ |

As before, the `probability` method should take in a tuple and return its probability, using the probabilities stored in `mdl`. For instance, suppose the input tuple is `('when', 'I', 'drink', 'Coke', 'I', 'smile')`. Then,

$$P(\text{when I drink Coke I smile}) = P(\text{when}) \cdot P(\text{I}) \cdot P(\text{drink}) \cdot P(\text{Coke}) \cdot P(\text{I}) \cdot P(\text{smile}) = \frac{2}{19} \cdot \frac{3}{19} \cdot \frac{1}{19} \cdot \frac{1}{19} \cdot \frac{3}{19} \cdot \frac{1}{19}$$


```python
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
```


```python
# Testing Uni-Gram Language Model on The Great Gatsby

unigram = UnigramLM(tokenized)
unigram.sample(100)
```




    'figured reposing heat sure Ninth looks Auto diamond might realized ways III foot tilt remaining potatoes roaring sixty transition Poles slippers divisional reminded slammed skipper mess Sorry eats alive deserted muttering scattered flash unexpectedly clicking hen rustling mischief sheets captain choice Columbus shown afford resolved effort dismissed inevitable chords higher See local sender Trinity unbroken vacant breaths favours enlarged across dusk Well ocean walls intimate Handsome cousin negroes bothering crimson both incalculable Compared gentlemanly fur lies cabinets drinking procession paternal key sure polish shuts scales flicked nostrils vivid regal per realized shops bored cr Finally emptiness such Hill magic relieving'



### Creating N-Gram Model

The N-Gram language model relies on the assumption that only nearby tokens matter. Specifically, it assumes that the probability that a token occurs depends only on the previous $N-1$ tokens, rather than all previous tokens. That is:

$$P(w_n|w_1,\ldots,w_{n-1}) = P(w_n|w_{n-(N-1)},\ldots,w_{n-1})$$

In an N-Gram language model, there is a hyperparameter that we get to choose when creating the model, $N$. For any $N$, the resulting N-Gram model looks at the previous $N-1$ tokens when computing probabilities. (Note that the unigram model you built in Question 4 is really an N-Gram model with $N=1$, since it looked at 0 previous tokens when computing probabilities.)

Both when working with a training corpus and when implementing the `probability` method to compute the probabilities of other sentences, I use  "chunks" of $N$ tokens at a time.

**Definition:** The **N-Grams of a text** are a list of tuples containing sliding windows of length $N$.

For instance, the trigrams in the sentence `'when I drink Coke I smile'` are:

```py
[('when', 'I', 'drink'), ('I', 'drink', 'Coke'), ('drink', 'Coke', 'I'), ('Coke', 'I', 'smile')]
```

<br>

#### Computing N-Gram Probabilities

Notice in our trigram model above, I computed $P(\text{when I drink Coke I smile})$ as being the product of several conditional probabilities. These conditional probabilities are the result of **training** our N-Gram model on a training corpus.

To train an N-Gram model, I compute a conditional probability for every $N$-token sequence in the corpus. For instance, for every 3-token sequence $w_1, w_2, w_3$, I must compute $P(w_3 | w_1, w_2)$. To do so, I use:

$$P(w_3 | w_1, w_2) = \frac{C(w_1, w_2, w_3)}{C(w_1, w_2)}$$

where $C(w_1, w_2, w_3)$ is the number of occurrences of the trigram sequence $w_1, w_2, w_3$ in the training corpus and $C(w_1, w_2)$ is the number of occurrences of the bigram sequence  $w_1, w_2$ in the training corpus. (Technical note: the probabilities that I compute using the ratios of counts are _estimates_ of the true conditional probabilities of N-Grams in the population of corpuses from which our corpus was drawn.)

In general, for any $N$, conditional probabilities are computed by dividing the counts of N-Grams by the counts of the (N-1)-Grams they follow. 

<br>

### The `NGramLM` Class

The `NGramLM` class contains a few extra methods and attributes beyond those of `UniformLM` and `UnigramLM`:

1. Instantiating `NGramLM` requires both a list of tokens and a positive integer `N`, specifying the N in N-grams. This parameter is stored in an attribute `N`.
1. The `NGramLM` class has a method `create_ngrams` that takes in a list of tokens and returns a list of N-Grams (recall from above, an N-Gram is a **tuple** of length N). This list of N-Grams is then passed to the `train` method to train the N-Gram model.
1. While the `train` method still creates a language model (in this case, an N-Gram model) and stores it in the `mdl` attribute, this model is most naturally stored as a DataFrame. This DataFrame will have three columns:
    - `'ngram'`, containing the N-Grams found in the text.
    - `'n1gram'`, containing the (N-1)-Grams upon which the N-Grams in `ngram` are built.
    - `'prob'`, containing the probabilities of each N-Gram in `ngram`.
1. The `NGramLM` class has an attribute `prev_mdl` that stores an (N-1)-Gram language model over the same corpus (which in turn will store an (N-2)-Gram language model over the same corpus, and so on). This is necessary to compute the probability that a word occurs at the start of a text. 

N-Gram LM consists of probabilities of the form

$$P(w_n|w_{n-(N-1)},\ldots,w_{n-1})$$

Which can be estimated by:  

$$\frac{C(w_{n-(N-1)}, w_{n-(N-2)}, \ldots, w_{n-1}, w_n)}{C(w_{n-(N-1)}, w_{n-(N-2)}, \ldots, w_{n-1})}$$

for every N-Gram that occurs in the corpus. To illustrate, consider again the following example corpus:

```py
>>> corpus = 'when I eat pizza, I smile, but when I drink Coke, my stomach hurts'
>>> tokens = tokenize(corpus)
>>> tokens
['\x02', 'when', 'I', 'eat', 'pizza', ',', 'I', 'smile', ',', 'but', 'when', 'I', 'drink', 'Coke', ',', 'my', 'stomach', 'hurts', '\x03']
>>> pizza_model = NGrams(3, tokens)
```

Here, `pizza_model.train` must compute $P(\text{I | \x02 when})$, $P(\text{eat | when I})$, $P(\text{pizza | I eat})$, and so on, until $P(\text{\x03 | stomach hurts})$.

To compute $P(\text{eat | when I})$, I find the number of occurrences of `'when I eat'` in the training corpus, and divide it by the number of occurrences of `'when I'` in the training corpus. `'when I eat'` occurred exactly once in the training corpus, while `'when I'` occurred twice, so,

$$P(\text{eat | when I}) = \frac{C(\text{when I eat})}{C(\text{when I})} = \frac{1}{2}$$

To store the conditional probabilities of all N-Grams, I use a DataFrame with three columns, like so:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ngram</th>
      <th>n1gram</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(when, I, drink)</td>
      <td>(when, I)</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(when, I, eat)</td>
      <td>(when, I)</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(,, but, when)</td>
      <td>(,, but)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(,, I, smile)</td>
      <td>(,, I)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(I, smile, ,)</td>
      <td>(I, smile)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(,, my, stomach)</td>
      <td>(,, my)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(but, when, I)</td>
      <td>(but, when)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(, when, I)</td>
      <td>(, when)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(stomach, hurts, )</td>
      <td>(stomach, hurts)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Coke, ,, my)</td>
      <td>(Coke, ,)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(eat, pizza, ,)</td>
      <td>(eat, pizza)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(I, drink, Coke)</td>
      <td>(I, drink)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(my, stomach, hurts)</td>
      <td>(my, stomach)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(pizza, ,, I)</td>
      <td>(pizza, ,)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(I, eat, pizza)</td>
      <td>(I, eat)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(drink, Coke, ,)</td>
      <td>(drink, Coke)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(smile, ,, but)</td>
      <td>(smile, ,)</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>

The row at position **1** in the above table shows that the probability of the trigram `('when', 'I', 'eat')` conditioned on the bigram `('when', 'I')` is 0.5, as we computed above. Note that many of the above conditional probabilities are equal to 1 because many trigrams and their corresponding bigrams each appeared only once, and $\frac{1}{1} = 1$. Note that `'\x02'` and `'\x03'` appear as spaces above, such as in row **7**.



```python
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
```

## Part 4: Testing N-Gram Model
<a name='part4'></a>


```python
#Initialize N-Gram Language Model on The Great Gatsby
ngram = NGramLM(5, tokenized)
```


```python
# In List Form
ngram_list = ngram.create_ngrams(tokenized)
ngram_list[:10]
```




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




```python
# In Dataframe Form
ngram_df = ngram.mdl
ngram_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ngram</th>
      <th>n1gram</th>
      <th>prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(, The, Great, Gatsby, by)</td>
      <td>(, The, Great, Gatsby)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(The, Great, Gatsby, by, F)</td>
      <td>(The, Great, Gatsby, by)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Great, Gatsby, by, F, .)</td>
      <td>(Great, Gatsby, by, F)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Gatsby, by, F, ., Scott)</td>
      <td>(Gatsby, by, F, .)</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(by, F, ., Scott, Fitzgerald)</td>
      <td>(by, F, ., Scott)</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Genereate Sample Sentence using N-Gram Language Model on The Great Gatsby
ngram_sample = ngram.sample(200) 
ngram_sample
```




    '\x02 Daisy and Gatsby danced . I remember being surprised by his graceful , conservative foxtrot — I had never seen before . They were so engrossed in each other that she didn ’ t play around with the soldiers any more , but of this clean , hard , limited person , who dealt in universal scepticism , and who leaned back jauntily just within the circle of my arm . A phrase began to beat in my ears with a sort of heady excitement : “ There are only the pursued , the pursuing , the busy , and the tired . ” \x03 \x02 “ Listen , ” said Tom impatiently . “ You make it ten times worse by crabbing about it . ” \x03 \x02 Tom appeared from his oblivion as we were sitting down to supper together . “ Do you mind if I eat with some people over here ? ” he said . \x03 \x02 “ This ? ” he inquired , holding it up . \x03 \x02 “ Well , shall I help myself ? ” Tom demanded . “ You sounded well enough on the phone . \x03 \x02 “ \x03'




```python

```


```python

```


```python

```


```python

```
