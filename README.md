# PoetryClassification

## FEATURES USED:

## 1. WORD COUNT:
### Code:
    for ii in t:
            d[morphy_stem(ii)] += 1
    
### Explanantion: 
    a. Given feature in the baseline code provided. 
    b. It stores the word count of the words in a line having same morphological stem.
    

## 2. COUNT OF UPPER CASE WORDS:
### Code:
    d['upper'] = 0        
        for ii in text.translate(None, string.punctuation).split():
            if ii.isupper():
                d['upper'] += 1
                
### Explanation:
    a. This finds out count of Upper case words in the given text line.
    b. Emily Bronte has many Upper cased words while William Shakespeare does not have any single upper cased word.
    c. Thus this can be a good feature to indentify whether the work belongs to Emily Bronte or William Shakespeare.


## 3. AVERAGE WORD LENGTH:
### Code:
    d['avgChar'] = float(charcount)/wordcount
    
### Explanation:
    a. This gives the average characters in a word of a line.
    b. This is one of the most informative features where Bronte uses 4 letter words more than Shakespeare in the ratio 30.4:1.0
    
    
## 4. NUMBER OF CHARACTERS IN A TEXT LINE:
### Code:
    NoOfChar = 0
        for ii in text:
            if not ii == " ":
                NoOfChar += 1
        d["NoOfChar"] = NoOfChar
        
### Explanation:
    a. This is the most informative feature.
    b. Length of Emily Bronte's line is pretty less as compared to Shakespeare's length of the text line.
    c. Lines having length 24 is more dominant in Emily than Shakespeare in the ratio 56.3:1.0
    
 
## 5. NUMBER OF WORDS IN A LINE:
### Code:
    d["No_of_words"]=len(t)
    
### Explanation:
    a. As stated above, Emily Bronte's lines are smaller than Shakespeare's.
    b. This means that either the word length is smaller than Shakespeare's or Number of words are less than Shakespeare's. Hence the feature.
    c. This feature is also one of the highly informative features.
    

## 6. END WORD OF A LINE:
### Code:
    d["EndWord_"+morphy_stem(t[len(t)-1].translate(None, string.punctuation))] = 1
    
### Explanation:
    a. There are some words like "sky" with which Bronte ends her lines frequently.
    b. Shakespeare seldom uses same words to end the lines.
    

## 7. START POS TAG
### Code:
    tags = nltk.pos_tag(text.split())
        d["start_tag_"+str(tags[0][1])] = 1
    
### Explanation:
    a. This checks how many times the line starts with the same POS tag.


## 8. END POS TAG
### Code:
    tags = nltk.pos_tag(text.split())
    d["end_tag_"+str(tags[-1][1])] = 1
    
### Explanation:
    a. This checks how many times the line ends with the same POS tag.


## 9. NUMBER OF VOWELS:
### Code:
    No_of_vowels=0
    for ii in text.lower().translate(None,string.punctuation):
        if ii in {'a':1,'e':1,'i':1,'o':1,'u':1}:
            No_of_vowels += 1
    d["No_of_vowels"]=No_of_vowels
    
### Explanation:
    a. Number of vowels are directly proportional to the syllables in the line.
    b. Shakespeare uses around 10 syllables per line and thus, is a good feature to distinguish between Bronte and Shakespeare.
    c. This is one of the most informative feature having Bronte to Shakespeare vowel count ratio 35.9:1.0 for 6 vowels in a line.
