#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter
import string
import math
import nltk
import codecs
import sys
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        None

    def features(self, text):
        d = defaultdict(int)
        #Word Count Start
        #t = text.split(" ")
        t = text.split(" ")
        for ii in t:
            d[morphy_stem(ii)] += 1
        
        d['upper'] = 0        
        for ii in text.translate(None, string.punctuation).split():
            if ii.isupper():
                d['upper'] += 1
        
        '''    
        for ii in text:
            if ii in string.punctuation:
                d['punct'+ii] += 1
        '''
        
        charcount = 0
        wordcount = 0
        for ii in text.split():
            charcount += len(ii)
            wordcount += 1 
        d['avgChar'] = float(charcount)/wordcount
        #d["NoOfChar"] = len(text)
        NoOfChar = 0
        for ii in text:
            if not ii == " ":
                NoOfChar += 1
        d["NoOfChar"] = NoOfChar
        
        #No_of_words start
        No_of_words = 0
        d["No_of_words"]=len(t)
        #d["StartWord_"+morphy_stem(t[0].translate(None, string.punctuation))] = 1
        d["EndWord_"+morphy_stem(t[len(t)-1].translate(None, string.punctuation))] = 1
        #No_of_words end
        tags = nltk.pos_tag(text.split())
        d["start_tag_"+str(tags[0][1])] = 1
        d["end_tag_"+str(tags[-1][1])] = 1
        
        #No_of_vowels start
        No_of_vowels=0
        for ii in text.lower().translate(None,string.punctuation):
            if ii in {'a':1,'e':1,'i':1,'o':1,'u':1}:
                No_of_vowels += 1
        d["No_of_vowels"]=No_of_vowels
        #No_of_vowels end
        
        return d
reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(trainfile, delimiter='\t')
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))
    #classifier.show_most_informative_features(50)
    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})

