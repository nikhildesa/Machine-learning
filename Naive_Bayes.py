
import math

def naive_bayes_prediction(p_spam,p_ham,per_spam,per_ham):
    ans = p_spam * per_spam / ((p_spam * per_spam) + (p_ham * per_ham))
    return ans
 
def vocabulary(vocab, words):
    p_spam = 0
    p_ham = 0
    for key, value in vocab.items():
        if key in words:
            p_spam += math.log(value[1])
            p_ham += math.log(value[0])
        else:
            p_spam += math.log((1 - value[1]))
            p_ham += math.log((1 - value[0]))
    return math.exp(p_spam), math.exp(p_ham)



def test(test_file, vocab, stop_words, per_spam, per_ham):
    spam_test = 0
    ham_test = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    with open(test_file, 'r', encoding = 'unicode-escape') as f:
        for textline in f:
            is_spamtest = int(textline[:1])
            if is_spamtest ==1:
                spam_test += 1
            else:
                ham_test += 1
            textline = cleantext(textline[1:])
            words = clean_stopwords(textline, stop_words)
            p_spam, p_ham = vocabulary(vocab, words)
            
            ans = naive_bayes_prediction(p_spam, p_ham,per_spam, per_ham)
            if ans >= 0.5 and is_spamtest == 1:  
                tp += 1
            elif ans < 0.5 and is_spamtest == 0: 
                tn += 1
            elif ans < 0.5 and is_spamtest == 1: 
                fn += 1
            elif ans >= 0.5 and is_spamtest == 0:
                fp += 1
    
    print("Spam subject line in test set: ", spam_test)
    print("Ham subject line in test set: ", ham_test)
    
    print("True Positive is ",tp)
    print("True negative is ",tn)
    print("False positive is ",fp)
    print("False negative is ",fn)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (1 / ((1 / precision) + (1 / recall)))
    
    print("Accuracy on Test set: ", accuracy)
    print("Recall Test set: ", recall)
    print("f1_score Test set: ", f1_score)
    print("Precision Test set: ", precision)
   
    
    
    
def make_percent_list(k,counted,spam,ham):
    for each_key in counted:
        counted[each_key][0] = (counted[each_key][0] + k) / (2 * k + ham)
        counted[each_key][1] = (counted[each_key][1] + k) / (2 * k + spam)
    return counted
        
    
    
def countwords(words,is_spam,counted):
    for each_word in words:
        if each_word in counted:
            if is_spam == 1:
                counted[each_word][1] = counted[each_word][1] + 1
            else:
                counted[each_word][0] = counted[each_word][0] + 1
        else:
            if is_spam == 1:
                counted[each_word] = [0, 1] 
            else:
                counted[each_word] = [1, 0]
    return counted



def clean_stopwords(textline,stop_words):
    textline = textline.split()
    words = set(textline) 
    words = words.difference(stop_words)
    return words
    
      
def cleantext(text):
    text = text.lower()
    text = text.strip()
    for letters in text:
        if letters in """[]!.,"-!—@;':#$%^&*()+/?""":
            text = text.replace(letters," ")
    return text


def compute(train_file,stop_words):
    spam = 0
    ham = 0
    counted = {}
    with open(train_file, 'r', encoding = 'unicode-escape') as f:
        for textline in f:
            is_spam = int(textline[:1])
            if is_spam == 1:
                spam += 1
            else:
                ham += 1
            textline = cleantext(textline[1:])
            words = clean_stopwords(textline, stop_words)
            counted = countwords(words, is_spam, counted)
                        
    return counted, spam, ham



def getstop_words(stop_file):
    stopwords = set()
    with open(stop_file) as f:
        for line in f:
            if line != '\n':
                word = line.strip()
                stopwords.add(word)
    return stopwords
    

def naivebayes():
    train_file = input("Enter your training set: ")  
    stop_file = input("Enter your stop words: ")        
    test_file = input("Enter your test set: ")            
    
    stop_words = getstop_words(stop_file)
    counted, spam, ham = compute(train_file, stop_words)
    vocab = make_percent_list(1, counted, spam, ham)
    
    per_spam = spam /(spam+ham)
    per_ham = ham /(spam+ham)
    test(test_file, vocab, stop_words, per_spam, per_ham)

if __name__ == '__main__':
    naivebayes()