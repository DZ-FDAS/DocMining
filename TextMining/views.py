from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse

from rest_framework.decorators import api_view

from django.shortcuts import render
from django.http import HttpResponse

import sys
import pandas as pd
import re
from fuzzywuzzy import fuzz, process
import string
import nltk
nltk.download("all")
import en_core_web_sm
import datefinder
import json
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from io import TextIOWrapper

from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view

from django.core.files.storage import FileSystemStorage

from django.conf import settings

import os


#Reading file from path given by user
def readFile(path):
    try:
        #implement to read any kind of file
        with open(path) as f:
            lines = f.readlines()
        lines = map(lambda s: s.strip(), lines)
        lines = [i for i in lines if i] 
        return lines
    except FileNotFoundError:
        print("File not found")
        sys.exit()
    except Exception as e:
        print("Error occured in readFile: ",e)
        sys.exit()
            
#Convert list to dataframe and create old_index column
def convertToDataframe(lines):
    try:
        df = pd.DataFrame(lines,columns=['line'])
        df['old_index'] = df.index
        return df
    except Exception as e:
        print("Error occured in convertToDataframe: ",e)
        sys.exit()
    
#extract all headers from dataframe
def extractHeaders(df):
    try:
        pattern = re.compile(r'\d+\.|\([a-zA-Z]+\)\.?|[0-9a-zA-Z]\.|^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.$')
        df_headers=df[df['line'].str.match(pattern)]
        return df_headers
    except Exception as e:
        print("Error occured in extractHeaders: ",e)
        sys.exit()

#filter headers from dataframe based on first header
def filterHeaders(df_headers):
    try:
        token = df_headers['line'].iloc[0]
        
        if re.match(r'\([a-zA-Z]+\)\.?', token):
            pat = r'\([a-zA-Z]+\)\.?'
        elif re.match(r'\d+\.',token):
            pat = r'\d+\.'
        elif re.match(r'[0-9a-zA-Z]\.',token):
            pat = r'[0-9a-zA-Z]\.'
        elif re.match(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.$', token):
            pat = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.$'
        else:
            print("No headings found in document")
        df_headers_filter = df_headers[df_headers['line'].str.match(pat)].reset_index()
        return df_headers_filter
    except Exception as e:
        print("Error occured in filterHeaders: ",e)
        sys.exit()

#further filter only relevant files
def secondFilterHeaders(df_headers_filter):
    try:
        #implement code to let user pic his own headings
        words = ['deliverable','delivery','scope','work','term','termination',
                 'payment','milestone','responsibility','confidentiality','intellectual',
                 'property','liability','dispute','legal']
        words_string = ' '.join([str(item) for item in words]) 
        df_headers_second_filter = process.extract(words_string, df_headers_filter['line'],scorer=fuzz.token_set_ratio)
        return df_headers_second_filter
    except Exception as e:
        print("Error occured in secondFilterHeaders: ",e)
        sys.exit()
   
#extract text to find organization and start date
def extractImpText(df_headers_filter,df):
    try:
        var = df_headers_filter['old_index'][0]
        df_text = df[df['old_index']<var]
        imptext = ' '.join([str(item) for item in df_text['line']]) 
        return imptext
    except Exception as e:
        print("Error occured in extractImpText: ",e)
  
#extract organization
def extractOrganization(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        organization_entities = set()
        exclude = set(string.punctuation)
        stopword = nltk.corpus.stopwords.words('english')
        text = ''.join(ch for ch in text if ch not in (exclude,stopword))
        doc1 = nlp(text.strip())
        for i in doc1.ents:
            entry = str(i.lemma_).lower()
            text = text.replace(str(i).lower(), "")
            if i.label_ in ["ORG"]:
                organization_entities.add(entry)
        organization_entities = list(organization_entities)    
        return organization_entities
    except Exception as e:
        print("Error occured in extractOrganization: ",e)
        sys.exit()
 
#extract date
def extractDate(text):
    try:
        matches = datefinder.find_dates(text)
        match_str = ' '.join([str(item) for item in matches])
        return match_str
    except Exception as e:
        print("Error occured in extractDate: ",e)
        sys.exit()
  
#frequency table for text summarization
def _create_frequency_table(text_string) -> dict:
    try:
        stopWords = set(stopwords.words("english"))
        words = word_tokenize(text_string)
        ps = PorterStemmer()
        freqTable = dict()
        for word in words:
            word = ps.stem(word)
            if word in stopWords:
                continue
            if word in freqTable:
                freqTable[word] += 1
            else:
                freqTable[word] = 1
        return freqTable
    except Exception as e:
        print("Error occured in _create_frequency_table: ",e)
        sys.exit()
        
#sentence score for text summarization
def _score_sentences(sentences, freqTable) -> dict:
    try:
        sentenceValue = dict()
        for sentence in sentences:
            word_count_in_sentence = (len(word_tokenize(sentence)))
            for wordValue in freqTable:
                if wordValue in sentence.lower():
                    if sentence[:10] in sentenceValue:
                        sentenceValue[sentence[:10]] += freqTable[wordValue]
                    else:
                        sentenceValue[sentence[:10]] = freqTable[wordValue]
            sentenceValue[sentence[:10]] = sentenceValue[sentence[:10]] // word_count_in_sentence
        return sentenceValue
    except Exception as e:
        print("Error occured in _score_sentences: ",e)
        sys.exit()
        
#calculationg treshold
def _find_average_score(sentenceValue) -> int:
    try:
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]
        average = int(sumValues / len(sentenceValue))
        return average/2
    except Exception as e:
        print("Error occured in _find_average_score: ",e)
        sys.exit()
        
#generating summary
def _generate_summary(sentences, sentenceValue, threshold):
    try:
        sentence_count = 0
        summary = ''
        for sentence in sentences:
            if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] > (threshold):
                summary += " " + sentence
                sentence_count += 1
        return summary
    except Exception as e:
        print("Error occured in _generate_summary: ",e)
        sys.exit()
        
#text summarization with word rank algorithm
def summarizeText(text):
    try:
        text = text.replace('\n','')
        #print('\noriginal text:\n',text)
        freq_table = _create_frequency_table(text)
        sentences = sent_tokenize(text)
        sentence_scores = _score_sentences(sentences, freq_table)
        threshold = _find_average_score(sentence_scores)
        summary = _generate_summary(sentences, sentence_scores, 1.5 * threshold)
        return summary
    except Exception as e:
        print("Error occured in summarizeText: ",e)
        sys.exit()

def index(request):
    return render(request,'TextMining/index.html')

def fileDisplay(request):
    print("Hello")
    path1 = settings.MEDIA_ROOT
    
    entries = os.listdir(path1)
    print(entries)
    filepath = os.path.join(path1,entries[-1])
    #filex = open(filepath)
    print(filepath)
    print("Hello2")
    context = {'filepath': filepath}
    return Response(filepath)

@api_view(['POST'])
def MainFun(request):
    myfile = request.FILES['myfile']
    path1 = settings.MEDIA_ROOT
    fs = FileSystemStorage(location='TextMining/media/')
    file1 = fs.save(myfile.name, myfile)
    fileurl = fs.url(file1)
    print(fileurl)
    summary_list = {}
    str_text = []
    for line in myfile:
        str_text.append(line.decode('unicode_escape'))
    df = convertToDataframe(str_text)
    df_headers = extractHeaders(df)
    df_headers_filter = filterHeaders(df_headers)
    df_headers_second_filter = secondFilterHeaders(df_headers_filter)
    text = extractImpText(df_headers_filter,df)
    org = extractOrganization(text)
    summary_list['org']= org
    date = extractDate(text)
    summary_list['date'] = date
    for i in range(0,len(df_headers_second_filter)):
        code = df_headers_second_filter[i][0]
        df_text = df_headers_filter[df_headers_filter["line"].str.contains(code)]
        old_index1 = df_text["old_index"].iloc[0]
        index1 = df_headers_filter[df_headers_filter['old_index']==old_index1].index.values
        if old_index1 == max(df_headers_filter['old_index']):
            old_index2 = max(df_headers_filter['old_index'])
        else:
            index2=index1[0]+1
            old_index2 = df_headers_filter['old_index'].loc[index2]
        ans = ''
        df1 = df[(df['old_index'] >= old_index1) & (df['old_index'] < old_index2)]
        for j in df1['line']:
            ans=ans+j
        summary = summarizeText(ans)
        summary_list[i+2] = summary
    json_data = json.dumps(summary_list)
    return render(request, 'TextMining/Doc-detail.html', summary_list)