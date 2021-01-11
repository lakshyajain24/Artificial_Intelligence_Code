'''
Here We are going to find Levenshtein Distance between two words 'Funnily' and 'Funny'
'''

# Installing the Library
!pip install python-Levenshtein
import Levenshtein

# To find the Levenshtine Distance between two strings
from Levenshtein import StringMatcher
word1 = 'funnily'
word2 = 'funny'
result = StringMatcher.distance(word1, word2)
print("\nLevenshtein distance between", word1, "and", word2, "is", result)

# a function that tells the cost of Levenshtein_Distance operations
'''
Here Substitution opperation gets cost='2' 
else other operations gets cost='1'
'''
def Levenshtein_Distance_Cost(str1,str2):
# Importing the Library 
  from Levenshtein import StringMatcher
  # converting upper letters into lower letters
  str1 = str1.lower()
  str2 = str2.lower()
  #Using the Editops function to get the operations
  LD_editops = StringMatcher.editops(str1,str2)
  # assigning the cost
  cost = 0
  for i in LD_editops:
  #Assigning the cost
    if(i[0]=='replace'):
      cost = cost+2
    else:
      cost = cost+1
  return cost

#example
Levenshtein_Distance_Cost('Funnily',"fanny")# cost will be 4
