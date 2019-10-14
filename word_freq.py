import re

filename = input()
frequency = {}
document_text = open('D:\\BIOS-Hackaton\\Katalis\\dataset\\' + filename + ".txt", 'r')
text_string = document_text.read().lower()
match_pattern = re.findall(r'\b[a-z]{3,15}\b', text_string)

for word in match_pattern:
    count = frequency.get(word, 0)
    frequency[word] = count + 1

frequency_list = frequency.keys()
print(len(frequency_list))

for words in frequency_list:
    print (words, frequency[words])
