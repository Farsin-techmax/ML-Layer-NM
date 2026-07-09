with open('features.py', 'r', encoding='utf-8') as f:
    code = f.read()
code = code.replace(', format=\'mixed\'', '')
code = code.replace(', format=\"mixed\"', '')
code = code.replace('format=\'mixed\', ', '')
code = code.replace('format=\"mixed\", ', '')
with open('features.py', 'w', encoding='utf-8') as f:
    f.write(code)
print('Done!')
