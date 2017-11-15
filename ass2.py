import nltk
import re
import itertools

def ExtractRelFromIEER(NameOfDoc, SubjClass, ObjClass, Pattern, NameOfPattern):
    result = []
    for doc in nltk.corpus.ieer.parsed_docs(NameOfDoc):
        for rel in nltk.extract_rels(SubjClass, ObjClass, doc, corpus='ieer', pattern=Pattern):
            result.append(nltk.clause(rel, relsym=NameOfPattern))

    return result

pattern_of = re.compile(r'.*\bof\b')
pattern_in = re.compile(r'.*\bin\b')

NameOfDocList = ['APW_19980314','APW_19980424','APW_19980429','NYT_19980315','NYT_19980403','NYT_19980407']
SubjClassList = ['LOCATION','ORGANIZATION','PERSON','DURATION','DATE','CARDINAL','PERCENT','MONEY','MEASURE']
final_result = []

for name in NameOfDocList:
    for sub in SubjClassList:
        extracted = ExtractRelFromIEER(name, sub, 'ORGANIZATION', pattern_of, 'OF')
        final_result.append(extracted)
        extracted = ExtractRelFromIEER(name, sub, 'ORGANIZATION', pattern_in, 'IN')
        final_result.append(extracted)

final_result=list(itertools.chain.from_iterable(final_result))

print(final_result)
print(len(final_result))