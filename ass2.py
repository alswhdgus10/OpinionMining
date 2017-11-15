import nltk
import re

def ExtractRelFromIEER(NameOfDoc, SubjClass, ObjClass, Pattern, NameOfPattern):
    result = []
    for doc in nltk.corpus.ieer.parsed_docs(NameOfDoc):
        for rel in nltk.extract_rels(SubjClass, ObjClass, doc, corpus='ieer', pattern=Pattern):
            # result.append(nltk.rtuple(rel, lcon=False, rcon=False))
            result.append(nltk.clause(rel, relsym=NameOfPattern))

    return result

pattern_list = []
pattern_of = re.compile(r'.*\bof\b')
pattern_list.append(pattern_of)
pattern_in = re.compile(r'.*\bin\b')
pattern_list.append(pattern_in)

final_result = []
for k in pattern_list:
    extracted = ExtractRelFromIEER('APW_19980314', 'PERSON', 'ORGANIZATION', k, 'OF')
    final_result.append(extracted)

print(final_result)