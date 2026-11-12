from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

LABELS=["insurance_policy","insurance_claim_form","medical_invoice","vehicle_repair_invoice","identity_document","unknown"]
RULES={"insurance_policy":["policy number","sum insured","premium","policy period","insurance policy"],
       "insurance_claim_form":["claim number","claimant","date of incident","claim form","incident description"],
       "medical_invoice":["hospital","patient","doctor","consultation","medical","pharmacy","diagnostic"],
       "vehicle_repair_invoice":["garage","vehicle repair","spare parts","labour","workshop","repair invoice","service center","service centre"],
       "identity_document":["date of birth","passport","driving licence","identity card","national id"]}

def classify_document(text:str)->dict:
    normalized=re.sub(r"\s+"," ",text.lower())
    scores={label:sum(normalized.count(term) for term in terms) for label,terms in RULES.items()}
    best=max(scores,key=scores.get)
    if scores[best]==0: return {"label":"unknown","confidence":0.0,"status":"uncertain","method":"keyword_baseline","scores":scores}
    total=sum(scores.values()); confidence=scores[best]/total
    return {"label":best,"confidence":round(confidence,4),"status":"classified" if scores[best]>=2 else "low_evidence","method":"keyword_baseline","scores":scores}

def train_classifier(texts:list[str],labels:list[str]):
    if len(texts)!=len(labels) or len(set(labels))<2: raise ValueError("Training requires aligned examples from at least two classes")
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=20000),LogisticRegression(max_iter=1000,class_weight="balanced"))
    return model.fit(texts,labels)

def predict_trained(model,text:str)->dict:
    probabilities=model.predict_proba([text])[0]; idx=int(probabilities.argmax())
    return {"label":str(model.classes_[idx]),"confidence":float(probabilities[idx]),"status":"classified" if probabilities[idx]>=0.6 else "uncertain","method":"tfidf_logistic_regression"}
