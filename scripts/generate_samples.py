from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import cv2,numpy as np
def make(text:list[str],name:str):
    img=np.full((1000,1400,3),245,np.uint8)
    y=100
    for line in text:
        cv2.putText(img,line,(90,y),cv2.FONT_HERSHEY_SIMPLEX,0.85,(35,45,55),2,cv2.LINE_AA);y+=70
    path=ROOT/"datasets"/"samples"/name;path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(path),img)
if __name__=="__main__":
    make(["SYNTHETIC DEMO - NOT A REAL POLICY","Insurance Policy","Policy Number: DEMO-2026-001","Policyholder: Alex Example","Insurance Company: Sample Cover Ltd","Effective Date: 01/01/2026","Expiry Date: 31/12/2026","Sum Insured: 500000","Premium: 12500"],"sample_policy.png")
    make(["SYNTHETIC DEMO - NOT A REAL CLAIM","Insurance Claim Form","Claim Number: CLAIM-DEMO-04","Policy Number: DEMO-2026-001","Claimant: Alex Example","Date of Incident: 15/06/2026","Claimed Amount: 12500","Incident Description: Demonstration only, no real event."],"sample_claim.png")
    make(["SYNTHETIC DEMO - NOT A REAL INVOICE","Invoice Number: INV-DEMO-009","Date: 16/06/2026","Hospital: Example Demo Hospital","Tax: 500","Total Amount: 12500"],"sample_invoice.png")
    print("Created synthetic files in datasets/samples; they contain no real policy or claim.")
