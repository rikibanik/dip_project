from __future__ import annotations
import os
import shutil
import numpy as np
from PIL import Image
from src.config import TESSERACT_CMD

class OCREngine:
    def __init__(self, prefer: str = "paddleocr", tesseract_psm: int | None = None):
        if prefer not in {"paddleocr", "tesseract"}:
            raise ValueError("prefer must be 'paddleocr' or 'tesseract'")
        if tesseract_psm is not None and tesseract_psm not in {3, 6, 11}:
            raise ValueError("tesseract_psm must be one of 3, 6, or 11")
        self.tesseract_psm=tesseract_psm
        self.kind="unavailable"; self.reader=None; self.error=None
        if prefer == "paddleocr":
            try:
                from paddleocr import PaddleOCR
                self.reader=PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
                self.kind="paddleocr"
                return
            except Exception as exc: self.error=f"PaddleOCR unavailable: {exc}"
        try:
            import pytesseract
            command=os.getenv("TESSERACT_CMD")
            candidates=[command,TESSERACT_CMD,r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",shutil.which("tesseract")]
            executable=next((candidate for candidate in candidates if candidate and (os.path.isfile(candidate) or shutil.which(candidate))),None)
            if executable:
                pytesseract.pytesseract.tesseract_cmd=executable
                self.reader=pytesseract; self.kind="tesseract"
            elif prefer=="tesseract":
                self.error=f"Tesseract executable not found. Checked configured path {TESSERACT_CMD!r}; set TESSERACT_CMD to the installed executable."
        except Exception as exc: self.error=f"Tesseract unavailable: {exc}"

    def extract(self,image:np.ndarray,page_number:int=1)->dict:
        if self.kind=="unavailable": return {"page":page_number,"engine":self.kind,"status":"unavailable","text":"","lines":[],"error":self.error or "Install PaddleOCR or Tesseract."}
        lines=[]
        try:
            if self.kind=="paddleocr":
                result=self.reader.ocr(image,cls=True) or []
                for page in result:
                    for item in page or []:
                        box, (text, confidence)=item
                        points=np.asarray(box).reshape(-1,2)
                        lines.append({"text":str(text),"confidence":float(confidence),"bbox":[int(points[:,0].min()),int(points[:,1].min()),int(points[:,0].max()),int(points[:,1].max())]})
            else:
                from pytesseract import Output
                config=f"--psm {self.tesseract_psm}" if self.tesseract_psm is not None else ""
                data=self.reader.image_to_data(Image.fromarray(image),output_type=Output.DICT,config=config)
                grouped={}
                for i,text in enumerate(data["text"]):
                    text=str(text).strip()
                    if not text: continue
                    key=(data["block_num"][i],data["par_num"][i],data["line_num"][i])
                    box=[int(data["left"][i]),int(data["top"][i]),int(data["left"][i]+data["width"][i]),int(data["top"][i]+data["height"][i])]
                    conf=float(data["conf"][i]); conf=conf/100 if conf>=0 else 0
                    if key not in grouped: grouped[key]={"text":[],"confidence":[],"boxes":[]}
                    grouped[key]["text"].append(text); grouped[key]["confidence"].append(conf); grouped[key]["boxes"].append(box)
                for item in grouped.values():
                    boxes=np.asarray(item["boxes"])
                    lines.append({"text":" ".join(item["text"]),"confidence":float(np.mean(item["confidence"])),"bbox":[int(boxes[:,0].min()),int(boxes[:,1].min()),int(boxes[:,2].max()),int(boxes[:,3].max())]})
            return {"page":page_number,"engine":self.kind,"status":"ok","text":"\n".join(x["text"] for x in lines),"lines":lines,"error":None}
        except Exception as exc:
            return {"page":page_number,"engine":self.kind,"status":"error","text":"","lines":[],"error":str(exc)}

def compare_variants(variants:dict, engine:OCREngine, page_number:int=1)->dict:
    results={}
    for name, variant in variants.items():
        item=engine.extract(variant.image,page_number)
        scores=[line["confidence"] for line in item.get("lines",[]) if line.get("confidence") is not None]
        item.update({"variant":name,"parameters":variant.parameters,"mean_confidence":float(np.mean(scores)) if scores else None,"character_count":len(item.get("text","")),"word_count":len(item.get("text","").split())})
        results[name]=item
    successful=[x for x in results.values() if x["status"]=="ok"]
    best=max(successful,key=lambda x:((x["mean_confidence"] or 0),x["character_count"]),default=None)
    return {"page":page_number,"engine":engine.kind,"variants":results,"best_variant":best["variant"] if best else None,"status":"ok" if successful else "unavailable"}
