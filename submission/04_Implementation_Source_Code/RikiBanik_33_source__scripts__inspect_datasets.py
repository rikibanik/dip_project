"""Print dataset source and initial licensing/size review checklist. No downloads occur."""
SOURCES={
 "Strata Insurance Corpus (753 rows, synthetic, CC BY 4.0)":"https://huggingface.co/datasets/NikolaiSachok/strata-insurance-corpus",
 "FUNSD (199 forms; non-commercial research/education terms)":"https://guillaumejaume.github.io/FUNSD/work/",
 "RVL-CDIP (400K pages; license listed as other)":"https://huggingface.co/datasets/aharley/rvl_cdip",
 "CORD (receipt parsing; CC BY 4.0, hosted set about 2.3 GB)":"https://github.com/clovaai/cord",
 "Policy Structure Pages (gated, research-only, no redistribution)":"https://huggingface.co/datasets/Ashish-Ranjan/Policy-Structure-Pages"}
if __name__=="__main__":
    print("No dataset has been downloaded. Before acquisition, re-check each live card/terms for license, consent, copyright, format, size and permitted use. See docs/datasets.md.")
    for name,url in SOURCES.items():print(f"- {name}: {url}")
