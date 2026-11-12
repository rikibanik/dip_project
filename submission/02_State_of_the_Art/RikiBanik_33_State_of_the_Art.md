# State of the Art

Document OCR combines image preparation, text recognition, and semantic extraction. Global thresholding uses a common intensity cutoff; Otsu selects a threshold from gray-level histogram class separation [1]. CLAHE improves local contrast with clipped histogram amplification [2]. Morphology changes foreground structures according to a structuring element, which can remove faint strokes or join glyphs. These operations have image-dependent effects.

Tesseract offers page segmentation modes that encode layout assumptions [3]. PSM 3 performs automatic segmentation; PSM 6 assumes one uniform text block; PSM 11 targets sparse text. CER and WER measure edit distance normalized by reference characters or words, and depend on transcription fidelity, tokenization, normalization, and order. FUNSD is a noisy scanned form dataset [4], used separately here as a general-form benchmark rather than insurance evidence.

This work integrates established operations into an inspectable local workflow. Its contribution is engineering integration and descriptive comparison, not a new OCR or image-processing algorithm. Results on a small synthetic sample are exploratory.

## References
[1] N. Otsu, “A Threshold Selection Method from Gray-Level Histograms,” IEEE Trans. Syst., Man, Cybern., vol. 9, no. 1, pp. 62–66, 1979, doi: 10.1109/TSMC.1979.4310076.
[2] K. Zuiderveld, “Contrast Limited Adaptive Histogram Equalization,” in Graphics Gems IV, 1994, pp. 474–485.
[3] R. Smith, “An Overview of the Tesseract OCR Engine,” Proc. ICDAR, 2007, pp. 629–633, doi: 10.1109/ICDAR.2007.4376991.
[4] G. Jaume et al., “FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents,” Proc. ICDAR Workshops, 2019, doi: 10.1109/ICDARW.2019.10029. Verify full author metadata and pagination against publisher record before submission.
