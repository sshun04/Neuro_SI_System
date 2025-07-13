Department of Language Science and Technology
UdS
The EPIC-UdS Corpus
A Corpus for Interpretation Research


Description
EPIC-UdS is a trilingual parallel and comparable corpus of speeches held in the European Parliament (by MEPs) and follows the European Parliament Interpreting Corpora tradition of EPIC (Bologna) and EPICG (Ghent). It contains original speeches from 2008 to 2013 by English, German and Spanish native speakers and their interpretation (EN to/from DE, ES to EN)

The main part of the English dataset is based on transcripts from existing European Parliament corpora: TIC (Kajzer-Wietrzny2012 for parts of ORG_SP_EN, SI_DE_EN and SI_ES_EN) and EPICG (Defrancq2015 for some ORG_SP_EN). Other speeches for the English dataset as well as the whole German and Spanish component were newly transcribed at Saarland University. All transcripts are based on videos of European Parliament Proceedings published by the European Parliament downloaded from https://www.europarl.europa.eu/plenary/en/debates-video.html (accessed 2018/2019).

All transcripts, including the existing material, were revised according to transcription guidelines based on EPICG (Bernardini2018) ensuring comparability across the different datasets. The material includes typical characteristics of spoken language such as false starts, hesitations and truncated words. To obtain better results for source-target alignment as well as sentence parsing the transcripts were segmented using a main clause approach: compound sentences were segmented separately.

For the EPIC-UdS V2 Version, transcripts were processed clause by clause with spaCy NLP tools (version 2.3.4) and the corresponding language models (de_core_news_lg-2.3.0, en_core_web_lg-2.3.1, es_core_news_lg-2.3.1). Data is encoded in CoNLL-U format and provides universal POS tags, fine-grained language-specific POS as well as universal dependency relations. Parsing accuracy of the German and English dataset was assessed as LAS = 69.74 and UAS = 76.64 for German and LAS = 78.73 and UAS = 86.42. The evaluation was carried out manually by two evaluators independently on a set of 50 randomly selected sentences for each language.

All data was enriched with relevant metadata such as source language, name of original speaker, speech timing, mode of delivery and delivery rate.

Corpus Overview
Subcorpus	Tokens	Sentences
Original spoken English (ORG_SP_EN)	67,526	3,622
Simultaneous interpreting from English into German (SI_EN_DE)	57,532	4,076
Original spoken German (ORG_SP_DE)	56,488	3,409
Simultaneous interpreting from German into English (SI_DE_EN)	58,503	3,623
Original spoken Spanish (ORG_SP_ES)	53,947	2,537
Simultaneous interpreting from Spanish into English (SI_ES_EN)	54,630	3,076
Download
The EPIC-UdS corpus comes in a single zip archive containing three corpora enriched with metadata: Spanish texts (originals), German texts (originals and interpretations), English texts (originals and interpretations from Spanish and German)

EPIC-UdS metadata-rich files for English, German and Spanish as a zip archive (2.1 MB)
Citing the EPIC-UdS corpus
If you use this corpus, please cite the following reference:

Heike Przybyl, Alina Karakanta, Katrin Menzel, and Elke Teich (2022). Exploring linguistic variation in mediated discourse: translation vs. interpreting. In Marta Kajzer-Wietrzny, Adriano Ferraresi, Ilmari Ivaska, and Silvia Bernardini, editors, Empirical investigations into the forms of mediated discourse at the European Parliament, Translation and Multilingual Natural Language Processing, p. 191–218. Language Science Press, Berlin.



Licence
All versions of the EPIC-UdS corpus are licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.