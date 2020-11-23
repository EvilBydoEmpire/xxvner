# XXVner


XXVner is a large NER model capable of labelling *PER*, *LOC*, *ORG* and *MISC* named entities in 17 languages: English, German, Danish, Dutch, Afrikaans, Norwergian (Bokmal & Nynorsk), Swedish, Finnish, Hungarian, Estonian, Basque, Italian, Spanish, Romanian, Urdu, Persian and Chinese (Mandarin, XXVner-FM only).

To reproduce these results, corpora and sources are provided. Running the .ipynb file in a Jupyter Notebook Environment for training with a GPU is recommended. To use the linked models for prediction, consult the Flair [documentation](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md).

&nbsp;

**Models**

[XXVner-XLM](https://drive.google.com/file/d/10txivO02tKV7kXT6ja9VnGfgRhZG10nD) 

F1 score: 0.8353 - excluding Chinese 'daily, 'msra' & 'weibo' corpora, trained with RoBERTa-XLM embeddings

&nbsp;

[XXVner-FM](https://drive.google.com/file/d/1ViwtJVWgTrBNrojNc9VKd3GplvYK7vT1)  

F1 score: 0.7299 - trained  with Flair's 'multi-forward' & 'multi-backward' embeddings

&nbsp;

**Citations**

*ConLL2003 corpora omitted due to licensing issues, might be easy to find.*
 
&nbsp;

https://github.com/juand-r/entity-recognition-datasets/tree/master/data/BTC 

Leon Derczynski, Kalina Bontcheva, and Ian Roberts: Broad Twitter Corpus: A Diverse Named Entity Recognition Resource. Proceedings of COLING, Osaka, Japan, 2016, pp. 1169-1179.
 
&nbsp;
 
https://www.clips.uantwerpen.be/conll2002/ner/ 

Erik F. Tjong Kim Sang: Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition. Proceedings of CoNLL-2002, Taipei, Taiwan, 2002, pp. 155-158.
 
&nbsp;
 
https://www.clips.uantwerpen.be/conll2003/ner/ 

Erik F. Tjong Kim Sang and Fien De Meulder: Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. Proceedings of CoNLL-2003, Edmonton, Canada, 2003, pp. 142-147.
 
&nbsp;
 
https://danlp.alexandra.dk/304bd159d5de/datasets/ddt.zip

Hvingelby et al.: DaNE: A Named Entity Resource for Danish. Proceedings of the 12th Conference on Language Resources and Evaluation, Marseille, France, 2020, pp. 4597–4604.
 
&nbsp;
 
http://ixa.eus/node/4486

Iñaki Alegria, Olatz Arregi, Nerea Ezeiza, Izaskun Fernandez, Ruben Urizar: Design and Development of a Named Entity Recognizer for an Agglutinative Language. First International Joint Conference on NLP, Sanya, China, 2004.
 
&nbsp;
 
https://nlpado.de/~sebastian/software/ner_german.shtml

Manaal Faruqui and Sebastian Padó: Training and Evaluating a German Named Entity Recognizer with Semantic Generalization. Proceedings of Konvens 2010, Saarbrücken, Germany, 2010.
 
&nbsp;
 
http://linghub.org/metashare/88d030c0acde11e2a6e4005056b40024f1def472ed254e77a8952e1003d9f81e 

A. Tkachenko, Timo Petmanson and S. Laur: Named Entity Recognition in Estonian. Proceedings of the 4th Biennial International Workshop on Balto-Slavic Natural Language Processing. Sofia, Bulgaria, 2013, pp. 78 -83.
 
&nbsp;
 
http://www.evalita.it/2009/tasks/entity 

B. Magnini, E. Pianta, M. Speranza, V. Bartalesi Lenzi and R. Sprugnoli: Local Entity Detection and Recognition Annotation for Evalita 2009. Proceedings of the 11th Conference of the Italian Association for Artificial Intelligence. Reggio Emilia, Italy, 2009.
 
&nbsp;
 
https://github.com/mpsilfve/finer-data 

Teemu Ruokolainen, Pekka Kauppinen, Miikka Silfverberg, and Krister Lindén: A finnish news corpus for named entity recognition. Language Resources and Evaluation, Springer, 2019, pp. 1-26.
 
&nbsp;
 
https://rgai.inf.u-szeged.hu/node/130 

György Szarvas, Richárd Farkas, László Felföldi, András Kocsor, János Csirik: Highly accurate Named Entity corpus for Hungarian. International Conference on Language Resources and Evaluation, Genova, Italy, 2006.
 
&nbsp;
 
https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER

Mandarin Chinese Corpus: Zhibiao Wu: Mandarin Chinese News Text LDC95T13. Linguistic Data Consortium, Philadelphia, USA, 1995.

MSRA: Gina-Anne Levow: Word Segmentation and Named Entity Recognition. The Third International Chinese Language Processing Bakeoff, Chicago, USA, 2006.

Weibo: Nanyun Peng and Mark Dredze: Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, Lisbon, Portual, 2015.
 
&nbsp;
 
https://repo.sadilar.org/handle/20.500.12185/299 

R. Eiselen: Government domain named entity recognition for South African languages. Proceedings of the 10th Language Resource and Evaluation Conference, Portorož, Slovenia, 2016.
 
&nbsp;
 
https://github.com/ljos/navnkjenner

Bjarte Johansen: Named-Entity Recognition for Norwegian. Proceedings of the 22nd Nordic Conference on Computational Linguistics, Turku, Finnland, 2019.
 
&nbsp;
 
https://github.com/HaniehP/PersianNER 

Hanieh Poostchi, Ehsan Zare Borzeshi and Massimo Piccardi: BiLSTM-CRF for Persian Named-Entity Recognition; ArmanPersoNERCorpus: the First Entity-Annotated Persian Dataset. The 11th Edition of the Language Resources and Evaluation Conference, Miyazaki, Japan, 2018.
 
&nbsp;
 
https://www.dropbox.com/sh/1ivw7ykm2tugg94/AAB9t5wnN7FynESpo7TjJW8la 

Safia Kanwal, Kamran Malik, Khurram Shahzad, Faisal Aslam: Urdu Named Entity Recognition: Corpus Generation and Deep Learning Applications. ACM Transactions on Asian and Low-Resource Language Information Processing, 2019.
 
&nbsp;
 
https://github.com/aritter/twitter_nlp 

Alan Ritter, Sam Clark, Mausam and Oren Etzioni. Named entity recognition in tweets: An experimental study. Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing, Edinburgh, UK, 2011, pp 1524-1534.
 
&nbsp;
 
https://github.com/dumitrescustefan/ronec 

Stefan Daniel Dumitrescu and Andrei-Marius Avram: Introducing RONEC - The Romanian Named Entity Corpus. arXiv: 1909.01247, 2019.
 
&nbsp;
 
https://www.ling.su.se/english/nlp/corpora-and-resources/sic 

Robert Östling, Johan Sjons and Johannes Bjerva: Stockholm Internet Corpus. Web, retrieved on 18.11.2020, 2016.
 
&nbsp;
 
https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold

Dominic Balasuriya, Nicky Ringland, Joel Nothman, Tara Murphy and James R. Curran: Named entity recognition in Wikipedia. Proceedings of the 2009 Workshop on The People's Web Meets NLP: Collaboratively Constructed Semantic Resources, Suntec, Singapore, 2009.
