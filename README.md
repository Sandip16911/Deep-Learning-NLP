# Table of Contents:
1. Introduction to Torch's Tensor Library
2. Computation Graphs and Automatic Differentiation
3. Deep Learning Building Blocks: Affine maps, non-linearities, and objectives
4. Optimization and Training
5. Creating Network Components in Pytorch
  * Example: Logistic Regression Bag-of-Words text classifier
6. Word Embeddings: Encoding Lexical Semantics
  * Example: N-Gram Language Modeling
  * Exercise: Continuous Bag-of-Words for learning word embeddings
7. Sequence modeling and Long-Short Term Memory Networks
  * Example: An LSTM for Part-of-Speech Tagging
  * Exercise: Augmenting the LSTM tagger with character-level features
8. Advanced: Dynamic Toolkits, Dynamic Programming, and the BiLSTM-CRF
  * Example: Bi-LSTM Conditional Random Field for named-entity recognition
  * Exercise: A new loss function for discriminative tagging

# What is this tutorial?
I am writing this tutorial because, although there are plenty of other tutorials out there, they all seem to have one of three problems:
* They have a lot of content on computer vision and conv nets, which is irrelevant for most NLP (although conv nets have been applied in cool ways to NLP problems).
* Pytorch is brand new, and so many deep learning for NLP tutorials are in older frameworks, and usually not in dynamic frameworks like Pytorch, which have a totally different flavor.
* The examples don't move beyond RNN language models and show the awesome stuff you can do when trying to do lingusitic structure prediction.  I think this is a problem, because Pytorch's dynamic graphs make structure prediction one of its biggest strengths.

Specifically, I am writing this tutorial for a Natural Language Processing class at Georgia Tech, to ease into a problem set I wrote for the class on deep transition parsing.
The problem set uses some advanced techniques.  The intention of this tutorial is to cover the basics, so that students can focus on the more challenging aspects of the problem set.
The aim is to start with the basics and move up to linguistic structure prediction, which I feel is almost completely absent in other Pytorch tutorials.
The general deep learning basics have short expositions.  Topics more NLP-specific received more in-depth discussions, although I have referred to other sources when I felt a full description would be reinventing the wheel and take up too much space.

### Dependency Parsing Problem Set

As mentioned above, [here](https://github.com/jacobeisenstein/gt-nlp-class/tree/master/psets/ps4) is the problem set that goes through implementing
a high-performing dependency parser in Pytorch.  I wanted to add a link here since it might be useful, provided you ignore the things that were specific to the class.
A few notes:

* There is a lot of code, so the beginning of the problem set was mainly to get people familiar with the way my code represented the relevant data, and the interfaces you need to use.  The rest of the problem set is actually implementing components for the parser.  Since we hadn't done deep learning in the class before, I tried to provide an enormous amount of comments and hints when writing it.
* There is a unit test for every deliverable, which you can run with nosetests.
* Since we use this problem set in the class, please don't publically post solutions.
* The same repo has some notes that include a section on shift-reduce dependency parsing, if you are looking for a written source to complement the problem set.
* The link above might not work if it is taken down at the start of a new semester.

# References:
* I learned a lot about deep structure prediction at EMNLP 2016 from [this](https://github.com/clab/dynet_tutorial_examples) tutorial on [Dynet](http://dynet.readthedocs.io/en/latest/), given by Chris Dyer and Graham Neubig of CMU and Yoav Goldberg of Bar Ilan University.  Dynet is a great package, especially if you want to use C++ and avoid dynamic typing.  The final BiLSTM CRF exercise and the character-level features exercise are things I learned from this tutorial.
* A great book on structure prediction is [Linguistic Structure Prediction](https://www.amazon.com/Linguistic-Structure-Prediction-Synthesis-Technologies/dp/1608454053/ref=sr_1_1?ie=UTF8&qid=1489510387&sr=8-1&keywords=Linguistic+Structure+Prediction) by Noah Smith.  It doesn't use deep learning, but that is ok.
* The best deep learning book I am aware of is [Deep Learning](http://deeplearningbook.org), which is by some major contributors to the field and very comprehensive, although there is not an NLP focus.  It is free online, but worth having on your shelf.

# Exercises:
There are a few exercises in the tutorial, which are either to implement a popular model (CBOW) or augment one of my models.
The character-level features exercise especially is very non-trivial, but very useful (I can't quote the exact numbers, but I have run the experiment before and usually the character-level features increase accuracy 2-3%).
Since they aren't simple exercises, I will soon implement them myself and add them to the repo.

# Suggestions:
Please open a GitHub issue if you find any mistakes or think there is a particular model that would be useful to add.



Deep-Learning-for-NLP-Resources

List of resources to get started with Deep Learning for NLP. (Updated incrementally)
Deep Learning (general + NLP) links:

    https://www.youtube.com/playlist?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH : This lecture series has very good introduction to Neural Network and Deep Learning.

    https://www.coursera.org/course/neuralnets : This lecture series is from Geof Hinton. The concepts explained are bit abstract, concepts are hard to understand in first go. Generally people recommend these lectures as starting point but I am skeptical about it. I would suggest going through 1st one before this.

    https://www.youtube.com/playlist?list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu : Deep Learning Lectures from Oxford University

    https://www.iro.umontreal.ca/~lisa/pointeurs/TR1312.pdf : This is a short book on Deep Learning written by Yoshua Bengio. It deals with theoritical aspects related to Deep Architectures. Great book though.

    http://www.deeplearningbook.org/ : This web page has a book draft written by Yoshua Bengio and Ian Goodfellow. Later person is author of Theano library. This is holy bible on Deep Learning.

    http://cs231n.stanford.edu/ : Deep Learning for Vision by Stanford. Good lectures by Andrej Karpathy on introduction to DL (some initial lectures)

    http://videolectures.net/yoshua_bengio/ : Video Lectures By Yoshua Bengio on Theoritical Aspects of Deep Learning. They are counterparts of resource [4].

    http://videolectures.net/geoffrey_e_hinton/ : Video Lectures by the GodFather Geoffrey Hinton on introduction to Deep Learning and some advanced stuff too.

    https://github.com/ChristosChristofidis/awesome-deep-learning : Good collection of resources.

    http://deeplearning.net/reading-list/ : Reading resources

    http://www.cs.toronto.edu/~hinton/csc2515/deeprefs.html : Reading list by Hinton

    http://videolectures.net/mlss05us_lecun_ebmli/ : Intro to Energy based model by Yann Lecunn.

    http://videolectures.net/kdd2014_bengio_deep_learning/?q=ICLR# : Yoshua Bengio's lecture series recorded in KDD' 14.

    http://videolectures.net/nips09_collobert_weston_dlnl/ : Ronan Collobert lecture (it's quite old new, from 2008 but I think it is still useful).

    https://www.youtube.com/watch?v=eixGKz0Asr8 : Lecture series by Chris Manning and Richard Socher given at NAACL 2013

    https://www.youtube.com/watch?v=AmG4jzmBZ88 : Lecture series for DL4NLP with some practical guidelines.

    https://blog.wtf.sg/2014/08/24/nlp-with-neural-networks/ : Blogpost on some DL applications.

    http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html : Some useful tricks for training Neural Networks

    http://cs224d.stanford.edu/lectures/CS224d-Lecture11.pdf : Short notes on backprop and word embeddings

    http://cilvr.nyu.edu/doku.php?id=courses:deeplearning2014:start : A course by Yann Lecunn on Deep Learning taught at NYU.

    http://cs224d.stanford.edu/ : Course Specifically designed for DEEP LEARNING FOR NLP

    https://devblogs.nvidia.com/parallelforall/understanding-natural-language-deep-neural-networks-using-torch/#.VPYhS2vB09E.reddit : NLP using Torch

    http://www.kyunghyuncho.me/home/courses/ds-ga-3001-fall-2015 : Natural Language Understanding with Distributed Representations

    http://mlwave.com/kaggle-ensembling-guide/ : ENSEMBLING guide. Very useful for designing practical ML systems

    http://joanbruna.github.io/stat212b/ : TOPIC COURSE IN DEEP LEARNING by Joan Brune, UC Berkley Stats Department

    https://medium.com/@memoakten/selection-of-resources-to-learn-artificial-intelligence-machine-learning-statistical-inference-23bc56ba655#.s5kjy7bgo : LIST of Deep Learning Talk

Deep Learning for Information Retrieval Links:

There are two very good survey papers on using Deep Learning for Information Retrieval. There reference section in these articles is an exhaustive list (I think) for IR using DL.

    https://arxiv.org/abs/1611.03305 : Getting started with Neural Models for Semantic Matching in Web Search

    https://arxiv.org/pdf/1611.06792.pdf : Neural Information Retrieval: A Literature Review

    http://www.slideshare.net/BhaskarMitra3/neural-text-embeddings-for-information-retrieval-wsdm-2017 : WSDM'17 Tut. on Deep Learning for IR

Word Embeddings related articles

    https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html : Tensorflow tutorial on word2vec

    http://textminingonline.com/getting-started-with-word2vec-and-glove : Intro to word2vec and glove

    http://rare-technologies.com/deep-learning-with-word2vec-and-gensim/ : Getting starting with word2vec and gensim.

    http://www.lab41.org/anything2vec/ : Great explaination of word2vec and it's relation to neural networks

    http://www.offconvex.org/2015/12/12/word-embeddings-1/ : Intuition on word embedding methods

    http://www.offconvex.org/2016/02/14/word-embeddings-2/ : Explains the mathy stuff behind word2vec and glove (Also contains some links pointing to some other good articles on word2vec)

    http://textminingonline.com/getting-started-with-word2vec-and-glove-in-python : Getting started with glove and word2vec with python

    http://www.foldl.me/2014/glove-python/ : Glove implementation details in python

    http://videolectures.net/kdd2014_salakhutdinov_deep_learning/ : Tutorial by Ruslan

    http://www.openu.ac.il/iscol2015/downloads/ISCOL2015_submission25_e_2.pdf : Comparing various word embedding models

    http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf : Comparision between word2vec and glove

    https://levyomer.files.wordpress.com/2014/09/neural-word-embeddings-as-implicit-matrix-factorization.pdf : word2vec as matrix factorization

    http://research.microsoft.com/pubs/232372/CIKM14_tutorial_HeGaoDeng.pdf : Tutorial by Microsoft on DL for NLP at CIKM '14

    http://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/ : How backprop works in LSTM's (the so-called BPTT (back prop. through time)

RNN related stuff

    http://www.neutronest.moe/2015-11-15-LSTM-survey.html

    http://www.kdnuggets.com/2015/06/rnn-tutorial-sequence-learning-recurrent-neural-networks.html

    http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/ : Series of posts explaining RNN with some code

    http://colah.github.io/posts/2015-08-Understanding-LSTMs/ : Great post explaining LSTMs

    https://www.reddit.com/r/MachineLearning/comments/2zkb3b/lstm_a_search_space_odyssey_comparison_of_lstm/ : Comparision of various LSTM architectures

    http://www.fit.vutbr.cz/~imikolov/rnnlm/ : RNN based language modelling toolkit by Tomas Micholov

    http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf : A new technique in solving sequence tasks which I belive will be point of interest in few years : subword based language models. Usually good at handling OOV, spelling error problems

    https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition : Human activity recognition using TensorFlow on smartphone sensors dataset and an LSTM RNN (predict time series).

    https://github.com/guillaume-chevalier/seq2seq-signal-prediction : Learn and practice seq2seq in TensorFlow on time series data for signal prediction

Solving NLP tasks using Deep Learning

    http://eric-yuan.me/ner_1/ : Named Entity Recognition using CNN

    http://arxiv.org/pdf/1511.06388.pdf : Word Sense Disambiguation using Word Embeddings

    http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow : CNN for Text Classification

    http://research.microsoft.com/en-us/projects/dssm/ : Deep Learning Models for learning Semantic Representation of text(document, paragraph, phrase) which can be used to solve variety of tasks including Machine Translation, Document ranking for web search etc.

    http://www.aclweb.org/anthology/P15-1130 : Sentiment Analysis using RNN (LSTMs)

    http://ir.hit.edu.cn/~dytang/paper/emnlp2015/emnlp2015.pdf : Sentiment Analysis using Hierarchical RNN's (GRU)

    https://devblogs.nvidia.com/parallelforall/introduction-neural-machine-translation-with-gpus/ : Machine translation using RNN's

    http://neon.nervanasys.com/docs/latest/lstm.html : Practical example of using LSTM for sentiment analysis

    https://cs224d.stanford.edu/reports/HongJames.pdf : Again Sentiment Analysis using LSTMs

    arxiv.org/pdf/1412.5335 : ICLR '15 paper on using ensembles of NN + Generative models (Language model, Naive bayes) for solving Sentiment prediction task

    http://research.microsoft.com/pubs/214617/www2014_cdssm_p07.pdf : Extension of paper mentioned in [4] which used Convolution and max-pooling operations to learn low-dimensional semanti c representation of text

Optimization for Neural Networks

    http://cs231n.github.io/neural-networks-3/#update

    http://nptel.ac.in/courses/106108056/10 : JUMP TO SECTION : Uncontstrained optimization. Has tutorials on Non-convex optimization essential in deep Learning.

    http://online.stanford.edu/course/convex-optimization-winter-2014 : Has more convex optimization part, contains basics of Optimization

    http://videolectures.net/deeplearning2015_schmidt_smooth_finite/ : Deep Learning Summer School optimization lecture

Datasets

    https://bigquery.cloud.google.com/table/fh-bigquery:reddit_comments.2015_08?pli=1 : Reddit comments dataset

    https://code.google.com/archive/p/word2vec/ : Links to unlabelled english corpus

    http://github.com/brmson/dataset-sts : Variety of datasets wrapped in Python with focus on comparing two sentences, sample implementations of popular deep NN models in Keras

    http://www.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html : Conversation dataset (for learning seq2seq models possible leading to a chatbot kind of application)

    https://github.com/rkadlec/ubuntu-ranking-dataset-creator : Ubuntu Dialog Corpus 5.1 : http://arxiv.org/pdf/1506.08909v3.pdf : Accompanying paper for Ubuntu dataset

    http://www.aclweb.org/anthology/P12-2040 : Another Dialogue corpus

    http://www.lrec-conf.org/proceedings/lrec2012/pdf/1114_Paper.pdf : yet another dialogue corpus

    http://www.cs.technion.ac.il/~gabr/resources/data/ne_datasets.html : NER resources

    http://linguistics.cornell.edu/language-corpora : List of NLP resources

    https://github.com/aritter/twitter_nlp/blob/master/data/annotated/ner.txt : Annotated twitter corpus

    http://schwa.org/projects/resources/wiki/Wikiner

    https://www.aclweb.org/anthology/W/W10/W10-0712.pdf : Paper describing annotation process for NER on large email data (could not find any link, if anyone finds out please feel free to send a PR)

    http://www.cs.cmu.edu/~mgormley/papers/napoles+gormley+van-durme.naaclw.2012.pdf : Annotated gigawords

    http://jmcauley.ucsd.edu/data/amazon/ : Amazon review dataset (LARGE CORPUS)

    http://curtis.ml.cmu.edu/w/courses/index.php/Amazon_product_reviews_dataset : Amazon product review dataset (available only on request)

    http://times.cs.uiuc.edu/~wang296/Data/ : Amazon review dataset

    https://www.yelp.com/dataset_challenge : Yelp dataset (review + images)

Practical tools for Deep Learning

    Deep Learning libraries

    1.1. theano

    1.2. torch

    1.3. tensorflow

    1.4. keras

    1.5. lasagne

    1.6. blocks and fuel

    1.7. skflow

    1.8. scicuda

    (Automatic Differentiation tool in python)[https://github.com/HIPS/autograd]

    (Spearmint : Hyperparamter optimization using Bayesian optimization)

