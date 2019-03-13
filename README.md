# Sequence-to-Sequence (Seq2Seq)
Sequence-to-Sequence (Seq2Seq) is a general end-to-end framework which maps sequences in source domain to sequences in target domain. Seq2Seq model first reads the source sequence using an encoder to build vector-based 'understanding' representations, then passes them through a decoder to generate a target sequence, so it's also referred to as the encoder-decoder architecture. Many NLP tasks have benefited from Seq2Seq framework, including machine translation, text summarization and question answering. Seq2Seq models vary in term of their exact architecture, multi-layer bi-directional RNN (e.g. LSTM, GRU, etc.) for encoder and multi-layer uni-directional RNN with autoregressive decoding (e.g. greedy, beam search, etc.) for decoder are natural choices for vanilla Seq2Seq model. Attention mechanism is later introduced to allow decoder to pay 'attention' to relevant encoder outputs directly, which brings significant improvement on top of already successful vanilla Se2Seq model. Furthermore, 'Transformer', a novel architecture based on self-attention mechanism is proposed and has outperformed both recurrent and convolutional models in various tasks, although out-of-scope for this repo, I'd like to refer interested readers to [this post](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html) for more details

<img src="/seq2seq/document/seq2seq.abstract.architecture.png" width=500><br />
*Figure 1: Encoder-Decoder architecture of Seq2Seq model*

## Setting
* Python 3.6.6
* Tensorflow 1.12
* NumPy 1.15.4

## DataSet
* [IWSLT'15 English-Vietnamese](https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/) is a small dataset for English-Vietnamese translation task, it contains 133K training pairs and top 50K frequent words are used as vocabularies.
* [WMT'14 English-French](http://statmt.org/wmt14/translation-task.html) is large dataset for English-French translation task. The goals of this WMT shared translation task are, (1) to investigate the applicability of various MT techniques; (2) to examine special challenges in translating between English and French; (3) to create publicly available corpora for training and evaluating; (4) to generate up-to-date performance numbers as a basis of comparison in future research.
* [fastText](https://fasttext.cc/docs/en/crawl-vectors.html) is open source library for efficient text classification and representation learning. Pre-trained word vectors for 157 languages are distributed by fastText. These  models were trained on Common Crawl and Wikipedia dataset using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.
* [GloVe](https://nlp.stanford.edu/projects/glove/) is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

## Usage
* Run experiment
```bash
# run experiment in train mode
python seq2seq_run.py --mode train --config config/config_seq2seq_template.xxx.json
# run experiment in eval mode
python seq2seq_run.py --mode eval --config config/config_seq2seq_template.xxx.json
```
* Encode source
```bash
# encode source as CoVe vector
python seq2seq_run.py --mode encode --config config/config_seq2seq_template.xxx.json
```
* Search hyper-parameter
```bash
# random search hyper-parameters
python hparam_search.py --base-config config/config_seq2seq_template.xxx.json --search-config config/config_search_template.xxx.json --num-group 10 --random-seed 100 --output-dir config/search
```
* Visualize summary
```bash
# visualize summary via tensorboard
tensorboard --logdir=output
```

## Experiment
### Vanilla Seq2Seq
<img src="/seq2seq/document/seq2seq.vanilla.architecture.jpg" width=500><br />
*Figure 1: Vanilla Seq2Seq architecture*

### Attention-based Seq2Seq
<img src="/seq2seq/document/seq2seq.attention.architecture.jpg" width=500><br />
*Figure 2: Attention-based Seq2Seq architecture*

## Reference
* Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio. [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078) [2014]
* Ilya Sutskever, Oriol Vinyals, Quoc V. Le. [Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215) [2014]
* Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473) [2014]
* Minh-Thang Luong, Hieu Pham, Christopher D. Manning. [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025) [2015]
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. [Attention is all you need](https://arxiv.org/abs/1706.03762) [2017]
* Luong, Minh-Thang and Manning, Christopher D. [Stanford neural machine translation systems for spoken language domains](https://nlp.stanford.edu/pubs/luong-manning-iwslt15.pdf) [2015]
* Minh-Thang Luong. [Neural machine translation](https://github.com/lmthang/thesis) [2016]
* Thang Luong, Eugene Brevdo, Rui Zhao. [Neural machine translation (seq2seq) tutorial](https://github.com/tensorflow/nmt)
