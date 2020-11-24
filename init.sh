# downloading spacy resources for English
python -m spacy download en

# downloading nltk wordnet
python -m nltk.downloader wordnet

# installing context2vec
git clone https://github.com/orenmel/context2vec.git
cd context2vec
python setup.py install
cd ..
rm -rf context2vec

# installing word_forms
git clone https://github.com/gutfeeling/word_forms.git
cd word_forms
python setup.py install
cd ..
rm -rf word_forms
