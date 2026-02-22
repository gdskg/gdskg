import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Set
import re
import collections
import functools

class KeywordExtractor:
    """
    Analyzes text to extract and lemmatize nouns, used for keyword indexing.
    """

    def __init__(self):
        """
        Initialize the KeywordExtractor, ensuring all necessary NLTK corpora and 
        tokenizers are downloaded and ready for use.
        """
        # Initialize NLTK components, downloading if necessary
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        try:
            self.lemmatizer.lemmatize('a')
            nltk.pos_tag(['a'])
        except Exception:
            pass
        
        # Add programming language reserved keywords and common noise
        self.stop_words.update({
            'use', 'using', 'used', 'add', 'added', 'adding', 'remove', 'removed', 'removing',
            'fix', 'fixed', 'fixing', 'update', 'updated', 'updating', 'feature', 'feat',
            'bug', 'issue', 'initial', 'commit', 'repo', 'repository', 'file', 'files',
            'test', 'tests', 'testing', 'code', 'change', 'changes', 'changed',
            # Reserved words from common languages (Python, JS, C++, etc.)
            'class', 'def', 'function', 'async', 'await', 'return', 'import', 'from',
            'while', 'for', 'true', 'false', 'none', 'null', 'void', 'public', 'private',
            'protected', 'static', 'const', 'var', 'let', 'type', 'interface', 'struct',
            'enum', 'match', 'case', 'switch', 'break', 'continue', 'yield', 'lambda',
            'try', 'except', 'catch', 'finally', 'raise', 'throw', 'assert', 'with',
            'global', 'nonlocal', 'del', 'pass', 'yield', 'none', 'self', 'cls', 'super'
        })

    _URL_REGEX = re.compile(r'https?://\S+|www\.\S+')
    _NON_ALPHA_REGEX = re.compile(r'[^a-zA-Z\s]')
    _CAMEL_CASE_REGEX = re.compile(r'([a-z])([A-Z])')

    @functools.lru_cache(maxsize=100000)
    def extract_keywords(self, text: str, is_code: bool = False) -> List[str]:
        """
        Extracts nouns from text, lemmatizes them, and filters out stop words.
        
        Args:
            text (str): The text to analyze.
            is_code (bool): Whether the text is a code identifier (needs splitting).
            
        Returns:
            List[str]: A list of lemmatized nouns.
        """
        if not text:
            return []

        if is_code:
            # Split camelCase and snake_case
            text = self._CAMEL_CASE_REGEX.sub(r'\1 \2', text)
            text = text.replace('_', ' ')
            text = text.replace('-', ' ')

        # Simple cleaning
        text = text.lower()
        # Remove URLs
        text = self._URL_REGEX.sub('', text)
        # Remove non-alphanumeric characters but keep spaces
        text = self._NON_ALPHA_REGEX.sub(' ', text)
        
        tokens = text.split() if is_code else word_tokenize(text)
        
        if is_code:
            keywords = []
            for word in tokens:
                if word not in self.stop_words and len(word) > 2:
                    lemma = self.lemmatizer.lemmatize(word)
                    if lemma not in self.stop_words and len(lemma) > 2:
                         keywords.append(lemma)
            return keywords

        # POS Tagging for non-code text (e.g. commit messages)
        tagged = nltk.pos_tag(tokens)
        
        keywords = []
        for word, tag in tagged:
            # We are interested in nouns (NN, NNS, NNP, NNPS)
            if tag.startswith('NN'):
                if word not in self.stop_words and len(word) > 2:
                    lemma = self.lemmatizer.lemmatize(word)
                    if lemma not in self.stop_words and len(lemma) > 2:
                        keywords.append(lemma)
        
        return keywords

    def get_major_terms(self, term_counts: collections.Counter, top_n: int = 50, min_freq: int = 2) -> List[str]:
        """
        Identify the most frequent terms from a collection of counts.
        
        Args:
            term_counts (collections.Counter): A counter object containing term frequencies.
            top_n (int): The maximum number of top terms to return. Defaults to 50.
            min_freq (int): The minimum frequency required for a term to be returned. Defaults to 2.
            
        Returns:
            List[str]: A list of the most frequent terms that meet the criteria.
        """
        return [term for term, count in term_counts.most_common(top_n) if count >= min_freq]
