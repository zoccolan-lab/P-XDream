
from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Tuple, Type
from collections import defaultdict
from dataclasses import dataclass, field

from tqdm import tqdm

from zdream.utils.io_ import read_txt, read_json
from zdream.utils.logger import Logger, SilentLogger
import pickle

@dataclass
class Word:
    '''
    This class represents a word in WordNet.
    '''

    # Attributes
    name  : str
    code  : str
    
    # Precomputed attributes. None if not computed
    _depth             :       int | None = None
    _children_codes    : List[str] | None = None
    _parents_codes     : List[str] | None = None
    _descendants_codes : List[str] | None = None
    _ancestors_codes   : List[str] | None = None
    
    # String representation
    def __str__(self)  -> str:  
        
        try:               depth       = self.depth 
        except ValueError: depth       = 'NC'
        
        try:               descendants = len(self.descendants_codes)
        except ValueError: descendants = 'NC'
        
        try:               ancestors   = len(self.ancestors_codes)
        except ValueError: ancestors   = 'NC'
        
        try:               children    = len(self.children_codes)
        except ValueError: children    = 'NC'
        
        try:               parents     = len(self.parents_codes)
        except ValueError: parents     = 'NC'
        
        return  f'{self.code} [{self.name}; '\
                f'depth: {depth}; '\
                f'children: {children}; '\
                f'parents: {parents}; '\
                f'descendants: {descendants}; '\
                f'ancestors: {ancestors}]'
                
    def __repr__(self) -> str:  return str(self)   
    
    # Hashable properties
    def __hash__(self)        -> int:  return hash(self.code)
    def __eq__  (self, other) -> bool: return self.code == other.code
    
    # Properties to check precomputed attributes
    
    @property
    def depth(self) -> int:
        if self._depth is not None: return self._depth
        raise ValueError(f'Depth not computed')
    
    @property
    def children_codes(self) -> List[str]:
        if self._children_codes is not None: return self._children_codes
        raise ValueError(f'Children not computed')
    
    @property
    def parents_codes(self) -> List[str]:
        if self._parents_codes is not None: return self._parents_codes
        raise ValueError(f'Parents not computed')
    
    @property
    def descendants_codes(self) -> List[str]:
        if self._descendants_codes is not None: return self._descendants_codes
        raise ValueError(f'Descendants not computed')
    
    @property
    def ancestors_codes(self) -> List[str]:
        if self._ancestors_codes is not None: return self._ancestors_codes
        raise ValueError(f'Ancestors not computed')
    
    

@dataclass
class ImageNetWord(Word):
    ''' 
    This class represents a word in ImageNet, 
    which is a subclass of Word with an additional id attribute.
    '''
    
    _id: int | None = None
    
    # String representation
    def __str__ (self) -> str: 
        
        try:               id = self.id
        except ValueError: id = 'NC'
        
        return super().__str__()[:-1] + f'; id: {id}]'
    def __repr__(self) -> str: return str(self)
    
    # Hashable properties
    def __hash__(self)        -> int:  return hash(self.code)
    def __eq__  (self, other) -> bool: return self.code == other.code
    
    @property
    def id(self) -> int:
        if self._id is not None: return self._id
        raise ValueError(f'ID not computed')
    

'''
Requires:
- words.txt        [https://github.com/innerlee/ImagenetSampling/blob/master/Imagenet/data/words.txt]
- wordnet.is_a.txt [https://github.com/innerlee/ImagenetSampling/blob/master/Imagenet/data/wordnet.is_a.txt]
'''
class WordNet:
    '''
    This class represents a WordNet object. It contains a list of words and their hierarchy.
    It provides methods for precomputing:
    - the depth of a word.
    - the children and parents of a word.
    - the ancestors and descendants of a word.

    Moreover it offers methods for:
    - retrieving the common ancestor of two words.
    - retrieving the distance between two words in the hierarchy tree.
    '''
    
    DUMP_NAME = 'words.pkl'
    
    # --- Instantiation ---
    
    def __init__(
        self, 
        wordnet_fp   : str, 
        hierarchy_fp : str,
        precompute   : bool = True,
        logger       : Logger = SilentLogger(),
    ) -> None:
        '''
        Create a WordNet object by loading words and hierarchy from files.
        
        - It creates mapping for getting the word given its code or name
        - It maintains the children and parents of each word
        
        It offers the possibility of precomputing each word attributes 
        by saving results in the object attributes.

        :param wordnet_fp: File path to the `wordnet.is_a.txt` file.
        :type wordnet_fp: str
        :param hierarchy_fp: File path to the `words.txt` file.
        :type hierarchy_fp: str
        :param precompute: Whether to precompute the attributes of each word.
        :type precompute: bool
        :param logger: A logger object.
        :type logger: Logger
        '''
        
        self._logger = logger
        
        # Load words and create mappings
        self._words        : List[Word]
        self._code_mapping : Dict[str, Word]
        self._name_mapping : Dict[str, Word]
        self._words, self._code_mapping, self._name_mapping = self._load_words(wordnet_fp=wordnet_fp)
        
        # Load Hierarchy couples
        self._children :  Dict[Word, List[Word]]
        self._parents  :  Dict[Word, List[Word]]
        self._children, self._parents = self._load_hierarchy(hierarchy_fp=hierarchy_fp)
        
        if precompute:
            
            # Precomputation
            # NOTE: This is done by calling methods for each word to save results in their fields
            #       Object attributes are used as cache for dynamic programming
            
            self._logger.info(mess='Precomputing Depth');       _ = [self.get_depth            (word)  for word in tqdm(self._words)]
            self._logger.info(mess='Precomputing Ancestors');   _ = [self.get_ancestors  (word)  for word in tqdm(self._words)]
            self._logger.info(mess='Precomputing Descendants'); _ = [self.get_descendants(word)  for word in tqdm(self._words)]
    
    @classmethod
    def from_precomputed(
        cls, 
        wordnet_fp: str, 
        hierarchy_fp: str,
        words_precomputed: str,
        logger: Logger = SilentLogger(),
    ) -> 'WordNet':
        '''
        Load a precomputed WordNet from a file.

        :param cls: The class of the WordNet.
        :param wordnet_fp: The file path to the WordNet data.
        :param hierarchy_fp: The file path to the hierarchy data.
        :param words_precomputed: The file path to the precomputed words data.
        :param logger: The logger to use for logging messages (default: SilentLogger()).
        :return: The loaded WordNet object.
        '''
        
        logger.info(f'Loading precomputed WordNet from {words_precomputed}')
        with open(words_precomputed, 'rb') as f:
            words = pickle.load(f)
            
        wordnet = cls(
            wordnet_fp=wordnet_fp, 
            hierarchy_fp=hierarchy_fp, 
            logger=logger,
            precompute=False
        )
        
        # Adjust mappings
        word_map = {w.code: w for w in words}
        
        wordnet._name_mapping = {name: word_map[word.code]                  for name, word     in wordnet._name_mapping.items()}
        wordnet._code_mapping = {code: word_map[word.code]                  for code, word     in wordnet._code_mapping.items()}
        wordnet._children     = {word: [word_map[c.code] for c in children] for word, children in wordnet._children    .items()}
        wordnet._parents      = {word: [word_map[p.code] for p in parents]  for word, parents  in wordnet._parents     .items()}
        
        wordnet._words = words
        
        return wordnet
    
    # --- String representation ---
    
    def __str__ (self) -> str: return f'WordNet[{len(self)} words]'
    def __repr__(self) -> str: return str(self)
    
    # --- Magic methods ---
    
    def __len__(self)  -> int:            return  len(self.words)
    def __iter__(self) -> Iterable[Word]: return iter(self.words)
    
    def __getitem__(self, key: str) -> Word: 
        if re.match(r'^n\d+$', key):  # matching code `n` followed by numeric
            return self._code_mapping[key]
        return self._name_mapping[key]
    
    # --- Properties ---
    
    @property
    def words(self) -> List[Word]: return self._words
    
    # --- Loading ---
    
    def _load_words(self, wordnet_fp: str) -> Tuple[List[Word], Dict[str, Word], Dict[str, Word]]:
        ''' 
        Load words from a file and create mappings.
        
        NOTE:   Since the same word can appear multiple times in the file,
                we append a number to the name of the word to avoid duplicates.
        '''
        
        # Load words from file
        word_lines = read_txt(wordnet_fp)
        
        words      = []
        name_map   = {}
        code_map   = {}
        duplicates = defaultdict(lambda: 0)
        
        # Create the word associated to each line and create mappings
        for word_line in word_lines:
            
            code, name = word_line.split('\t')
            
            duplicates[name] += 1
            
            # Append a number to the name to avoid duplicates
            if duplicates[name] > 1:
                name = f'{name}({duplicates[name]})'
            
            word = Word(name=name, code=code)
            
            # Update mappings
            words.append(word)
            name_map[name] = word
            code_map[code] = word  
            
        self._duplicates = duplicates
        
        return words, code_map, name_map

    def _load_hierarchy(self, hierarchy_fp: str) -> Tuple[Dict[Word, List[Word]], Dict[Word, List[Word]]]:
        ''' Load words hierarchy as a tuple (parent, children) '''
        
        # Load hierarchy from file
        hierarchy_lines = read_txt(hierarchy_fp)
        
        children = defaultdict(list)
        parents  = defaultdict(list)
        
        for hierarchy_line in hierarchy_lines:
            
            parent_code, child_code = hierarchy_line.split(' ')
            
            # Retrieve words objects
            parent = self[parent_code]
            child  = self[child_code]
            
            # Update children and parents mappings
            children[parent].append(child)
            parents [child ].append(parent)
            
        return children, parents
    
    # --- Precomputed attributes ---
    
    def codes_to_words(self, codes: List[str] ) -> List[Word]: return [self[code] for code in codes]
    def words_to_codes(self, words: List[Word]) -> List[str] : return [word.code  for word in words]
    
    def get_parents(self, word: Word) -> List[Word]: 
        ''' Returns list of words that are direct parents of a given word. '''
        
        # Check if the word is in the WordNet
        if word not in self.words: raise ValueError(f'Word {word} not in WordNet')
        
        # Return parents if already computed
        if word._parents_codes is not None: return self.codes_to_words(word._parents_codes)
        
        # In the case of the root there are no parents
        parents = self._parents.get(word, [])
        word._parents_codes = self.words_to_codes(parents)
        
        return parents
    
    def get_children(self, word: Word) -> List[Word]: 
        ''' Returns list of words that are direct children of a given word. '''
        
        # Check if the word is in the WordNet
        if word not in self.words: raise ValueError(f'Word {word} not in WordNet')
        
        # Return children if already computed
        if word._children_codes is not None: return self.codes_to_words(word._children_codes)
        
        # In the case of the leaves there are no children
        children = self._children.get(word, [])
        word._children_codes = self.words_to_codes(children)
        
        return children
    
    # Ancestors and descendants
    def get_descendants(self, word: Word) -> List[Word]:
        ''' Returns list of words that are descendants of a given word. '''
        
        # If the descendants are already computed, return them
        if word._descendants_codes is not None: return self.codes_to_words(word._descendants_codes)
        
        # Otherwise, the descendants are its children and the descendants of its children
        descendants = []
        for child_code in self.get_children(word): 
            descendants.extend(
                [child_code] +
                self.get_descendants(child_code)
            )
            
        # Remove duplicates
        descendants = list(set(descendants))
        
        word._descendants_codes = self.words_to_codes(descendants)
        return descendants
    
    def get_ancestors(self, word: Word) -> List[Word]:
        ''' Returns list of words that are ancestors of a given word. '''
        
        # If the ancestors are already computed, return them
        if word._ancestors_codes is not None: return self.codes_to_words(word._ancestors_codes)
        
        # Otherwise, the ancestors are the ancestors of its parents
        ancestors = []
        for parent in self.get_parents(word): 
            ancestors.extend(
                [parent] +
                self.get_ancestors(parent)
            )
        
        # Remove duplicates
        ancestors = list(set(ancestors))

        word._ancestors_codes = self.words_to_codes(ancestors)
        return ancestors
    
    
    def get_depth(self, word: Word) -> int:
        ''' 
        Recursively compute the depth of a word.
        When the depth is computed for the first time, it is stored in the word object.
        '''
        
        # If the depth is already computed, return it
        if word._depth is not None: return word._depth
        
        # If the word has no parents (corresponding to `entity`), its depth is 0
        parents = [self[parent.code] for parent in self.get_parents(word)]
        
        if len(parents) == 0:
            word._depth = 0
        
        # Otherwise, the depth is the maximum depth of its parents + 1
        else: 
            parent_depths = [self.get_depth(parent) for parent in parents]
            word._depth = max(parent_depths) + 1
        
        return word._depth
    
    # --- Common Ancestor ---
    
    def common_ancestor(self, word1: Word, word2: Word) -> Word:
        '''
        Returns the deepest common ancestor of two words.
        '''
        
        anc1 = self.get_ancestors(word1)
        anc2 = self.get_ancestors(word2)
        
        common = set(anc1).intersection(anc2)
        
        lowest = max(common, key=lambda x: x.depth)  # type: ignore
        
        return lowest

    def common_ancestor_distance(self, word1: Word, word2: Word) -> int:
        ''' Returns the distance between two words in the hierarchy tree.'''
        
        lowest = self.common_ancestor(word1, word2)
        
        return 2 * lowest.depth - (word1.depth +  word2.depth) 
    
    # --- Dump ---
    
    def dump_words(self, fp: str) -> None:
        ''' Dump the WordNet object to a file. '''
        
        # Dump self.words to pickle
        words_fp = os.path.join(fp, self.DUMP_NAME)
        
        self._logger.info(f'Dumping words to {words_fp}')
        with open(words_fp, 'wb') as f:
            pickle.dump(self.words, f)
    


'''
Requires:
- imagenet_class_index.json [https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json]
'''
class ImageNetWords:
    ''' Class representing the words in ImageNet. '''
    
    def __init__(
        self, 
        imagenet_fp: str,
        wordnet: WordNet,
    ) -> None:
        ''' 
        Initialize the ImageNetWords object by loading the words from a file.
        
        :param imagenet_fp: File path to the `imagenet_class_index.json` file.
        :type imagenet_fp: str
        :param wordnet: A WordNet object.
        :type wordnet: WordNet
        '''
        
        # Load ImageNet words from file
        imagenet_labels = read_json(imagenet_fp)
        
        self._words = []
        
        # Append ImageNet words to the list retrieving them from WordNet
        # NOTE: The word index in the list is the same for the output layer architecture
        for id, (code, _) in imagenet_labels.items():
            
            # Retrieve the word from WordNet
            word = wordnet[code]
            
            # Create an ImageNetWord object
            word_ = ImageNetWord(
                    name=word.name, 
                    code=word.code, 
                    _depth=word._depth,
                    _children_codes=word._children_codes,
                    _parents_codes=word._parents_codes,
                    _descendants_codes=word._descendants_codes,
                    _ancestors_codes=word._ancestors_codes,
                    _id=int(id)
                )
            
            self._words.append(word_)
            
    # String representation
    def __str__(self)  -> str: return f'ImageNetWords[{len(self.words)} words]'
    def __repr__(self) -> str: return str(self)
    
    # Magic methods
    def __len__(self)              -> int:                    return len(self.words)
    def __iter__(self)             -> Iterable[ImageNetWord]: return iter(self._words)
    def __getitem__(self, id: int) -> ImageNetWord:           return self.words[id]
    
    @property
    def words(self) -> List[ImageNetWord]: return self._words
    