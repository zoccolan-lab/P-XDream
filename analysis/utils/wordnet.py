
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
    depth       :        int | None = None
    children    : List[Word] | None = None
    parents     : List[Word] | None = None
    descendants : List[Word] | None = None
    ancestors   : List[Word] | None = None
    
    # String representation
    def __str__(self)  -> str:  
        
        depth       = self.depth       if self.depth       is not None else 'NC'
        descendants = len(self.descendants) if self.descendants is not None else 'NC'
        ancestors   = len(self.ancestors  ) if self.ancestors   is not None else 'NC'
        children    = len(self.children   ) if self.children    is not None else 'NC'
        parents     = len(self.parents    ) if self.parents     is not None else 'NC'
        
        return  f'{self.code} [{self.name}; depth: {depth}; '\
                f'children: {children}; parents: {parents}; '\
                f'descendants: {descendants}; ancestors: {ancestors}'
                
    def __repr__(self) -> str:  return str(self)   
    
    # Hashable properties
    def __hash__(self)        -> int:  return hash(self.code)
    def __eq__  (self, other) -> bool: return self.code == other.code


@dataclass
class ImageNetWord(Word):
    
    id: int | None = None
    
    # String representation
    def __str__ (self) -> str: return super().__str__()[:-1] + f'; id: {self.id}]'
    def __repr__(self) -> str: return str(self)

'''
Requires:
- words.txt        [https://github.com/kokil01/wordnet-data-files/blob/master/words.txt]
- wordnet.is_a.txt [https://github.com/innerlee/ImagenetSampling/blob/master/Imagenet/data/wordnet.is_a.txt]
'''
class WordNet:
    
    def __init__(
        self, 
        wordnet_fp: str, 
        hierarchy_fp: str,
        logger: Logger = SilentLogger(),
        precompute: bool = True
    ) -> None:
        '''
        Create a WordNet object by loading words and hierarchy from files.
        - It creates mapping for getting the word given its code or name
        - It maintains the children and parents of each word
        - It computes the depth of each word

        :param wordnet_fp: File path to the `wordnet.is_a.txt` file.
        :type wordnet_fp: str
        :param hierarchy_fp: File path to the `words.txt` file.
        :type hierarchy_fp: str
        :param logger: A logger object.
        :type logger: Logger
        '''
        
        self._logger = logger
        
        # Load words and create mappings
        self._words:    List[Word]
        self._code_map: Dict[str, Word]
        self._name_map: Dict[str, Word]
        
        self._words, self._code_mapping, self._name_mapping = self._load_words(wordnet_fp=wordnet_fp)
        
        # Load Hierarchy couples
        self._children: Dict[Word, List[Word]]
        self._parents:  Dict[Word, List[Word]]
        
        self._children, self._parents = self._load_hierarchy(hierarchy_fp=hierarchy_fp)
        
        if precompute:
            # Precomputation
            # NOTE: This is done by calling methods for each word to save results in their fields
            #       Object attributes are used as cache for dynamic programming
            self._logger.info(mess='Precomputing Depth');       _ = [self._get_depth      (word)  for word in tqdm(self._words)]
            self._logger.info(mess='Precomputing Ancestors');   _ = [self._get_ancestors  (word)  for word in tqdm(self._words)]
            self._logger.info(mess='Precomputing Descendants'); _ = [self._get_descendants(word)  for word in tqdm(self._words)]

            self._logger.info(mess='Sorting Ancestors and Descendants by depth')
            for word in self._words:
                word.ancestors   = sorted(word.ancestors  , key=lambda x: x.depth, reverse=True) # type: ignore
                word.descendants = sorted(word.descendants, key=lambda x: x.depth, reverse=True) # type: ignore
    
    # String representation
    def __str__ (self) -> str: return f'WordNet[{len(self)} words]'
    def __repr__(self) -> str: return str(self)
    
    # Magic methods
    def __len__(self)  -> int:            return  len(self.words)
    def __iter__(self) -> Iterable[Word]: return iter(self.words)
    
    @property
    def words(self) -> List[Word]: return self._words
    
    # Name code mapping
    def _words_from_code(self, code: str) -> Word: return self._code_mapping[code]
    def _words_from_name(self, name: str) -> Word: return self._name_mapping[name]
    
    def __getitem__(self, key: str) -> Word: 
        if re.match(r'^n\d+$', key):  # matching code `n` followed by numeric
            return self._words_from_code(code=key)
        return self._words_from_name(name=key)
    
    # Parents and Children
    def _get_parents(self, word: Word) -> List[Word]: 
        ''' Returns list of words that are direct parents of a given word. '''
        
        #print(f'Computing parents of {word.name}')
        
        if word not in self.words: raise ValueError(f'Word {word} not in WordNet')
        
        # Return parents if already computed
        if word.parents is not None: return word.parents
        
        # In the case of the root, there are no parents
        word.parents = self._parents.get(word, [])
        
        return word.parents
    
    def _get_children(self, word: Word) -> List[Word]: 
        ''' Returns list of words that are direct children of a given word. '''
        
        #print(f'Computing children of {word.name}')
        
        if word not in self.words: raise ValueError(f'Word {word} not in WordNet')
        
        if word.children is not None: return word.children
        
        word.children = self._children.get(word, [])
        
        return word.children
    
    # Ancestors and descendants
    def _get_descendants(self, word: Word) -> List[Word]:
        ''' Returns list of words that are descendants of a given word. '''
        
        #print(f'Computing descendants of {word.name}')
        
        # If the descendants are already computed, return them
        if word.descendants is not None: return word.descendants
        
        # Otherwise, the descendants are the descendants of its children
        descendants = self._get_children(word)
        for child in self._get_children(word): descendants.extend(self._get_descendants(child))
        
        word.descendants = list(set(descendants))
        return word.descendants
    
    def _get_ancestors(self, word: Word) -> List[Word]:
        ''' Returns list of words that are ancestors of a given word. '''
        
        #print(f'Computing ancestors of {word.name}')
        
        # If the ancestors are already computed, return them
        if word.ancestors is not None: return word.ancestors
        
        # Otherwise, the ancestors are the ancestors of its parents
        ancestors = self._get_parents(word)
        for parent in self._get_parents(word): ancestors.extend(self._get_ancestors(parent))
        
        word.ancestors = list(set(ancestors))
        return word.ancestors
    
    # Depth
    
    def _get_depth(self, word: Word) -> int:
        ''' 
        Recursively compute the depth of a word.
        When the depth is computed for the first time, it is stored in the word object.
        '''
        
        #print(f'Computing depth of {word.name}')
        
        # If the depth is already computed, return it
        if word.depth is not None: return word.depth
        
        # If the word has no parents (corresponding to `entity`), its depth is 0
        if len(self._get_parents(word)) == 0:
            word.depth = 0
            
        # Otherwise, the depth is the maximum depth of its parents + 1
        else: 
            parent_depths = [self._get_depth(parent) for parent in self._parents[word]]
            word.depth = max(parent_depths) + 1
        
        return word.depth
    
    # Loading
    
    def _load_words(self, wordnet_fp: str) -> Tuple[List[Word], Dict[str, Word], Dict[str, Word]]:
        ''' Load words from a file and create mappings.'''
        
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
            
            if duplicates[name] > 1:
                a = 1
                #name = f'{name} ({duplicates[name]})'
            
            word = Word(name=name, code=code)
            
            words.append(word)
            name_map[name] = word
            code_map[code] = word  
            
        print(duplicates)
        
        return words, code_map, name_map
    
    # Loading files
    
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
    
    # Common Ancestor
    
    def common_ancestor(self, word1: Word, word2: Word) -> Word:
        
        anc1 = self._get_ancestors(word1)
        anc2 = self._get_ancestors(word2)
        
        common = set(anc1).intersection(anc2)
        
        lowest = max(common, key=lambda x: x.depth)  # type: ignore
        
        return lowest

    def common_ancestor_distance(self, word1, word2) -> int:
        
        lowest = self.common_ancestor(word1, word2)
        
        return max(word1.depth, word2.depth) - lowest.depth
    
    # Dump
    def dump_words(self, fp: str) -> None:
        ''' Dump the WordNet object to a file. '''
        
        # Dump self.words to pickle
        words_fp = os.path.join(fp, 'words.pickle')
        self._logger.info(f'Dumping words to {words_fp}')
        with open(fp, 'wb') as f:
            pickle.dump(self.words, f)
    
    @classmethod
    def from_precomputed(
        cls, 
        wordnet_fp: str, 
        hierarchy_fp: str,
        words_precomputed: str,
        logger: Logger = SilentLogger(),
    ) -> 'WordNet':
        
        with open(words_precomputed, 'rb') as f:
            words = pickle.load(f)
            
        wordnet = cls(
            wordnet_fp=wordnet_fp, 
            hierarchy_fp=hierarchy_fp, 
            logger=logger,
            precompute=False
        )
        
        setattr(wordnet, '_words', words)
        
        return wordnet


'''
Requires:
- imagenet_class_index.json [https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json]
'''
class ImageNetWords:
    ''' Classe representing the words in ImageNet. '''
    
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
            word = wordnet._words_from_code(code)
            
            # Create an ImageNetWord object
            word_ = ImageNetWord(
                    name=word.name, 
                    code=word.code, 
                    depth=word.depth,
                    children=word.children,
                    parents=word.parents,
                    descendants=word.descendants,
                    ancestors=word.ancestors,
                    id=int(id)
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
    