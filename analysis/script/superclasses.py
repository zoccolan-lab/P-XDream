'''
This script allows the user to select the superclass for each output unit of the ImageNet dataset.
The superclass is selected from a list of 50 predefined superclasses.
The superclass labeling is intended to be used as a ground truth labeling for the ImageNet dataset.
'''

import os
from analysis.utils.settings import FILE_NAMES, WORDNET_DIR
from analysis.utils.wordnet import ImageNetWords, WordNet

from zdream.utils.io_ import save_json
from zdream.utils.logger import LoguruLogger

# ------------------------------------------- SETTINGS ---------------------------------------

SUPER_LABELS = [
    'mollusk, mollusc, shellfish',
    'arthropod',
    'invertebrate',
    'plant, flora, plant life',
    'fish',
    'bird',
    'amphibian',
    'reptile, reptilian',
    'aquatic mammal',
    'dog, domestic dog, Canis familiaris',
    'canine, canid',
    'cat, true cat',
    'feline, felid',
    'carnivore(2)',
    'rodent, gnawer',
    'primate',
    'mammal, mammalian',
    'clothing, article of clothing, vesture, wear, wearable, habiliment',
    'machine',
    'musical instrument, instrument',
    'ship',
    'wheeled vehicle',
    'aircraft',
    'structure, construction',
    'measuring instrument, measuring system, measuring device',
    'container',
    'weapon, arm, weapon system',
    'sports equipment',
    'implement',
    'covering(2)',
    'furnishing(2)',
    'piece of cloth, piece of material',
    'instrument',
    'restraint, constraint',
    'public transport',
    'electronic equipment',
    'toiletry, toilet articles',
    'vessel, watercraft',
    'mechanical device',
    'game equipment',
    'home appliance, household appliance',
    'device',
    'equipment',
    'instrumentality, instrumentation',
    'artifact, artefact',
    'dish(2)',
    'bread, breadstuff, staff of life',
    'vegetable, veggie, veg',
    'fruit(2)',
    'food, nutrient',
    'geological formation, formation',
    'fungus',
    'person, individual, someone, somebody, mortal, soul',
    'abstraction, abstract entity'
]

# ---------------------------------------------------------------------------------------------

if __name__ == '__main__':
    
    # Initialize logger
    logger = LoguruLogger(on_file=False)
    
    # WordNet paths
    words_fp             = os.path.join(WORDNET_DIR, FILE_NAMES['words'])
    hierarchy_fp         = os.path.join(WORDNET_DIR, FILE_NAMES['hierarchy'])
    words_precomputed_fp = os.path.join(WORDNET_DIR, FILE_NAMES['words_precoputed'])
    
    # A) Load WordNet with precomputed words if available
    if os.path.exists(words_precomputed_fp):
        
        logger.info(mess='Loading precomputed WordNet')
        
        wordnet = WordNet.from_precomputed(
            wordnet_fp=words_fp, 
            hierarchy_fp=hierarchy_fp, 
            words_precomputed=words_precomputed_fp,
            logger=logger
        )
    
    else:
        
        logger.info(mess=f'No precomputation found at {words_precomputed_fp}. Loading WordNet from scratch')
        
        wordnet = WordNet(
            wordnet_fp=words_fp, 
            hierarchy_fp=hierarchy_fp,
            logger=logger
        )

        # Dump precomputed words for future use
        wordnet.dump_words(fp=WORDNET_DIR)

    # Load ImageNet
    logger.info(mess='Loading ImageNet')
    inet_fp = os.path.join(WORDNET_DIR, FILE_NAMES['imagenet'])
    inet    = ImageNetWords(imagenet_fp=inet_fp, wordnet=wordnet)

    # Save superclasses for each output unit
    out = {}

    for word in inet:  # type: ignore
        
        logger.info(mess='')
        logger.info(mess=f"Step {word.id} - {word.name}")
        
        # Compute all word ancestors
        ancestors = [wordnet[a] for a in word.ancestors_codes]  # type: ignore
        
        
        # A) Check if the ancestors are superclasses list
        found = False
        
        for a in ancestors:
            if a.name in SUPER_LABELS:
                logger.info(f'> Automatically selected class: {a.name}')
                out[word.id] = a.name
                found = True
                break
        
        # If the class is found, go to the next one
        if found: continue
        
        # B) Ask the user to select one
        
        logger.info(mess='No class found. Please select one from the list below:')
        for j, a in enumerate(ancestors): logger.info(mess=f'> {j}. {a.name}')
        k = int(input('Enter the number of the class: '))
        chosen_class = ancestors[k].name
        out[word.id] = chosen_class
        logger.info(mess=f'Chosen class: {chosen_class}')
    
    # Log the total number of superclasses used
    logger.info(mess='')
    logger.info(mess=f"TOTAL CLASSES: {len(set(out.values()))}")
    
    # Save superclasses to file
    superclass_fp = os.path.join(WORDNET_DIR, FILE_NAMES['imagenet_super'])
    logger.info(mess=f'Saving superclasses to {superclass_fp}')
    save_json(out, superclass_fp)
    logger.info(mess='')
    
    logger.close()


