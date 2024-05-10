
from analysis.utils.wordnet import ImageNetWords, WordNet
from zdream.utils.io_ import save_json
from collections import Counter

WORDNET  = '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/words.txt'
HIERACHY = '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/wordnet.is_a.txt'
IMAGENET = '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/imagenet_class_index.json'
WORDS   =  '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/words.pkl'

NAMES = [
    'fish',
    'bird',
    'amphibian',
    'reptile, reptilian',
    'invertebrate',
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

if __name__ == '__main__':

    wordnet = WordNet.from_precomputed(
        wordnet_fp=WORDNET, 
        hierarchy_fp=HIERACHY, 
        words_precomputed=WORDS
    )

    imgnet = ImageNetWords(imagenet_fp=IMAGENET, wordnet=wordnet)

    out = {}

    for i in range(1000):
        
        word = imgnet[i]
        print()
        print(f"Step {i} - {word.name}")
        
        ancestors = [wordnet[a] for a in word.ancestors_codes]
        
        found = False
        for a in ancestors:
            if a.name in NAMES:
                print(f'> Automatically selected class: {a.name}')
                out[i] = a.name
                found = True
                break
        if found: continue
        
        print('No class found. Please select one from the list below:')
        for j, a in enumerate(ancestors):
            print(f'> {j}. {a.name}')
        k = int(input('Enter the number of the class: '))
        chosen_class = ancestors[k].name
        out[i] = chosen_class
        print(f'Chosen class: {chosen_class}')
        
    out = {k: [wordnet._words_from_name(v).code ,v] for k, v in out.items()}
    
    print(f"TOTAL CLASSES: {len(out)}")
    
    frequencies = Counter([a for _, a in out.values()])
    print(frequencies)
    
    
    save_json(out, 'sublcasses.json')

