
from analysis.utils.wordnet import ImageNetWords, WordNet
from zdream.utils.io_ import save_json

WORDNET  = '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/words.txt'
HIERACHY = '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/wordnet.is_a.txt'
IMAGENET = '/home/sebaq/Documents/GitHub/ZXDREAM/data/wordnet/imagenet_class_index.json'

wordnet = WordNet(wordnet_fp=WORDNET, hierarchy_fp=HIERACHY)

imgnet = ImageNetWords(imagenet_fp=IMAGENET, wordnet=wordnet)

names = [
    'furnishing',
    'electronic equipment',
    'game equipment',
    'clothing, article of clothing, vesture, wear, wearable, habiliment',
    'snake, serpent, ophidian',
    'dog, domestic dog, Canis familiaris',
    'insect',
    'reptile, reptilian',
    'mechanical device',
    'covering',
    'fruit',
    'primate',
    'bird',
    'musical instrument, instrument',
    'measuring instrument, measuring system, measuring device',
    'weapon, arm, weapon system',
    'implement',
    'instrument',
    'equipment',
    'machine',
    'device',
    'public transport',
    'toiletry, toilet articles',
    'artifact, artefact',
    'fish',
    'electrical device',
    'vehicle',
    'crocodilian reptile, crocodilian',
    'container',
    'fungus',
    'arthropod',
    'piece of cloth, piece of material',
    'building, edifice',
    'reptile, reptilian',
    'cat, true cat',
    'invertebrate',
    'home appliance, household appliance',
    'communication',
    'geological formation, formation',
    'structure, construction',
    'vegetable, veggie, veg',
    'mollusk, mollusc, shellfish',
    'aquatic mammal',
    'covering',
    'coral',
    'amphibian',
    'adornment',
    'fabric, cloth, material, textile',
    'person, individual, someone, somebody, mortal, soul',
    'surface',
    'material, stuff',
    'plant, flora, plant life',
    'canine, canid',
    'carnivore',
    'bread, breadstuff, staff of life',
    'food, nutrient',
    'mammal, mammalian'
]

out = {}

for i in range(1000):
    
    word = imgnet[i]
    print()
    print(f"Step {i} - {word.name}")
    
    ancestors = wordnet._get_ancestors(word)[::-1]
    
    found = False
    for a in ancestors:
        if a.name in names:
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
    
save_json(out, 'sublcasses.json')

