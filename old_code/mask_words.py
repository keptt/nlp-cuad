import json
from transformers import pipeline, AutoTokenizer
import spacy

# Using spaCy for NER
nlp = spacy.load("en_core_web_sm")

tokenizer = AutoTokenizer.from_pretrained('roberta-base')

def mask_entities(context):
    doc = nlp(context)
    for ent in doc.ents:
        # if ent.label_ == "PERSON" or ent.label_ == "GPE":  # These are the entity types for proper nouns and geopolitical entities respectively
        context = context.replace(ent.text, tokenizer.mask_token)
    return context


if __name__ == "__main__":
    # Load json file
    with open('./data/test.json', 'r') as file:
        data = json.load(file)

    # Apply mask_entities function to each context
    for article in data['data']:
        for paragraph in article['paragraphs']:
            paragraph['context'] = mask_entities(paragraph['context'])

    # Save modified data to new json file
    with open('./data/test_masked.json', 'w') as file:
        json.dump(data, file)







