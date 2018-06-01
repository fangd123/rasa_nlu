from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings
import glob
import csv
from builtins import str
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

from rasa_nlu import utils
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.utils import write_json_to_file

ENTITY_SYNONYMS_FILE_NAME = "entity_synonyms.json"


class EntitySynonymDictMapper(EntityExtractor):
    name = "ner_synonyms_dict"

    provides = ["entities"]

    def __init__(self, component_config=None, synonyms=None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(EntitySynonymDictMapper, self).__init__(component_config)

        self.synonyms = synonyms if synonyms else {}

    # def train(self, training_data, config, **kwargs):
    #     # type: (TrainingData) -> None
    #
    #     for key, value in list(training_data.entity_synonyms.items()):
    #         self.add_entities_if_synonyms(key, value)
    #
    #     for example in training_data.entity_examples:
    #         for entity in example.get("entities", []):
    #             entity_val = example.text[entity["start"]:entity["end"]]
    #             self.add_entities_if_synonyms(entity_val,
    #                                           str(entity.get("value")))

    @classmethod
    def add_entities_if_synonyms(cls, synonyms,entity_a, entity_b):
        if entity_b is not None:
            original = utils.as_text_type(entity_a)
            replacement = utils.as_text_type(entity_b)

            if original != replacement:
                original = original.lower()
                if (original in cls.synonyms
                        and synonyms[original] != replacement):
                    warnings.warn("Found conflicting synonym definitions "
                                  "for {}. Overwriting target {} with {}. "
                                  "Check your training data and remove "
                                  "conflicting synonym definitions to "
                                  "prevent this from happening."
                                  "".format(repr(original),
                                            repr(synonyms[original]),
                                            repr(replacement)))

                synonyms[original] = replacement

    @classmethod
    def init_synonym_dict(cls,dict_config):
        synonym_dict = dict()
        if dict_config.get("synonym_dicts"):
            if os.path.isdir(dict_config.get("synonym_dicts")):
                parse_pattern = "{}/*"
            else:
                parse_pattern = "{}"

            path_user_dicts = glob.glob(parse_pattern.format(dict_config.get("synonym_dicts")))
            if len(path_user_dicts) > 0:
                for path_user_dict in path_user_dicts:
                    print("Loading Synonym Dictionary at " + str(path_user_dict))
                    with open(path_user_dict, 'r', encoding='utf-8', newline='\n') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cls.add_entities_if_synonyms(synonym_dict,row['gen_name'],row['name'])
            else:
                print("No Synonym Dictionary found")
        else:
            print("No Synonym Dictionary found")
        return synonym_dict

    @classmethod
    def create(cls, config):
        # type: (RasaNLUModelConfig) -> EntitySynonymDictMapper

        component_config = config.for_component(cls.name, cls.defaults)
        synonym_dict = cls.init_synonym_dict(component_config)
        return EntitySynonymDictMapper(component_config, synonym_dict)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated_entities = message.get("entities", [])[:]
        self.replace_synonyms(updated_entities)
        message.set("entities", updated_entities, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "synonym_dicts": self.component_config.get("synonym_dicts"),
        }

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[EntitySynonymDictMapper]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EntitySynonymDictMapper

        meta = model_metadata.for_component(cls.name)
        synonyms = cls.init_ner_dict(meta)

        return EntitySynonymDictMapper(meta, synonyms)

    def replace_synonyms(self, entities):
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])
            if entity_value.lower() in self.synonyms:
                entity["value"] = self.synonyms[entity_value.lower()]
                self.add_processor_name(entity)