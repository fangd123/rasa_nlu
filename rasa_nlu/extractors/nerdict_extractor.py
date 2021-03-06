from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Any, Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

import os
import glob
import logging
from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa_nlu.tokenizers.hanlp_tokenizer import HanlpTokenizer

logger = logging.getLogger(__name__)
import csv


## TODO: 强烈依赖分词结果，无法做到跨多个词的NER标记
## TODO: 需要和NER算法提取进行区分

class NerdictExtractor(EntityExtractor):
    """Adds entity normalization by analyzing found entities and
    transforming them into regular formats."""

    name = "ner_dict"

    provides = ["entities"]

    requires = ["tokens"]

    def __init__(self, component_config=None, ner_dict=None):
        # type: (Dict[Text, Any],list ) -> None

        super(NerdictExtractor, self).__init__(component_config)
        self.ner_dict = ner_dict

    def extract_entities(self, text, tokens):
        ents = []
        tokens_strs = [token.text for token in tokens]
        if self.ner_dict:
            for token in tokens_strs:
                for ner in self.ner_dict:
                    if token.text == ner[0]:
                        entity = {"start": token.offset,
                                  "end": token.end,
                                  "value": token.text,
                                  "confidence": None,
                                  "entity": ner[1]}
                        ents.append(entity)

        return ents

    @classmethod
    def init_ner_dict(cls, dict_config):
        ner_dict = list()
        if dict_config.get("ner_dicts"):
            if os.path.isdir(dict_config.get("ner_dicts")):
                parse_pattern = "{}/*"
            else:
                parse_pattern = "{}"

            path_user_dicts = glob.glob(parse_pattern.format(dict_config.get("ner_dicts")))
            if len(path_user_dicts) > 0:
                for path_user_dict in path_user_dicts:
                    print("Loading NER Dictionary at " + str(path_user_dict))
                    with open(path_user_dict, 'r') as f:
                        reader = csv.reader(f)
                        ner_dict = list(reader)
            else:
                print("No NER Dictionary found")
        else:
            print("No NER Dictionary found")
        return ner_dict

    @classmethod
    def create(cls, config):
        # type: (RasaNLUModelConfig) -> NerdictExtractor

        component_config = config.for_component(cls.name, cls.defaults)
        ner_dict = cls.init_ner_dict(component_config)
        return NerdictExtractor(component_config, ner_dict)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        return None

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if self.ner_dict is None:
            raise Exception("无法进行NER词典提取. "
                            "缺少NER词典.")

        # matches = self.parse(message.text)



        ents = self.extract_entities(message.text, message.get("tokens"))
        extracted = self.add_extractor_name(ents)

        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def convert_format_to_rasa(self,token_strs):
        extracted = []
        for token in token_strs:
            for ner in self.ner_dict:
                if token.text == ner[0]:
                    entity = {"start": token.offset,
                              "end": token.end,
                              "value": token.text,
                              "confidence": None,
                              "entity": ner[1]}
                    extracted.append(entity)

        # for match in matches:
        #     value = match['cleaned_name']
        #     entity = {"start": 0,
        #               "end": len(value) - 1,
        #               "text": value,
        #               "value": value,
        #               "origin": match['From'],
        #               "confidence": 1.0,
        #               "entity": match['domain']}
        return extracted

    def parse(self, text):
        matches = []
        for ner in self.ner_dict:
            if text == ner['cleaned_name'] and ner['直达'] == 'T':
                matches.append(ner)
        return matches

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[NerdictExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> NerdictExtractor

        if cached_component:
            ner_dict = cached_component.ner_dict
        else:
            component_meta = model_metadata.for_component(cls.name, cls.defaults)
            ner_dict = cls.init_ner_dict(component_meta)

        component_config = model_metadata.for_component(cls.name)
        return cls(component_config, ner_dict)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "ner_dicts": self.component_config.get("ner_dicts"),
        }
