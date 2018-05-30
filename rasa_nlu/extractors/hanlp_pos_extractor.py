from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import typing
from builtins import str
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

logger = logging.getLogger(__name__)


class HanlpPOSExtractor(EntityExtractor):
    name = "pos_hanlp"

    provides = ["entities"]

    language_list = ["zh"]

    defaults = {
        # by default all dimensions recognized by duckling are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }

    def __init__(self, component_config=None, ent_tagger=None):
        # type: (Dict[Text, Any],...) -> None

        super(HanlpPOSExtractor, self).__init__(component_config)

        self.ent_tagger = ent_tagger


    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> JiebaTokenizer

        from pyhanlp import HanLP as tokenizer
        component_conf = cfg.for_component(cls.name, cls.defaults)
        return HanlpPOSExtractor(component_conf, tokenizer)

    @classmethod
    def required_packages(cls):
        return ["pyhanlp"]

    def extract_entities(self, text):

        from pyhanlp import HanLP
        terms = HanLP.segment(text)
        running_offset = 0
        ents = []
        for term in terms:
            word_offset = text.index(term.word, running_offset)
            word_len = len(term.word)
            running_offset = word_offset + word_len
            dimensions = self.component_config["dimensions"]
            pos = str(term.nature)
            if pos in dimensions:
                ents.append({
                    "entity": pos,
                    "value": term.word,
                    "start": word_offset,
                    "end": running_offset,
                    "confidence": None,
                })
        return ents

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        ents = self.extract_entities(message.text)
        extracted = self.add_extractor_name(ents)
        #print(extracted)
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)


    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> HanlpPOSExtractor

        from pyhanlp import HanLP as extractor
        component_meta = model_metadata.for_component(cls.name, cls.defaults)
        return HanlpPOSExtractor(component_meta, extractor)
