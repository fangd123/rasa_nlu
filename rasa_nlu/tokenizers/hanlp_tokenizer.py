from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import os
import glob
import logging
logger = logging.getLogger(__name__)

class HanlpTokenizer(Tokenizer, Component):
    name = "tokenizer_hanlp"

    provides = ["tokens"]

    language_list = ["zh"]

    def __init__(self,
                 component_config=None,  # type: Dict[Text, Any]
                 tokenizer=None
                 ):
        # type: (...) -> None

        super(HanlpTokenizer, self).__init__(component_config)

        self.tokenizer = tokenizer

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> HanlpTokenizer

        from pyhanlp import HanLP as tokenizer
        component_conf = cfg.for_component(cls.name, cls.defaults)
        tokenizer = cls.init_hanlp(tokenizer, component_conf)

        return HanlpTokenizer(component_conf, tokenizer)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> HanlpTokenizer

        from pyhanlp import HanLP as tokenizer
        component_meta = model_metadata.for_component(cls.name, cls.defaults)
        tokenizer = cls.init_hanlp(tokenizer, component_meta)

        return HanlpTokenizer(component_meta, tokenizer)


    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["pyhanlp"]



    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("tokens", self.tokenize(example.text))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("tokens", self.tokenize(message.text))

    def tokenize(self, text):
        # type: (Text) -> List[Token]
        from pyhanlp import HanLP
        terms = HanLP.segment(text)
        running_offset = 0
        tokens = []
        for term in terms:
            word_offset = text.index(term.word, running_offset)
            word_len = len(term.word)
            running_offset = word_offset + word_len
            tokens.append(Token(term.word, word_offset))
        logging.debug(terms)
        return tokens


    @classmethod
    def init_hanlp(cls, tokenizer, dict_config):


        if dict_config.get("user_dicts"):
            if os.path.isdir(dict_config.get("user_dicts")):
                parse_pattern = "{}/*"
            else:
                parse_pattern = "{}"

            path_user_dicts = glob.glob(parse_pattern.format(dict_config.get("user_dicts")))
            tokenizer = cls.set_user_dicts(tokenizer, path_user_dicts)
        else:
            logger.info("No Hanlp User Dictionary found")

        return tokenizer

    @staticmethod
    def set_user_dicts(tokenizer, path_user_dicts):
        from pyhanlp import JClass
        CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")

        if len(path_user_dicts) > 0:
            for path_user_dict in path_user_dicts:
                logger.info("Loading Hanlp User Dictionary at " + str(path_user_dict))
                with open(path_user_dict, 'r', encoding='utf-8') as f:
                    for word in f.readlines():
                        CustomDictionary.add(word.strip())  # 动态增加
        else:
            logger.info("No Hanlp User Dictionary found")

        return tokenizer

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "user_dicts": self.component_config.get("user_dicts"),
        }