from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import logging

import typing
from typing import Any, Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from time_extractor.time_extractor import TimeExtractor


def extract_value(match):
    if match["value"].get("type") == "interval":
        value = {"to": match["value"].get("to", {}).get("value"),
                 "from": match["value"].get("from", {}).get("value")}
    else:
        value = match["value"].get("value")

    return value


def filter_irrelevant_matches(matches, requested_dimensions):
    """Only return dimensions the user configured"""

    if requested_dimensions:
        return [match
                for match in matches
                if match["dim"] in requested_dimensions]
    else:
        return matches


def current_datetime_str():
    current_time = datetime.datetime.utcnow()
    return current_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')


class LingExtractor(EntityExtractor):
    """Adds entity normalization by analyzing found entities and
    transforming them into regular formats."""

    name = "ling_extractor"

    provides = ["entities"]

    defaults = {
        # by default all dimensions recognized by duckling are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": 'time'
    }

    # @staticmethod
    # def available_dimensions():
    #     from duckling.dim import Dim
    #     return [m[1]
    #             for m in getmembers(Dim)
    #             if not m[0].startswith("__") and not m[0].endswith("__")]

    def __init__(self, component_config=None, ling=None):
        # type: (Dict[Text, Any], TimeExtractor) -> None
        from time_extractor.time_extractor import TimeExtractor
        super(LingExtractor, self).__init__(component_config)
        self.time_extractor = TimeExtractor()

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["time_extractor"]

    @classmethod
    def create(cls, config):
        # type: (RasaNLUModelConfig) -> LingExtractor
        from time_extractor.time_extractor import TimeExtractor
        component_config = config.for_component(cls.name, cls.defaults)
        wrapper = TimeExtractor()
        return LingExtractor(component_config, wrapper)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        return None

    def convert_ling_format_to_rasa(self,matches):
        import time_extractor
        extracted = []

        for key in matches:
            if key == 'period_time':
                for item in matches[key]:
                    nlu_time = {}
                    std_time = self.time_extractor.getStdTimePeriod(item[0], 'next')
                    time_type = 'delay_time'
                    nlu_time['start_time'] = time_extractor.time_extractor.formatting(std_time['start_time'])
                    nlu_time['end_time'] = time_extractor.time_extractor.formatting(std_time['end_time'])
                    entity = {"start": item[1],
                              "end": item[2],
                              "text": item[0],
                              "value": [nlu_time],
                              "confidence": 1.0,
                              "entity": time_type}
                    extracted.append(entity)

            elif key == 'point_time':
                for item in matches[key]:
                    nlu_time = {}
                    time_type = 'on_time'
                    std_time = self.time_extractor.getStdTimePoint(item[0], fuzzy=True)
                    logging.debug(std_time)
                    #std_time = std_time['start_time']
                    #timestr = '%04d-%02d-%02d %02d:%02d:%02d' % (
                    #    std_time['y'], std_time['m'], std_time['d'], std_time['H'], std_time['M'], std_time['S'])
                    nlu_time['start_time'] = time_extractor.time_extractor.formatting(std_time['start_time'])
                    nlu_time['end_time'] = time_extractor.time_extractor.formatting(std_time['end_time'])

                    entity = {"start": item[1],
                              "end": item[2],
                              "text": item[0],
                              "value": [nlu_time],
                              "confidence": 1.0,
                              #"additional_info": match["value"],
                              "entity": time_type}
                    extracted.append(entity)

        return extracted

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if self.time_extractor is None:
            return

        ref_time = message.time

        try:
            matches = self.time_extractor.extractTime(message.text)
        except Exception as e:
            logging.debug("Invalid time extractor parse. Error {}".format(e))
            matches = []

        extracted = self.convert_ling_format_to_rasa(matches)
        extracted = self.add_extractor_name(extracted)

        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[LingExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> LingExtractor

        from time_extractor.time_extractor import TimeExtractor
        wrapper = TimeExtractor()
        component_config = model_metadata.for_component(cls.name)
        return cls(component_config, wrapper)
