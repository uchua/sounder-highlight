import re

from .textrank import TextRank4Sentences


def split_items(transcript: str) -> list:
    """
    Split transcript text into items
    :param str transcript: Transcript text as str
    :return: transcript split into items
    :rtype: list
    """
    return re.findall(r"[\w']+|[-.,!?;:]", transcript)


def process(payload_json: dict) -> dict:
    """
    Process payload JSON
    :param dict payload_json: Request payload JSON
    :return: dictionary with extracted highlight & start/end timestamps
    :rtype: dict
    """
    # get transcript and items from payload
    results = payload_json["results"]
    transcript = results["transcripts"][0]["transcript"]
    items = results["items"]

    # extract top sentence from transcript
    tr4s = TextRank4Sentences()
    tr4s.analyze(transcript)
    top_sentence = tr4s.get_top_sentences(1)[0]

    # get start and end timestamps from items
    match = next(re.finditer(top_sentence, transcript))
    start = match.start()
    stop = match.end()
    item_start = len(split_items(top_sentence[:start]))
    item_stop = len(split_items(top_sentence[:stop])) - 1

    # return top sentence and timestamps
    return {
        "highlight": top_sentence,
        "start_timestamp": items[item_start]["start_time"],
        "end_timestamp": items[item_stop]["end_time"],
    }
