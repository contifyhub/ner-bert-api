import logging
from datetime import datetime
from functools import lru_cache
from logging import config as logger_config
import traceback
import torch

import uvicorn
import spacy
from spacy.tokens import Doc
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
import os
import secrets
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import config
from serializers import NerText

app = FastAPI()
security = HTTPBasic()
ROOT_DIR = os.path.join(".")


@lru_cache()
def get_settings():
    return config.MlApiSettings()


@lru_cache()
def get_spacy_pipeline():
    return spacy.load('en_core_web_trf', disable=[
        'tagger', 'parser', 'attribute_ruler', 'lemmatizer'
    ])


settings = get_settings()
logger_config.dictConfig(settings.LOGGING)

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    spacy.require_gpu()
    logger.info("Running on GPU")
else:
    spacy.require_cpu()
    logger.warning("Running on CPU")

nlp = get_spacy_pipeline()


def is_authenticated_user(
        credentials: HTTPBasicCredentials = Depends(security),
        settings: config.MlApiSettings = Depends(get_settings)):
    correct_username = secrets.compare_digest(
        credentials.username, settings.BERT_NER_USERNAME
    )
    correct_password = secrets.compare_digest(
        credentials.password, settings.BERT_NER_PASSWORD
    )
    if not (correct_username and correct_password):
        logger.info(
            f"Authentication Failed: Incorrect: {credentials.username},"
            f" username or Password {credentials.password}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


@app.post('/predict/ner/')
async def predict_ner(story: NerText,
                      auth_status: int = Depends(is_authenticated_user)):
    """This api is used to tag ner from Text.

    params: story: ArticleText
    Return: Tagged Entities
    """
    try:
        story_tuple = []
        data = story.dict()
        now = datetime.now()
        for story_val in data['text']:
            story_tuple.append(tuple(story_val))
        if not Doc.has_extension("story_id"):
            Doc.set_extension("story_id", default=None)
        doc_tuples = nlp.pipe(story_tuple, as_tuples=True)
        results = []
        for doc, context in doc_tuples:
            doc._.story_id = context["story_id"]
            prediction = [((ent.start_char, ent.end_char), ent.label_, ent.text)
                          for ent
                          in doc.ents]
            res = {doc._.story_id: prediction, "story_text": doc.text}
            results.append(res)
            logger.info(f"Total time taken to process story_id {doc._.story_id} "
                        f"is : {(datetime.now() - now).total_seconds()}")
        return results
    except Exception as err:
        logger.info(f"Ner Bert: Error occurred for story {story} "
                    f" Error: {err} , Traceback: {traceback.format_exc()}")


if __name__ == '__main__':
    uvicorn.run(
        f"{Path(__file__).stem}:app", port=8080, host='localhost', reload=True
    )