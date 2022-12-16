from pydantic import BaseModel


class NerText(BaseModel):
    text: list


class SummaryText(BaseModel):
    text: str

