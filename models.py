from pydantic import BaseModel


class CompoundCombo(BaseModel):
    word1: str
    word2: str

    relation1: str
    relation2: str

    scores: dict[str, float]

    @property
    def cumulative_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores)




