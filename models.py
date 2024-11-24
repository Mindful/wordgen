from pydantic import BaseModel


class CompoundCombo(BaseModel):
    base_word: str
    word1: str
    word2: str

    relation1: str
    relation2: str

    scores: dict[str, float]

    @property
    def cumulative_score(self) -> float:
        return sum(self.scores.values()) / len(self.scores)


    def __str__(self):
        # only include word1, word2 and cumulative score
        return f"Combo<{self.word1} {self.word2} ({self.cumulative_score:.2f})>"





