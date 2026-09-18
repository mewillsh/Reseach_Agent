from pydantic import BaseModel, Field
from typing import List

# creating our object Analyst
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst")
    role: str = Field(description="Role of the analyst in the context of the topic")
    description: str = Field(description="Description of the analyst focus, concerns, and motives")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="List of analysts with their roles and affiliations matching the requested count")

