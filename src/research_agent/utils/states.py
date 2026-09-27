from pydantic import BaseModel, Field
from typing_extensions import TypedDict, NotRequired
from typing import Optional, List
from research_agent.utils.objects import Analyst
from langgraph.graph import MessagesState
from typing_extensions import Annotated
import operator

# state
class GenerateAnalystsState(TypedDict):
    topic: str # Research Topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback for what is generated
    analysts: NotRequired[List[Analyst]] # List of all our Analysts

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs add new doc instead of replacement
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report