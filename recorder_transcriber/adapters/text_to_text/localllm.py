from typing import Any

from datetime import datetime, timezone
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
import langchain.agents as lc_agents

from recorder_transcriber.domain.models import Note, Transcript

create_agent = lc_agents.create_agent

class EnhancedTranscript(BaseModel):
    markdown: str = Field(
    description="""Clean Markdown transcript that:
1. **Faithfully preserves meaning** and intent from the original spoken content.
2. **Polishes spoken disfluencies** (ums, ahs, repetitions, false starts) into fluent text.
3. **Adds minimal Markdown structure** only when it significantly aids readability (e.g., headings for clear topic shifts, lists for sequences, emphasis for key points).
4. **Corrects likely transcription errors**: If a word or phrase sounds acoustically similar but doesn't fit the context, reason about and substitute a more semantically appropriate alternative. Flag major corrections with a `[?]` if uncertain.
"""
)
    title: str = Field(description="Crisp, human-friendly summary phrase for the transcript.")
    tags: list[str] = Field(description="3-4 comma-safe topical tag strings ordered by relevance.", min_length=3, max_length=5)

class LangchainAdapter:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: int = 160,
    ) -> None:
        model_obj: BaseChatModel = init_chat_model(
            str(model),
            temperature=float(temperature),
            timeout=int(timeout),
            base_url=str(base_url),
            api_key="not-needed",
        )
        
        self._model = model_obj

        self._agent: Any = create_agent(
            model=self._model,
            system_prompt="You clean up spoken transcripts into clear Markdown and extract 3-5 topical tags and title from the content. Provide both outputs in the required structured format.",
            response_format=ToolStrategy(
                schema=EnhancedTranscript,
                handle_errors="Please emit polished markdown plus 3-5 topical tags that satisfy the schema.",
            ),
        )

    def enhance(self, transcript: Transcript) -> Note:
        if not transcript.text or not transcript.text.strip():
            raise ValueError("Text to enhance must be non-empty")

        normalized_text = transcript.text.strip()

        result = self._agent.invoke({
            "messages": [HumanMessage(content=normalized_text)]
        })

        structured_output = result["structured_response"]

        markdown = structured_output.markdown.strip()
        title = structured_output.title.strip()
        tags = [tag.strip() for tag in structured_output.tags if tag.strip()]

        return Note(body=markdown, title=title, tags=tags, created_at=datetime.now(timezone.utc))
