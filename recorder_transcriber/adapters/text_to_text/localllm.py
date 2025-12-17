from typing import Any

from datetime import datetime, timezone
from pydantic import BaseModel, Field

import langchain.agents as lc_agents
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from recorder_transcriber.config import config
from recorder_transcriber.model import Note

create_agent = lc_agents.create_agent # type: ignore

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

class TextEnhancer:
    def __init__(self) -> None:
        self.cfg = config
        llm_cfg = self.cfg.llm

        model: BaseChatModel = init_chat_model( # type: ignore
                llm_cfg["model"],
                temperature=llm_cfg.get("temperature", 0.4),
                timeout=llm_cfg.get("timeout", 160),
                base_url=llm_cfg.get("base_url"),
                api_key="not-needed"
            )
        
        self._model = model # type: ignore

        self._agent: Any = create_agent(
            model=self._model, # type: ignore
            system_prompt="You clean up spoken transcripts into clear Markdown and extract 3-5 topical tags and title from the content. Provide both outputs in the required structured format.",
            response_format=ToolStrategy(
                schema=EnhancedTranscript,
                handle_errors="Please emit polished markdown plus 3-5 topical tags that satisfy the schema.",
            ),
        )

    def enhance(self, text: str) -> Note:
        if not text or not text.strip():
            raise ValueError("Text to enhance must be non-empty")

        normalized_text = text.strip()

        result = self._agent.invoke({
            "messages": [HumanMessage(content=normalized_text)]
        })

        structured_output = result["structured_response"]

        markdown = structured_output.markdown.strip()
        title = structured_output.title.strip()
        tags = [tag.strip() for tag in structured_output.tags if tag.strip()]

        return Note(body=markdown, title=title, tags=tags, created_at=datetime.now(timezone.utc))