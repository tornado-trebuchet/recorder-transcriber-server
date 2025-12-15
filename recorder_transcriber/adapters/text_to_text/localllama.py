from datetime import datetime, timezone
from typing import Any
from pydantic import BaseModel, Field
import langchain.agents as lc_agents
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from recorder_transcriber.config import Config, config

create_agent = lc_agents.create_agent

class EnhancedTranscript(BaseModel):
    markdown: str = Field(description="Clean Markdown transcript that keeps meaning, fixes disfluencies, and only adds structure when it helps readability.")
    tags: list[str] = Field(description="3-7 comma-safe topical tag strings ordered by relevance.", min_length=3, max_length=7)

class TextEnhancer:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or config
        llm_cfg = self.cfg.llm

        model: BaseChatModel = init_chat_model(
                llm_cfg["model"],
                temperature=llm_cfg.get("temperature", 0.2),
                timeout=llm_cfg.get("timeout", 30),
                base_url=llm_cfg.get("base_url"),
            )
        
        self._model = model

        self._agent: Any = create_agent(
            model=self._model,
            system_prompt="You clean up spoken transcripts into clear Markdown and extract 3-7 topical tags from the content. Provide both outputs in the required structured format.",
            response_format=ToolStrategy(
                schema=EnhancedTranscript,
                handle_errors="Please emit polished markdown plus 3-7 topical tags that satisfy the schema.",
            ),
        )

    def enhance(self, text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Text to enhance must be non-empty")

        normalized_text = text.strip()

        result = self._agent.invoke({
            "messages": [HumanMessage(content=normalized_text)]
        })

        structured_output = result["structured_response"]

        return {
            "markdown": structured_output.markdown.strip(),
            "tags": [tag.strip() for tag in structured_output.tags if tag.strip()],
            "date": datetime.now(timezone.utc).date().isoformat()
        }