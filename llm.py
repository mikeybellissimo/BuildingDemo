from pydantic import BaseModel, Field
import streamlit as st
import uuid
from typing import Dict, List, TypedDict

from typing import Optional
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

class Extractor:
    class Building(BaseModel):
        """Information about a building."""

        # ^ Doc-string for the entity Person.
        # This doc-string is sent to the LLM as the description of the schema Person,
        # and it can help to improve extraction results.

        # Note that:
        # 1. Each field is an `optional` -- this allows the model to decline to extract it!
        # 2. Each field has a `description` -- this description is used by the LLM.
        # Having a good description can help improve extraction results.
        number_of_floors: Optional[int] = Field(default=None, description="The number of floors that the building has")
        number_of_rooms: Optional[int] = Field(default=None, description="The number of rooms that the building has")

    class Example(TypedDict):
        """A representation of an example consisting of text input and expected tool calls.

        For extraction, the tool calls are represented as instances of pydantic model.
        """
        
        input: str  # This is the example text
        tool_calls: List[BaseModel]  # Instances of pydantic model that should be extracted


    def tool_example_to_messages(example: Example) -> List[BaseMessage]:
        """Convert an example into a list of messages that can be fed into an LLM.

        This code is an adapter that converts our example to a list of messages
        that can be fed into a chat model.

        The list of messages per example corresponds to:

        1) HumanMessage: contains the content from which content should be extracted.
        2) AIMessage: contains the extracted information from the model
        3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

        The ToolMessage is required because some of the chat models are hyper-optimized for agents
        rather than for an extraction use case.
        """
        messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
        tool_calls = []
        for tool_call in example["tool_calls"]:
            tool_calls.append(
                {
                    "id": str(uuid.uuid4()),
                    "args": tool_call.dict(),
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                },
            )
        messages.append(AIMessage(content="", tool_calls=tool_calls))
        tool_outputs = example.get("tool_outputs") or [
            "You have correctly called this tool."
        ] * len(tool_calls)
        for output, tool_call in zip(tool_outputs, tool_calls):
            messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
        return messages
    
    def find_next_missing_data(state_data):
        for i in state_data:
            if state_data[i] == None:
                return i
        else:
            return None

    def extract(user_input, state):
        requested_data_prefix = ""
        if state.next_missing_data == None:
            examples = [
                (
                    "A blue 4 story hotel with 804 rooms",
                    Extractor.Building(number_of_floors=4, number_of_rooms=804),
                ),
                (
                    "I want to build an apartment with 63 floors and is rated 5 star",
                    Extractor.Building(number_of_floors=63),
                ),
                (
                    "I want to build a green house with 42 rooms",
                    Extractor.Building(number_of_rooms=42),
                ),
            ]
        else:
            examples = [
                (
                    "Requested Data: number_of_floors. \n"
                    "A blue 4 story hotel with 804 rooms",
                    Extractor.Building(number_of_floors=4, number_of_rooms=804),
                ),
                (
                    "Requested Data: number_of_floors. \n"
                    "63",
                    Extractor.Building(number_of_floors=63),
                ),
                (
                    "Requested Data: number_of_rooms. \n"
                    "42",
                    Extractor.Building(number_of_rooms=42),
                ),
            ]

            requested_data_prefix += "Requested Data: " + state.next_missing_data + ". \n"

        messages = []

        for text, tool_call in examples:
            messages.extend(
                Extractor.tool_example_to_messages({"input": text, "tool_calls": [tool_call]})
            )

        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert extraction algorithm that extracts information about a building that is described in the input text. "
                    "Only extract relevant information from the text. "
                    "If you do not know the value of an attribute asked "
                    "to extract, return null for the attribute's value.",
                ),
                # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
                MessagesPlaceholder("examples"),  # <-- EXAMPLES!
                # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                ("human", "{text}"),
            ]
        )
        
        openai_api_key = st.secrets["open_ai_api"]

        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key = openai_api_key)

        runnable = prompt | llm.with_structured_output(
            schema=Extractor.Building,
            method="function_calling",
            include_raw=False,
        )
        resp =  runnable.invoke({"text": requested_data_prefix + user_input, "examples": messages})
        

        if state.is_first:
            return resp.model_dump()
            
        else :
            # replace non-empty values in state
            new_values = resp.model_dump()
            for dict_key in new_values:
                if new_values[dict_key] != None:
                    state.data[dict_key] = new_values[dict_key]
            return state.data
    

