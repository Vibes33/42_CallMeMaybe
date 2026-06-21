from pydantic import BaseModel, Field
from typing import Dict, Any, List

class ParameterDefinition(BaseModel):
    type: str

class ReturnsDefinition(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnsDefinition

class PromptInput(BaseModel):
    prompt: str

class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]