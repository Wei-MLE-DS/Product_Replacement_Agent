import os
import sys
import logging
import json
import pandas as pd
from typing_extensions import TypedDict
from typing import Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



# Import the agents
from image_agent import ImageAgent
from reason_agent import ReasonAgent
from recommendation_agent import RecommendationAgent


# --- State Definition ---
class ReturnAgentState(TypedDict):
    user_id: str
    image_path: str
    image_classification: Literal["real", "ai_generated", "photoshopped"]
    product_title: str
    return_reason: str
    reason_classification: str
    reason_summary: str
    recommendation: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

# --- Constants for Image Messages ---
IMAGE_CLASSIFICATION_MESSAGES = {
    "ai_generated": "Invalid picture: AI-generated. Please re-upload a valid picture.",
    "photoshopped": "Invalid picture: Photoshopped. Please re-upload a valid picture.",
    "real": "Image is valid. Please provide product title and reason for return."
}

# --- Agent 1: ImageAgent ---
def image_agent(state: ReturnAgentState) -> ReturnAgentState:
    """
    Agent responsible for image classification
    Returns: real, ai_generated, photoshopped
    """
    image_path = state.get("image_path", "")
    
    # Skip classification if already done (for testing)
    if state.get("image_classification"):
        return state
    
    # Use the ImageAgent from separate script
    image_analyzer = ImageAgent()
    classification = image_analyzer.classify_image(image_path)
    
    state["image_classification"] = classification
    
    return state

# ---  Agent 2: ReasonAgent  ---
def reason_agent(state: ReturnAgentState) -> ReturnAgentState:
    """
    Agent responsible for analyzing return reasons
    Delegates to ReasonAgent class in separate script
    """
    return_reason = state.get("return_reason", "")
    product_title = state.get("product_title", "")
    
    if not return_reason:
        return state
    
    reason_analyzer = ReasonAgent()
    analysis = reason_analyzer.analyze_reason(return_reason, product_title)
    
    state["reason_classification"] = analysis["classification"]
    state["reason_summary"] = analysis["summary"]
    
    return state

# --- Agent 3: RecommendationAgent ---
def recommendation_agent(state: ReturnAgentState) -> ReturnAgentState:
    """
    Agent responsible for finding alternative products
    Delegates to RecommendationAgent class in separate script
    """
    product_title = state.get("product_title", "")
    reason_classification = state.get("reason_classification", "")
    reason_summary = state.get("reason_summary", "")
    
    if not product_title or not reason_classification:
        state["recommendation"] = "Missing product information or reason classification."
        return state
    
    rec_agent = RecommendationAgent()
    recommendations = rec_agent.find_alternatives(
        product_title=product_title,
        reason_classification=reason_classification,
        reason_summary=reason_summary
    )
    
    state["recommendation"] = rec_agent.format_recommendations(recommendations)
    return state

# --- Router Functions ---
def router_image_to_reason(state: ReturnAgentState) -> str:
    """
    Router: Decide whether to proceed to reason analysis based on image classification
    Only proceed if image is "real"
    """
    if state["image_classification"] == "real":
        return "reason" # Go to reason_agent 
    else:
        return "end" # End workflow (invalid image)

def router_reason_to_recommendation(state: ReturnAgentState) -> str:
    """
    Router: Decide whether to proceed to recommendation based on reason analysis
    """
    if state.get("reason_classification"):
        return "recommendation"
    else:
        return "end"

# --- Router-Based Workflow Construction ---
def create_router_workflow() -> StateGraph:
    """
    Create the router-based workflow with 3 agents
    """
    workflow = StateGraph(ReturnAgentState)
    
    # Add the 3 core agents
    workflow.add_node("image", image_agent)
    workflow.add_node("reason", reason_agent)
    workflow.add_node("recommendation", recommendation_agent)
    
    # Set entry point
    workflow.set_entry_point("image")
    
    # Add router-based conditional edges
    workflow.add_conditional_edges(
        "image", 
        router_image_to_reason, 
        {"reason": "reason", "end": END}
    )
    workflow.add_conditional_edges(
        "reason", 
        router_reason_to_recommendation, 
        {"recommendation": "recommendation", "end": END}
    )
    workflow.add_edge("recommendation", END)
    
    return workflow.compile(checkpointer=MemorySaver())


# --- Main Functions ---
def run_agent_streamlit(image_path: str, product_title: str, return_reason: str, 
                       image_classification_override: str = None) -> str:
    """
    Main function for Streamlit integration
    Router-based workflow execution
    """
    graph = create_router_workflow()
    
    initial_state = {
        "user_id": "user1",
        "image_path": image_path,
        "image_classification": image_classification_override or "",
        "product_title": product_title,
        "return_reason": return_reason,
        "reason_classification": "",
        "reason_summary": "",
        "recommendation": "",
        "messages": [],
    }
    
    # Handle invalid image classifications early
    if image_classification_override and image_classification_override != "real":
        return IMAGE_CLASSIFICATION_MESSAGES.get(image_classification_override, 
                                                IMAGE_CLASSIFICATION_MESSAGES["real"])
    
    config = {"configurable": {"thread_id": "app-thread"}}
    result_state = graph.invoke(initial_state, config)
    
    return result_state.get("recommendation", "No recommendation found.")


# --- Testing Functions ---
def test_router_workflow():
    """
    Test the router-based workflow with sample data
    """
    test_cases = [
        {
            "image_path": "test.jpg",
            "product_title": "Dog Toy",
            "return_reason": "Too expensive for what it is",
            "image_classification_override": "real"
        },
        {
            "image_path": "test.jpg", 
            "product_title": "Cat Food",
            "return_reason": "Poor quality, my cat won't eat it",
            "image_classification_override": "real"
        },
        {
            "image_path": "test.jpg",
            "product_title": "Dog Toy", 
            "return_reason": "Too expensive",
            "image_classification_override": "ai_generated"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        result = run_agent_streamlit(**test_case)
        print(f"Result: {result}")

if __name__ == "__main__":
    test_router_workflow()