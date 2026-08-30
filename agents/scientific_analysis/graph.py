"""
Workflow de raisonnement de l'agent, structure avec LangGraph.

Schema :
    Input -> Agent (appel API Responses) -> Decision -> Tool(s) si besoin -> Agent (re-raisonne) -> Output

Le LLM (via l'API Responses) decide lui-meme, par function-calling, s'il doit
appeler un ou plusieurs tools, et boucle jusqu'a avoir assez d'information.
"""
import json
from typing import TypedDict, Any

from langgraph.graph import StateGraph, END

from agents.scientific_analysis.llm_client import get_llm_client, get_deployment_name
from agents.scientific_analysis.system_prompt import SYSTEM_PROMPT
from agents.scientific_analysis.tools import TOOL_SCHEMAS, TOOL_DISPATCH
from agents.scientific_analysis.schemas import AgentInput, AgentOutput

MAX_ITERATIONS = 4  # garde-fou : evite une boucle infinie tool<->LLM


class AgentState(TypedDict):
    input_items: list[dict]       # historique envoye a l'API Responses (system, user, function_call, function_call_output...)
    last_output: list[Any]        # dernier "output" retourne par l'API
    tool_calls_made: list[str]
    iterations: int
    final_text: str


def _agent_node(state: AgentState) -> AgentState:
    """Appelle l'API Responses avec l'historique courant et les tools disponibles."""
    client = get_llm_client()
    deployment = get_deployment_name()

    response = client.responses.create(
        model=deployment,
        input=state["input_items"],
        tools=TOOL_SCHEMAS,
    )

    state["last_output"] = response.output
    state["iterations"] += 1

    # On ne renvoie en entree que les function_call, avec uniquement les champs
    # attendus par l'API (renvoyer l'item complet via model_dump() inclut des
    # champs internes comme "status" que l'API refuse en entree).
    items_to_replay = []
    for item in response.output:
        if getattr(item, "type", None) == "function_call":
            items_to_replay.append({
                "type": "function_call",
                "call_id": item.call_id,
                "name": item.name,
                "arguments": item.arguments,
            })

    state["input_items"] = state["input_items"] + items_to_replay

    # Extraire le texte final si le modele a repondu directement (pas d'appel de tool)
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for content in item.content:
                if getattr(content, "type", None) == "output_text":
                    state["final_text"] = content.text

    return state


def _tools_node(state: AgentState) -> AgentState:
    """Execute chaque function_call demande par le LLM et ajoute le resultat a l'historique."""
    outputs = []
    for item in state["last_output"]:
        if getattr(item, "type", None) == "function_call":
            tool_name = item.name
            call_id = item.call_id
            try:
                args = json.loads(item.arguments) if item.arguments else {}
            except json.JSONDecodeError:
                args = {}

            fn = TOOL_DISPATCH.get(tool_name)
            if fn is None:
                result_text = f"ERREUR_OUTIL: unknown tool '{tool_name}'"
            else:
                try:
                    result_text = fn(**args)
                except TypeError as e:
                    # arguments invalides fournis par le LLM
                    result_text = f"ERREUR_OUTIL: invalid arguments for '{tool_name}' ({e})"

            state["tool_calls_made"].append(tool_name)
            outputs.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result_text,
            })

    state["input_items"] = state["input_items"] + outputs
    return state


def _should_continue(state: AgentState) -> str:
    """Route vers 'tools' si le dernier tour contenait des function_call, sinon termine."""
    if state["iterations"] >= MAX_ITERATIONS:
        return END

    has_function_call = any(
        getattr(item, "type", None) == "function_call" for item in state["last_output"]
    )
    return "tools" if has_function_call else END


def build_graph():
    """Construit et compile le graphe LangGraph de l'agent."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_scientific_analysis_agent(query: str) -> AgentOutput:
    """
    Point d'entree principal de l'agent.
    Valide la query, execute le graphe, retourne un AgentOutput structure
    quoi qu'il arrive (succes ou erreur).
    """
    try:
        validated_input = AgentInput(query=query)
    except Exception as e:
        return AgentOutput(answer="Invalid request.", status="error", error_message=str(e))

    try:
        compiled_graph = build_graph()
        initial_state: AgentState = {
            "input_items": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": validated_input.query},
            ],
            "last_output": [],
            "tool_calls_made": [],
            "iterations": 0,
            "final_text": "",
        }

        final_state = compiled_graph.invoke(initial_state)

        answer_text = final_state.get("final_text", "").strip()
        status = "insufficient_info" if not answer_text or "AUCUNE_INFO" in answer_text else "ok"

        return AgentOutput(
            answer=answer_text or "No answer could be generated.",
            tool_calls_made=final_state["tool_calls_made"],
            status=status,
        )

    except Exception as e:
        return AgentOutput(
            answer="A technical error occurred while processing the request.",
            status="error",
            error_message=f"{type(e).__name__}: {e}",
        )