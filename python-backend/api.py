from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from uuid import uuid4
import time
import logging
import os

from main import (
    triage_agent,
    faq_agent,
    seat_booking_agent,
    flight_status_agent,
    cancellation_agent,
    create_initial_context,
    AirlineAgentContext,
    groq_model_provider
)

from agents import (
    Runner,
    ItemHelpers,
    MessageOutputItem,
    HandoffOutputItem,
    ToolCallItem,
    ToolCallOutputItem,
    InputGuardrailTripwireTriggered,
    Handoff,
    RunConfig
)
from database import db_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Airline Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    account_number: Optional[str] = None 

class LoginResponse(BaseModel):
    conversation_id: str
    customer_name: str
    account_number: str
    message: str

class MessageResponse(BaseModel):
    content: str
    agent: str

class AgentEvent(BaseModel):
    id: str
    type: str
    agent: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

class GuardrailCheck(BaseModel):
    id: str
    name: str
    input: str
    reasoning: str
    passed: bool
    timestamp: float

class ChatResponse(BaseModel):
    conversation_id: str
    current_agent: str
    messages: List[MessageResponse]
    events: List[AgentEvent]
    context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    guardrails: List[GuardrailCheck] = []

# =========================
# Conversation Store
# =========================

class SupabaseConversationStore:
    async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation state from Supabase."""
        conversation_data = await db_client.load_conversation(conversation_id)
        if conversation_data:
            if "context" in conversation_data and isinstance(conversation_data["context"], dict):
                conversation_data["context"] = AirlineAgentContext(**conversation_data["context"])
            else:
                logger.warning(f"Loaded conversation {conversation_id} has invalid context format. Reinitializing context.")
                conversation_data["context"] = await create_initial_context(account_number=None)
            
            conversation_data["input_items"] = conversation_data.get("history", []) 
            return conversation_data
        return None

    async def save(self, conversation_id: str, state: Dict[str, Any]):
        """Save conversation state to Supabase."""
        context_dict = state["context"].model_dump() if isinstance(state["context"], BaseModel) else state["context"]
        await db_client.save_conversation(
            session_id=conversation_id,
            history=state.get("input_items", []),
            context=context_dict,
            current_agent=state["current_agent"]
        )

conversation_store = SupabaseConversationStore()

# =========================
# Helpers
# =========================

def _get_agent_by_name(name: str):
    """Return the agent object by name."""
    agents = {
        triage_agent.name: triage_agent,
        faq_agent.name: faq_agent,
        seat_booking_agent.name: seat_booking_agent,
        flight_status_agent.name: flight_status_agent,
        cancellation_agent.name: cancellation_agent,
    }
    return agents.get(name, triage_agent)

def _get_guardrail_name(g) -> str:
    """Extract a friendly guardrail name."""
    name_attr = getattr(g, "name", None)
    if isinstance(name_attr, str) and name_attr:
        return name_attr
    guard_fn = getattr(g, "guardrail_function", None)
    if guard_fn is not None and hasattr(guard_fn, "__name__"):
        return guard_fn.__name__.replace("_", " ").title()
    fn_name = getattr(g, "__name__", None)
    if isinstance(fn_name, str) and fn_name:
        return fn_name.replace("_", " ").title()
    return str(g)

def _build_agents_list() -> List[Dict[str, Any]]:
    """Build a list of all available agents and their metadata."""
    def make_agent_dict(agent):
        return {
            "name": agent.name,
            "description": getattr(agent, "handoff_description", ""),
            "handoffs": [getattr(h, "agent_name", getattr(h, "name", "")) for h in getattr(agent, "handoffs", [])],
            "tools": [getattr(t, "name", getattr(t, "__name__", "")) for t in getattr(agent, "tools", [])],
            "input_guardrails": [_get_guardrail_name(g) for g in getattr(agent, "input_guardrails", [])],
        }
    return [
        make_agent_dict(triage_agent),
        make_agent_dict(faq_agent),
        make_agent_dict(seat_booking_agent),
        make_agent_dict(flight_status_agent),
        make_agent_dict(cancellation_agent),
    ]

# =========================
# Login Endpoint
# =========================

@app.get("/user/{account_number}", response_model=LoginResponse)
async def get_user_details_and_login(account_number: str):
    """
    Retrieves customer details by account number and initiates a new conversation session.
    """
    customer_data = await db_client.get_customer_by_account_number(account_number)

    if not customer_data:
        raise HTTPException(status_code=404, detail=f"Customer with account number '{account_number}' not found.")
    
    conversation_id = uuid4().hex
    ctx = await create_initial_context(account_number=account_number)
    current_agent_name = triage_agent.name

    state: Dict[str, Any] = {
        "input_items": [],
        "context": ctx,
        "current_agent": current_agent_name,
    }
    await conversation_store.save(conversation_id, state)

    return LoginResponse(
        conversation_id=conversation_id,
        customer_name=ctx.customer_name,
        account_number=ctx.account_number,
        message=f"Welcome, {ctx.customer_name}! You are logged in."
    )

# =========================
# Chat Endpoint
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main chat endpoint for agent orchestration.
    """
    conversation_id: str
    state: Dict[str, Any]

    if req.conversation_id:
        conversation_id = req.conversation_id
        state = await conversation_store.get(conversation_id)
        if not state:
            logger.warning(f"Conversation ID {conversation_id} not found. Starting new conversation.")
            ctx = await create_initial_context(account_number=req.account_number)
            current_agent_name = triage_agent.name
            state = {
                "input_items": [],
                "context": ctx,
                "current_agent": current_agent_name,
            }
    else:
        conversation_id = uuid4().hex
        ctx = await create_initial_context(account_number=req.account_number)
        current_agent_name = triage_agent.name
        state = {
            "input_items": [],
            "context": ctx,
            "current_agent": current_agent_name,
        }
    
    # If the message is empty and it's a new conversation, just return initial state
    if req.message.strip() == "" and not req.conversation_id:
        await conversation_store.save(conversation_id, state)
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=state["current_agent"],
            messages=[],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=[],
        )

    current_agent = _get_agent_by_name(state["current_agent"])
    if "input_items" not in state:
        state["input_items"] = []
    state["input_items"].append({"content": req.message, "role": "user"})
    old_context = state["context"].model_dump().copy()
    guardrail_checks: List[GuardrailCheck] = []

    # Create RunConfig with the GroqModelProvider instance
    run_config = RunConfig(model_provider=groq_model_provider)

    if groq_model_provider is None:
        logger.error("GroqModelProvider is not initialized. Cannot run agent.")
        raise HTTPException(status_code=500, detail="AI model service not available. GROQ_API_KEY might be missing.")

    try:
        result = await Runner.run(current_agent, state["input_items"], context=state["context"], run_config=run_config)
    except InputGuardrailTripwireTriggered as e:
        failed = e.guardrail_result.guardrail
        gr_output = e.guardrail_result.output.output_info
        gr_reasoning = getattr(gr_output, "reasoning", "")
        gr_input = req.message
        gr_timestamp = time.time() * 1000
        for g in current_agent.input_guardrails:
            guardrail_checks.append(GuardrailCheck(
                id=uuid4().hex,
                name=_get_guardrail_name(g),
                input=gr_input,
                reasoning=(gr_reasoning if g == failed else ""),
                passed=(g != failed),
                timestamp=gr_timestamp,
            ))
        refusal = "Sorry, I can only answer questions related to airline travel."
        state["input_items"].append({"role": "assistant", "content": refusal})
        await conversation_store.save(conversation_id, state)
        return ChatResponse(
            conversation_id=conversation_id,
            current_agent=current_agent.name,
            messages=[MessageResponse(content=refusal, agent=current_agent.name)],
            events=[],
            context=state["context"].model_dump(),
            agents=_build_agents_list(),
            guardrails=guardrail_checks,
        )

    messages: List[MessageResponse] = []
    events: List[AgentEvent] = []

    for item in result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            messages.append(MessageResponse(content=text, agent=item.agent.name))
            events.append(AgentEvent(id=uuid4().hex, type="message", agent=item.agent.name, content=text))
        elif isinstance(item, HandoffOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="handoff",
                    agent=item.source_agent.name,
                    content=f"{item.source_agent.name} -> {item.target_agent.name}",
                    metadata={"source_agent": item.source_agent.name, "target_agent": item.target_agent.name},
                )
            )
            from_agent = item.source_agent
            to_agent = item.target_agent
            ho = next(
                (h for h in getattr(from_agent, "handoffs", [])
                 if isinstance(h, Handoff) and getattr(h, "agent_name", None) == to_agent.name),
                None,
            )
            if ho:
                fn = ho.on_invoke_handoff
                fv = fn.__code__.co_freevars
                cl = fn.__closure__ or []
                if "on_handoff" in fv:
                    idx = fv.index("on_handoff")
                    if idx < len(cl) and cl[idx].cell_contents:
                        cb = cl[idx].cell_contents
                        cb_name = getattr(cb, "__name__", repr(cb))
                        events.append(
                            AgentEvent(
                                id=uuid4().hex,
                                type="tool_call",
                                agent=to_agent.name,
                                content=cb_name,
                            )
                        )
            current_agent = item.target_agent
        elif isinstance(item, ToolCallItem):
            tool_name = getattr(item.raw_item, "name", None)
            raw_args = getattr(item.raw_item, "arguments", None)
            tool_args: Any = raw_args
            if isinstance(raw_args, str):
                try:
                    import json
                    tool_args = json.loads(raw_args)
                except Exception:
                    pass
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_call",
                    agent=item.agent.name,
                    content=tool_name or "",
                    metadata={"tool_args": tool_args},
                )
            )
            if tool_name == "display_seat_map":
                messages.append(
                    MessageResponse(
                        content="DISPLAY_SEAT_MAP",
                        agent=item.agent.name,
                    )
                )
        elif isinstance(item, ToolCallOutputItem):
            events.append(
                AgentEvent(
                    id=uuid4().hex,
                    type="tool_output",
                    agent=item.agent.name,
                    content=str(item.output),
                    metadata={"tool_result": item.output},
                )
            )

    new_context = state["context"].model_dump()
    changes = {k: new_context[k] for k in new_context if old_context.get(k) != new_context[k]}
    if changes:
        events.append(
            AgentEvent(
                id=uuid4().hex,
                type="context_update",
                agent=current_agent.name,
                content="",
                metadata={"changes": changes},
            )
        )

    state["input_items"] = result.to_input_list()
    state["current_agent"] = current_agent.name
    await conversation_store.save(conversation_id, state)

    final_guardrails: List[GuardrailCheck] = []
    for g in getattr(current_agent, "input_guardrails", []):
        name = _get_guardrail_name(g)
        failed = next((gc for gc in guardrail_checks if gc.name == name), None)
        if failed:
            final_guardrails.append(failed)
        else:
            final_guardrails.append(GuardrailCheck(
                id=uuid4().hex,
                name=name,
                input=req.message,
                reasoning="",
                passed=True,
                timestamp=time.time() * 1000,
            ))

    return ChatResponse(
        conversation_id=conversation_id,
        current_agent=current_agent.name,
        messages=messages,
        events=events,
        context=state["context"].model_dump(),
        agents=_build_agents_list(),
        guardrails=final_guardrails,
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "airline-agent-api"}