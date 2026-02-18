from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

FILE_DIR = "./data/"

# 1. Define the tool with your fixed absolute path
@tool
def write_to_notepad(content: str, filename: str = FILE_DIR + "new_notes.txt"):
    """Writes or appends a note to a local text file."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"\n--- New Entry ({content[:50]}...) ---\n{content}\n")
    return f"Successfully saved to {filename}. You can continue."

tools = [write_to_notepad]

# 2. Setup the LLM
llm = ChatOllama(
    model="qwen2.5:3b",  # or try qwen2.5:3b for better tool following
    temperature=0.1,
    base_url="http://localhost:11434"
)

# 3. Bind tools (no tool_choice here – use default 'auto')
llm_with_tools = llm.bind_tools(tools)

# 4. System prompt
system_prompt = SystemMessage(content="""You are a helpful notepad / observation assistant that MUST follow these rules:

- If the user says anything like note, record, save, write down, remember, log, jot down, add to notes, or any phrase meaning 'save this information' → YOU MUST CALL the write_to_notepad tool. NO EXCEPTIONS. Do not write a normal reply instead.
- Never reply with text like "Noted" or "Saved" unless the tool has been called and returned success.
- Take this raw input and turn it into clean, organized notes. Add bullets, headings if useful, and keep it concise.
- After tool success, reply ONLY with "Noted and saved to new_notes.txt" or similar short confirmation.
- If no save action is requested, answer normally without tools.

Be concise and accurate.""")

# 5. Create the agent (no extra kwargs)
agent = create_agent(
    llm_with_tools,  # bound LLM runnable
    tools=tools,
)

# 6. Interactive loop
if __name__ == "__main__":
    print("Agent ready! Type 'quit' to exit.")
    print("Ensure 'ollama serve' is running in another terminal.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Invoke with messages including system prompt
        input_messages = [system_prompt, HumanMessage(content=user_input)]
        response = agent.invoke({"messages": input_messages})

        # Extract final output
        final_msg = response["messages"][-1]
        print("\nAgent:", final_msg.content)

        # Debug: show full response (look for tool_calls here)
        print("\nFull response:", response)