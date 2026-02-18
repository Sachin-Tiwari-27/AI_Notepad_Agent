from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Connect to your local Ollama model (server must be running!)
llm = OllamaLLM(model="phi3.5", temperature=0.7)

# Simple prompt template - tells the AI how to behave
prompt = PromptTemplate.from_template(
    "You are a helpful notepad assistant. "
    "Take this raw input and turn it into clean, organized notes. "
    "Add bullets, headings if useful, and keep it concise.\n\n"
    "User input: {input}\n\n"
    "Clean notes:"
)

# Chain: prompt → LLM → string output
chain = prompt | llm | StrOutputParser()

# Function to save notes to a file
def save_to_notepad(content):
    with open("my_notes.txt", "a", encoding="utf-8") as f:
        f.write("\n---\n" + content + "\n")
    print("Notes saved to my_notes.txt!")

# Main loop - talk to the agent
print("Notepad AI Agent ready! Type your raw notes (or 'quit' to exit).")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    # Run the chain
    result = chain.invoke({"input": user_input})
    
    print("\nAI Notes:")
    print(result)
    
    # Ask if save
    save_choice = input("\nSave these notes to file? (y/n): ").lower()
    if save_choice == 'y':
        save_to_notepad(result)