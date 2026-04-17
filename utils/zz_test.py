import questionary

choice = questionary.select(
    "Choose a model:",
    choices=["gpt-4", "claude-3", "llama-3"]
).ask()

print(f"You selected: {choice}")