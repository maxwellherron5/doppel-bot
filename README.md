# Doppel Bot

## Description
Doppel Bot is a Discord bot that utilizes OpenAI's language models to mimic user messages based on their past interactions. It leverages ChromaDB for storing and retrieving message history, allowing for personalized responses that reflect the user's tone and personality.

## Basic Setup
To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/doppel-bot.git
   cd doppel-bot
   ```

2. **Install `uv`**:
   `uv` is used as the package manager for this project. Install it by running: 
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Run commands with `just`**:
   The Justfile provides commands to format the code and run the bot. To install just, run:
   ```bash
   brew install just
   ```

## How It Works
Doppel Bot operates by listening to messages in Discord channels and storing them in a ChromaDB collection. When a user invokes the mimic command, the bot retrieves past messages from the specified user and generates a response that mimics their style.

### Key Features:
- **Message Storage**: Stores messages in ChromaDB for retrieval.
- **Mimicry**: Uses OpenAI's language models to generate responses that reflect the user's personality.
- **Command Handling**: Supports commands to mimic users and scrape message history.

### Main Components:
- **Discord Bot**: Built using the `discord.py` library, it handles events and commands.
- **ChromaDB**: A database for storing and querying user messages.
- **OpenAI Integration**: Utilizes OpenAI's models for generating responses based on user input.

### Running the Bot
To run the bot, execute the following command:
```bash
just run
```
Prior to running the bot, you will need to invoke the historical message scrape in order to build up and store the chat history.

Make sure to set your environment variables for `BOT_TOKEN` and `OPENAI_API_KEY` before running the bot.

## Environment Variables
You need to set the following environment variables for the bot to function correctly:
- `BOT_TOKEN`: Your Discord bot token.
- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_ORG`: (Optional) Your OpenAI organization ID.
