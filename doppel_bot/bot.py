import discord
import os
import chromadb
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAIError
import asyncio

from doppel_bot.logger import get_logger

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def create_bot():
    # Load environment variables
    load_dotenv()
    TOKEN = os.getenv("BOT_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Setup Discord bot with intents
    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    # Setup ChromaDB client and collection
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="messages")

    # Setup OpenAI embeddings and vector store
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        organization=os.getenv("OPENAI_ORG"),
        model="text-embedding-ada-002",
    )
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # Setup ChatOpenAI LLM and RAG chain
    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        organization=os.getenv("OPENAI_ORG"),
        model_name="gpt-4",
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    logger = get_logger()

    @bot.event
    async def on_ready():
        logger.info(f"Logged in as {bot.user}")

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        # Only store non-empty messages
        if message.content.strip():
            collection.add(
                documents=[message.content],
                metadatas=[{"author": str(message.author)}],
                ids=[str(message.id)],
            )
        await bot.process_commands(message)

    @bot.command()
    async def mimic(ctx, user: discord.Member, *, prompt: str):
        """Mimic a user based on their past messages."""
        results = collection.query(
            query_texts=[prompt], n_results=10, where={"author": str(user)}
        )

        if not results["documents"] or not results["documents"][0]:
            await ctx.send(
                f"I don't have enough messages from {user.mention} to mimic them!"
            )
            return

        user_messages = results["documents"][0]

        extended_context = " ".join(user_messages[:20])
        personality_prompt = (
            f"Analyze the following messages from {user.display_name} and extract detailed adjectives, tone, mood, "
            "and recurring language patterns that best characterize their style. Provide a concise, bullet-point summary of these traits.\n"
            f"Messages: {extended_context}"
        )

        try:
            personality_summary = llm.invoke(personality_prompt)
        except Exception as e:
            personality_summary = ""

        chain_input = {
            "query": (
                f"You are impersonating {user.display_name}. Based on the following detailed personality summary and example messages, "
                "respond in a way that matches their unique style, tone, and mood. Make sure to incorporate specific phrases and mannerisms that are characteristic of them.\n\n"
                f"Personality summary: {personality_summary}\n\n"
                f"Example messages: {' '.join(user_messages[:10])}\n\n"
                f"User prompt: {prompt}"
            )
        }

        try:
            logger.info(f"Sending request to OpenAI for user {user.name}...")
            chain_output = rag_chain.invoke(chain_input)
        except Exception as e:
            await ctx.send(f"Error: {type(e).__name__} - {e}")
            return

        response = (
            chain_output.get("result")
            if isinstance(chain_output, dict)
            else str(chain_output)
        )

        if not response:
            await ctx.send("Sorry, I couldn't generate a response.")
            return

        await ctx.send(f"**{user.name} says:** {response}")

    @bot.command()
    async def scrape_history(ctx, limit: int = 1000):
        """Scrape message history from all text channels and store them in ChromaDB."""
        # await ctx.send("Starting message history scrape...")
        logger.info("Starting message history scrape...")

        async def scrape_channel(channel):
            count = 0
            try:
                async for message in channel.history(limit=limit, oldest_first=True):
                    if message.author == bot.user or not message.content.strip():
                        continue

                    collection.add(
                        documents=[message.content],
                        metadatas=[{"author": str(message.author)}],
                        ids=[str(message.id)],
                    )
                    count += 1

                    if count % 100 == 0:
                        await asyncio.sleep(0.5)
            except discord.HTTPException as e:
                logger.info(f"Error scraping {channel.name}: {e}")
                if e.status == 429:
                    retry_after = e.retry_after or 5
                    logger.info(
                        f"Rate limit exceeded. Retrying after {retry_after} seconds..."
                    )
                    await asyncio.sleep(retry_after)
                    return await scrape_channel(channel)
            except Exception as e:
                logger.info(f"Unexpected error scraping {channel.name}: {e}")
            return count

        total_count = 0
        for channel in ctx.guild.text_channels:
            logger.info(f"Scraping messages from #{channel.name}...")
            # await ctx.send(f"Scraping messages from #{channel.name}...")
            count = await scrape_channel(channel)
            total_count += count
            logger.info(f"Scraped {count} messages from {channel.name}")
            logger.info(
                f"Completed scraping #{channel.name}: {count} messages collected"
            )
            # await ctx.send(
            #     f"Completed scraping #{channel.name}: {count} messages collected"
            # )
            await asyncio.sleep(2)  # Wait between channels

        logger.info(f"Done! Collected {total_count} messages across all channels.")
        # await ctx.send(f"Done! Collected {total_count} messages across all channels.")

    return bot, TOKEN


def run_bot():
    bot, TOKEN = create_bot()
    bot.run(TOKEN)
