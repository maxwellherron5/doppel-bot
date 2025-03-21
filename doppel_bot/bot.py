import os
import discord
import chromadb
import asyncio
from discord.ext import commands
from dotenv import load_dotenv

from doppel_bot.logger import get_logger
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def setup_llm(kwargs={}):
    if kwargs:
        return ChatOpenAI(**kwargs)
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        organization=os.getenv("OPENAI_ORG"),
        model_name="gpt-4",
        temperature=0.2,
        max_tokens=512,
        top_p=1.0,
        frequency_penalty=0.3,
        presence_penalty=0.2,
    )


def setup_vectorstore(kwargs={}):
    if kwargs:
        embeddings = OpenAIEmbeddings(**kwargs)
    else:
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORG"),
            model="text-embedding-ada-002",
        )
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)


def setup_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="messages")


def register_events(bot, logger, collection):
    @bot.event
    async def on_ready():
        logger.info(f"Logged in as {bot.user}")

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        if message.content.strip() or message.content.strip(" ")[0][0] == "!":
            collection.add(
                documents=[message.content],
                metadatas=[{"author": str(message.author)}],
                ids=[str(message.id)],
            )
        await bot.process_commands(message)


def register_commands(bot, llm, rag_chain, collection, logger):
    @bot.command()
    async def mimic(ctx, user: discord.Member, *, prompt: str):
        logger.info(f"Mimicking user with prompt: {prompt}")
        results = collection.query(
            query_texts=[prompt], n_results=10, where={"author": str(user)}
        )

        if not results["documents"] or not results["documents"][0]:
            await ctx.send(
                f"I don't have enough messages from {user.display_name} to mimic them!"
            )
            return

        user_messages = results["documents"][0]
        extended_context = " ".join(user_messages[:50])
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
                "respond in a way that matches their unique style, tone, and mood. Make sure to incorporate specific phrases and mannerisms that are characteristic of them. "
                f"Never mention that you are an AI agent, completely assume the identity of {user.display_name}. \n"
                f"Personality summary: {personality_summary}\n\n"
                f"Example messages: {' '.join(user_messages[:50])}\n\n"
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

        if not response.strip():
            await ctx.send("Sorry, I couldn't generate a response.")
            return

        await ctx.send(f"**{user.name} says:** {response}")

    @bot.command()
    async def scrape_history(ctx, limit: int = 1000):
        logger.info("Starting message history scrape...")

        async def scrape_channel(channel):
            count = 0
            try:
                async for message in channel.history(limit=limit, oldest_first=True):
                    if (
                        message.author == bot.user
                        or not message.content.strip()
                        or message.content.strip(" ")[0][0] == "!"
                    ):
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
            count = await scrape_channel(channel)
            total_count += count
            logger.info(f"Scraped {count} messages from {channel.name}")
            logger.info(
                f"Completed scraping #{channel.name}: {count} messages collected"
            )
            await asyncio.sleep(2)

        logger.info(f"Done! Collected {total_count} messages across all channels.")


def run_bot(llm_kwargs=None, embedding_kwargs=None):
    load_dotenv()

    intents = discord.Intents.default()
    intents.messages = True
    intents.message_content = True
    bot = commands.Bot(command_prefix="!", intents=intents)

    logger = get_logger(__name__)

    llm = setup_llm(llm_kwargs)
    collection = setup_collection()
    vectorstore = setup_vectorstore(embedding_kwargs)
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )

    register_events(bot, logger, collection)
    register_commands(bot, llm, rag_chain, collection, logger)

    bot.run(os.getenv("BOT_TOKEN"))
