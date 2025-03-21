import os
import asyncio

from doppel_bot.logger import get_logger

import discord
import chromadb
from discord.ext import commands
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


class DoppelBot:
    def __init__(self, llm_kwargs=None, embedding_kwargs=None):
        """
        Initialize and configure the Discord bot, LLM, vectorstore, etc.
        """
        load_dotenv()

        # Setup logger
        self.logger = get_logger(__name__)

        # Setup Discord bot
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        self.bot = commands.Bot(command_prefix="!", intents=intents)

        # Initialize ChromaDB collection
        self.collection = self.setup_collection()

        # Initialize Embeddings + Vectorstore
        self.vectorstore = self.setup_vectorstore(embedding_kwargs or {})
        self.retriever = self.vectorstore.as_retriever()

        # Initialize LLM
        self.llm = self.setup_llm(llm_kwargs or {})

        # Initialize RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.retriever
        )

        # Register events and commands
        self.register_events()
        self.register_commands()

    @staticmethod
    def setup_collection():
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        return chroma_client.get_or_create_collection(name="messages")

    @staticmethod
    def setup_llm(kwargs):
        """
        Create the ChatOpenAI LLM with either default settings or user-provided kwargs.
        """
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

    @staticmethod
    def setup_vectorstore(kwargs):
        """
        Create the embeddings and pass them into a Chroma vectorstore.
        """
        if kwargs:
            embeddings = OpenAIEmbeddings(**kwargs)
        else:
            embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                organization=os.getenv("OPENAI_ORG"),
                model="text-embedding-ada-002",
            )
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    def register_events(self):
        """
        Register on_ready and on_message events for the bot.
        """

        @self.bot.event
        async def on_ready():
            self.logger.info(f"Logged in as {self.bot.user}")

        @self.bot.event
        async def on_message(message):
            if message.author == self.bot.user:
                return

            # Only store non-empty messages
            content = message.content.strip()
            if content and (len(content) > 0 or content.startswith("!")):
                self.collection.add(
                    documents=[content],
                    metadatas=[{"author": str(message.author)}],
                    ids=[str(message.id)],
                )
            await self.bot.process_commands(message)

    def register_commands(self):
        """
        Register commands for the bot (e.g., mimic, scrape_history).
        """

        @self.bot.command()
        async def mimic(ctx, user: discord.Member, *, prompt: str):
            self.logger.info(f"Mimicking user with prompt: {prompt}")

            results = self.collection.query(
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
                personality_summary = self.llm.invoke(personality_prompt)
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
                self.logger.info(f"Sending request to OpenAI for user {user.name}...")
                chain_output = self.rag_chain.invoke(chain_input)
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

        @self.bot.command()
        async def scrape_history(ctx, limit: int = 1000):
            self.logger.info("Starting message history scrape...")

            async def scrape_channel(channel):
                count = 0
                try:
                    async for message in channel.history(
                        limit=limit, oldest_first=True
                    ):
                        if (
                            message.author == self.bot.user
                            or not message.content.strip()
                            or message.content.strip(" ")[0][0] == "!"
                        ):
                            continue

                        self.collection.add(
                            documents=[message.content],
                            metadatas=[{"author": str(message.author)}],
                            ids=[str(message.id)],
                        )
                        count += 1

                        if count % 100 == 0:
                            await asyncio.sleep(0.5)
                except discord.HTTPException as e:
                    self.logger.info(f"Error scraping {channel.name}: {e}")
                    if e.status == 429:
                        retry_after = e.retry_after or 5
                        self.logger.info(
                            f"Rate limit exceeded. Retrying after {retry_after} seconds..."
                        )
                        await asyncio.sleep(retry_after)
                        return await scrape_channel(channel)
                except Exception as e:
                    self.logger.info(f"Unexpected error scraping {channel.name}: {e}")
                return count

            total_count = 0
            for channel in ctx.guild.text_channels:
                self.logger.info(f"Scraping messages from #{channel.name}...")
                count = await scrape_channel(channel)
                total_count += count
                self.logger.info(f"Scraped {count} messages from {channel.name}")
                self.logger.info(
                    f"Completed scraping #{channel.name}: {count} messages collected"
                )
                await asyncio.sleep(2)

            self.logger.info(
                f"Done! Collected {total_count} messages across all channels."
            )

    def run(self):
        """
        Start the bot with the environment token.
        """
        self.bot.run(os.getenv("BOT_TOKEN"))
