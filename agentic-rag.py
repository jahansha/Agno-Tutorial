from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.lancedb import LanceDb

from dotenv import load_dotenv

load_dotenv()

# Create a knowledge base from a PDF
vector_db = LanceDb(
    table_name="recipes",
    uri="/tmp/lancedb",  # You can change this path to store data elsewhere
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
)

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

# Comment out after first run as the knowledge base is loaded
knowledge_base.load(recreate=False) 

prompt = """
"You only answer with information from your RAG database.
You don't use your internal knowledge.
If you can't answer with the database, simple return 'I don't know'"
"""

agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    # Add the knowledge base to the agent
    knowledge=knowledge_base,
    show_tool_calls=True,
    markdown=True,
    instructions=[prompt],
)


agent.cli_app(stream=False)