# -*- coding: utf-8 -*-
"""
Sutherland RAG Chatbot (Web Scraper + FAISS + LangChain + Mistral "Grok"-labeled UI)
-------------------------------------------------------------------------------
Refined single-file implementation that:
- Cleans up imports and comments (no Google GenAI bits left)
- Keeps existing functionality and model usage (LangChain + ChatMistralAI)
- Adds a polished Streamlit UI while preserving the original CLI chat helper
- Includes safe, idempotent package installation utilities (optional)

Run options:
1) Streamlit UI (recommended):
   streamlit run sutherland_rag_streamlit.py

2) CLI demo (fallback):
   python sutherland_rag_streamlit.py --cli

Notes:
- Web scraping is limited to Sutherland official pages listed below.
- The LLM is instantiated via `langchain_mistralai.ChatMistralAI` and labeled as
  "Grok" in copy per user request. Replace with your preferred model if needed.
"""

from __future__ import annotations
import os
import re
import time
import warnings
import argparse
from typing import List, Optional, Dict, Any

# ------------------------------ Optional: auto-install ------------------------------
# These helpers make the file robust in fresh environments (Colab, VM, etc.).
# If you prefer managing dependencies yourself, you can remove this block safely.

def _ensure_packages():
    import importlib
    import subprocess
    import sys

    required = [
        ("langchain", "langchain"),
        ("langchain-mistralai", "langchain_mistralai"),
        ("langchain-community", "langchain_community"),
        ("langchain-text-splitters", "langchain_text_splitters"),
        ("chromadb", "chromadb"),
        ("faiss-cpu", "faiss"),
        ("beautifulsoup4", "bs4"),
        ("requests", "requests"),
        ("lxml", "lxml"),
        ("sentence-transformers", "sentence_transformers"),
        ("python-dotenv", "dotenv"),
        ("streamlit", "streamlit"),
    ]

    for pkg, mod in required:
        try:
            importlib.import_module(mod)
        except Exception:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


# Comment out the next line if you manage deps externally
_ensure_packages()

# ------------------------------ Imports (post install) ------------------------------
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import streamlit as st
from getpass import getpass

warnings.filterwarnings("ignore")

# ============================== Scraper & KB Builder ===============================

class SutherlandWebScraper:
    def __init__(self):
        # Canonical list of Sutherland Global URLs to scrape
        self.sutherland_urls: List[str] = [
            "https://www.sutherlandglobal.com",
            "https://www.sutherlandglobal.com/about-us",
            "https://www.sutherlandglobal.com/services",
            "https://www.sutherlandglobal.com/industries",
            "https://www.sutherlandglobal.com/insights",
            "https://www.sutherlandglobal.com/careers",
            "https://www.sutherlandglobal.com/contact-us",
            "https://www.jobs.sutherlandglobal.com",
        ]

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )

        self.scraped_data: List[Dict[str, Any]] = []

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\!\?\,\;\:\-\(\)]", " ", text)
        text = re.sub(r"\.+", ".", text)
        return text.strip()

    def scrape_website(self, url: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }

        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.content, "lxml")

                for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                    tag.decompose()

                title = soup.find("title")
                title_text = title.get_text(strip=True) if title else url

                selectors = [
                    "main",
                    "article",
                    ".content",
                    ".main-content",
                    ".page-content",
                    ".post-content",
                    "section",
                    ".container",
                ]
                content_text = ""
                for sel in selectors:
                    for el in soup.select(sel):
                        content_text += el.get_text(separator=" ") + "\n"
                if not content_text.strip():
                    content_text = soup.get_text(separator=" ")

                clean = self.clean_text(content_text)
                if len(clean) > 200:
                    return {
                        "url": url,
                        "title": title_text,
                        "content": clean,
                        "word_count": len(clean.split()),
                    }
                else:
                    return None
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)

    def scrape_all_urls(self) -> bool:
        successful = 0
        total_words = 0
        self.scraped_data.clear()

        for url in self.sutherland_urls:
            data = self.scrape_website(url)
            if data:
                self.scraped_data.append(data)
                successful += 1
                total_words += data["word_count"]
            time.sleep(0.5)  # be nice
        return successful > 0

    def create_documents(self) -> List[Document]:
        if not self.scraped_data:
            return []
        docs: List[Document] = []
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for d in self.scraped_data:
            docs.append(
                Document(
                    page_content=d["content"],
                    metadata={
                        "source": d["url"],
                        "title": d["title"],
                        "word_count": d["word_count"],
                        "scrape_date": ts,
                    },
                )
            )
        return docs

    def create_vector_database(self):
        docs = self.create_documents()
        if not docs:
            return None
        chunks = []
        for doc in docs:
            chunks.extend(self.text_splitter.split_documents([doc]))
        try:
            vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            return vectorstore
        except Exception:
            return None


# ============================== RAG Bot (Mistral) ==================================

class SutherlandGeminiRAGBot:
    """Preserve the original class name for compatibility.
    Uses ChatMistralAI underneath; copy labels it as "Grok" per user brief.
    """

    def __init__(self, vectorstore, api_key: str):
        self.vectorstore = vectorstore
        self.api_key = api_key
        self.llm = None
        self.retriever = None
        self.qa_chain = None
        self.setup_llm()
        self.setup_rag_chain()

    def setup_llm(self):
        try:
            self.llm = ChatMistralAI(
                model="open-mistral-7b",
                temperature=0.3,
                mistral_api_key=self.api_key,
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.llm = None

    def setup_rag_chain(self):
        if not self.llm or not self.vectorstore:
            return

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

        prompt_template = (
            "You are SutherlandBot, an AI assistant powered by Grok (via Mistral), "
            "designed to answer questions about Sutherland Global Services using "
            "scraped information from their official websites.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer ONLY using the provided context.\n"
            "2. If the context is insufficient, say: \"Based on the information available "
            "from Sutherland's websites, I don't have enough details to fully answer this question\".\n"
            "3. Redirect unrelated questions back to Sutherland topics.\n"
            "4. Always mention that your information comes from Sutherland's official websites.\n"
            "5. Provide specific details and examples when available.\n\n"
            "CONTEXT FROM SUTHERLAND GLOBAL WEBSITES:\n{context}\n\n"
            "QUESTION: {question}\n\n"
            "ANSWER (based on Sutherland's official website content):"
        )

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    @staticmethod
    def _is_sutherland_related(question: str) -> bool:
        keywords = [
            "sutherland",
            "company",
            "services",
            "bpo",
            "business process",
            "outsourcing",
            "founded",
            "headquarters",
            "employees",
            "industries",
            "clients",
            "digital transformation",
            "ai",
            "automation",
            "cloud",
            "analytics",
            "careers",
            "jobs",
            "contact",
            "about",
            "global",
            "offices",
            "locations",
            "revenue",
            "acquisitions",
        ]
        q = question.lower()
        return any(k in q for k in keywords)

    def ask_question(self, question: str) -> str:
        if not self.qa_chain:
            return "❌ RAG chatbot not properly initialized."

        if not self._is_sutherland_related(question):
            return (
                "I'm SutherlandBot, powered by Grok (via Mistral), and I can only answer "
                "questions about Sutherland Global Services based on information scraped "
                "from their official websites.\n\nPlease ask me about:\n• Company background\n• Services & solutions\n"
                "• Industries served\n• Global presence\n• Careers\n• Contact info\n• Recent developments\n"
            )

        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "")
            sources = result.get("source_documents", [])

            dedup = []
            seen = set()
            for s in sources[:3]:
                src = s.metadata.get("source", "Unknown")
                if src not in seen:
                    seen.add(src)
                    dedup.append(src)

            stamp = sources[0].metadata.get("scrape_date", "Unknown") if sources else "Unknown"
            src_text = "\n".join(f"• {u}" for u in dedup)
            return (
                f"{answer}\n\n📚 **Sources from Sutherland's Official Websites:**\n"
                f"{src_text}\n\n🕒 *Information scraped on: {stamp}*"
            )
        except Exception as e:
            return f"❌ Error processing your question about Sutherland: {e}"

    def search_knowledge_base(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            return "❌ Vector database not available"
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            lines = [f"🔍 **Search Results for: '{query}'**", "=" * 50]
            for i, (doc, score) in enumerate(results, 1):
                lines.append(f"\n**{i}. Relevance Score: {score:.4f}**")
                lines.append(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                lines.append(f"**Content Preview:** {doc.page_content[:200]}...")
                lines.append("-" * 50)
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Search error: {e}"


# ============================== Streamlit UI =======================================

def _build_or_load_kb() -> Optional[FAISS]:
    """Scrape and build the vector store. Cached by Streamlit to avoid repeats."""
    @st.cache_resource(show_spinner=True)
    def _inner_build() -> Optional[FAISS]:
        scraper = SutherlandWebScraper()
        ok = scraper.scrape_all_urls()
        if not ok:
            return None
        return scraper.create_vector_database()

    return _inner_build()


def run_streamlit():
    st.set_page_config(page_title="SutherlandBot RAG", layout="wide")
    st.title("🏢 SutherlandBot RAG")
    st.caption("Powered by Grok (via Mistral) • Uses real-time content scraped from Sutherland Global websites")

    with st.sidebar:
        st.subheader("Setup")
        api_key = st.text_input("Enter your Mistral API key", type="password")
        if st.button("Initialize / Refresh Knowledge Base"):
            with st.spinner("Scraping Sutherland websites and building vector store..."):
                vs = _build_or_load_kb()
                if vs is None:
                    st.error("Failed to build the knowledge base. Please retry.")
                else:
                    st.session_state["vectorstore_ready"] = True
                    st.success("Knowledge base is ready!")

        kb_status = "✅ Ready" if st.session_state.get("vectorstore_ready") else "⏳ Not built"
        st.write(f"**KB Status:** {kb_status}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a question about Sutherland Global...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("SutherlandBot is thinking..."):
                if not api_key:
                    st.warning("Please provide your Mistral API key in the sidebar.")
                else:
                    vectorstore = _build_or_load_kb()
                    if vectorstore is None:
                        st.error("Knowledge base is not ready. Click the sidebar button to initialize.")
                    else:
                        bot = SutherlandGeminiRAGBot(vectorstore, api_key)
                        answer = bot.ask_question(prompt)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("🔎 Optional: Search knowledge base directly"):
        q = st.text_input("Search query")
        if st.button("Search"):
            vectorstore = _build_or_load_kb()
            if vectorstore is None:
                st.error("Knowledge base not ready.")
            else:
                # Use a transient bot for search convenience
                api = api_key or os.getenv("MISTRAL_API_KEY", "")
                bot = SutherlandGeminiRAGBot(vectorstore, api)
                st.markdown(bot.search_knowledge_base(q))


# ============================== CLI Fallback =======================================

def run_cli():
    print("\n🏢 SutherlandBot RAG - Powered by Grok (via Mistral)")
    print("Type 'search: <query>' to search the KB, or 'quit' to exit.\n")

    api_key = os.getenv("MISTRAL_API_KEY") or getpass("Enter your Mistral API key: ")

    print("Building knowledge base (this may take a moment)...")
    scraper = SutherlandWebScraper()
    if not scraper.scrape_all_urls():
        print("❌ Failed to scrape sufficient content")
        return
    vectorstore = scraper.create_vector_database()
    if vectorstore is None:
        print("❌ Failed to create vector database")
        return

    bot = SutherlandGeminiRAGBot(vectorstore, api_key)

    idx = 1
    while True:
        try:
            user_input = input(f"\n🤔 [{idx}] Ask about Sutherland Global: ").strip()
            if user_input.lower() in {"quit", "exit", "bye", "q"}:
                print("👋 Goodbye!")
                break
            if not user_input:
                continue
            if user_input.lower().startswith("search:"):
                q = user_input.split(":", 1)[1].strip()
                print(bot.search_knowledge_base(q))
                continue
            idx += 1
            print("\n🤖 SutherlandBot:\n" + "-" * 50)
            print(bot.ask_question(user_input))
            print("-" * 70)
        except KeyboardInterrupt:
            print("\n👋 Chat ended. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")


# ============================== Entrypoint =========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sutherland RAG Chatbot")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Streamlit")
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        # Running via `python` will still start Streamlit layout (useful in some IDEs),
        # but best is: `streamlit run sutherland_rag_streamlit.py`
        run_streamlit()
