#!/usr/bin/env python3
"""
Local CLI to test the SCD bot end-to-end without WhatsApp.

Usage:
    python scripts/chat.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.config import FAISS_INDEX_PATH
from app.rag.vector_store import vector_store
from app.handlers.message_router import route_message

TEST_PHONE = "test_user_cli"


def _startup() -> None:
    print("Loading vector store...", end=" ", flush=True)
    vector_store.load(FAISS_INDEX_PATH)
    print("done.\n")


async def _repl() -> None:
    print("=" * 50)
    print("  SCD Bot - local test console")
    print("  Type /help for menu, /stop to end session")
    print("  or ask any question about Sickle Cell Disease")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        try:
            reply = await route_message(TEST_PHONE, user_input)
            print(f"\nBot: {reply}\n")
        except Exception as e:
            print(f"\n[error] {e}\n")


def main() -> None:
    _startup()
    asyncio.run(_repl())


if __name__ == "__main__":
    main()
