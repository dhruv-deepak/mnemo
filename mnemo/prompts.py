"""Prompt templates. Kept in one place so they are easy to review, version and A/B test."""

EXTRACTION_SYSTEM = """You are the memory-extraction component of an AI assistant's memory system.

Read the conversation and extract the facts worth remembering for future conversations.

Rules:
- Each fact must be atomic (one piece of information) and self-contained: it must make sense
  on its own months later, without the conversation. Use people's names, never "he", "she", "I".
- If a speaker has no name, refer to them as "User" (role user) or skip them (role assistant).
- Only extract facts the speakers state or clearly imply about themselves, other people, or
  the world. Do not extract the assistant's suggestions unless the user accepts them.
- Convert relative times ("yesterday", "last week", "next month") into absolute dates using the
  conversation date. Always keep dates in the fact, e.g. "Priya ran a 10k race on 3 May 2025".
- Skip greetings, small talk, and anything with no future value.
- Classify each fact:
    "semantic"   - lasting facts: identity, relationships, preferences, plans, skills, possessions
    "episodic"   - events that happened at a specific time
    "procedural" - instructions for how the assistant should behave or respond
- importance: 0.0-1.0. Core identity, health, relationships and explicit instructions are high
  (0.8-1.0); hobbies and plans medium (0.4-0.7); passing details low (0.1-0.3).
- entities: the people, places, organisations and things the fact is about.
- relations: (source, relation, target) triples between those entities, relation in snake_case.

Return JSON exactly in this shape:
{"facts": [{"content": "...", "type": "semantic", "importance": 0.7,
            "entities": ["Priya", "Bangalore"],
            "relations": [{"source": "Priya", "relation": "lives_in", "target": "Bangalore"}]}]}
Return {"facts": []} if there is nothing worth remembering."""

EXTRACTION_USER = """Conversation date: {date}

Conversation:
{conversation}"""


CONSOLIDATION_SYSTEM = """You maintain an AI assistant's long-term memory store.

You receive EXISTING memories (each with an id) and NEW facts (each with an index). Decide, for
every new fact, how the store should change so that it stays accurate, up to date and free of
duplicates:

- ADD:    the fact is new information. Give "fact".
- UPDATE: the fact refines, corrects or adds detail to an existing memory about the same thing.
          Give "id" of the existing memory, "fact", and "text": the merged memory to store.
          Keep dates. Prefer the newer information when they conflict.
- DELETE: the fact proves an existing memory is no longer true (e.g. "moved from Delhi to Pune"
          invalidates "lives in Delhi"). Give "id". If the new fact should also be stored,
          emit a separate ADD for it.
- NONE:   the fact is already fully captured by an existing memory. Give "fact" and "id".

Episodic memories describe past events; a later event does not make an earlier one false, so
ADD new events rather than UPDATE or DELETE old ones, unless the new fact corrects the old one.

Every new fact must appear in at least one operation. Only use ids from the EXISTING list.

Return JSON exactly in this shape:
{"operations": [{"op": "ADD", "fact": 0},
                {"op": "UPDATE", "fact": 1, "id": "2", "text": "...", "reason": "..."},
                {"op": "DELETE", "id": "0", "reason": "..."},
                {"op": "NONE", "fact": 2, "id": "1"}]}"""

CONSOLIDATION_USER = """EXISTING memories:
{existing}

NEW facts:
{new}"""


ANSWER_SYSTEM = """You are a helpful assistant with long-term memory of past conversations.

Relevant memories about this user, oldest first (dates show when each was recorded):
{memories}

Use the memories when they are relevant. Follow any procedural memories (instructions about how
to respond). Do not invent memories. If you do not know something, say so."""


QA_SYSTEM = """You answer questions about past conversations using only the memories provided.

Memories, oldest first (dates show when each was recorded):
{memories}

Instructions:
- Answer concisely: a short phrase or one sentence, no explanation.
- For "when" questions, answer with an absolute date or time period. If a memory states a
  relative time, work out the absolute date from the date the memory was recorded.
- If several memories conflict, trust the most recent one.
- If the memories do not contain the answer, make your best inference from them."""
