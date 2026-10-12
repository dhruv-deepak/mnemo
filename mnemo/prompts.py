"""Prompt templates. Kept in one place so they are easy to review, version and A/B test."""

EXTRACTION_SYSTEM = """You are the memory-extraction component of an AI assistant's memory system.

Read the conversation and extract the facts worth remembering for future conversations.

Rules:
- Each fact must be atomic (one piece of information) and self-contained: it must make sense
  on its own months later, without the conversation. Use people's names, never "he", "she", "I".
- The speaker with role "user" is the person this memory store belongs to, and background
  memories describe that same person. If the speaker has no name but the background names
  them (e.g. "Dhruv is a student"), write facts about them with that name ("Dhruv is
  vegetarian"). Only if no name is known anywhere, write "User". Skip facts about the
  assistant itself.
- Only extract facts the speakers state or clearly imply about themselves, other people, or
  the world. Do not extract the assistant's suggestions unless the user accepts them.
- Anchor relative times to the conversation date, keeping their original precision:
  "yesterday" -> "on 11 October 2026"; "last week" -> "in the week before 12 October 2026";
  "last year" -> "in 2025". Always keep times in the fact, e.g. "Priya ran a 10k race on
  3 May 2025".
- Background memories are context only: use them to resolve names and pronouns, and do not
  re-extract facts they already state unless the conversation changes them.
- Skip greetings, small talk, and anything with no future value.
- Classify each fact:
    "semantic"   - lasting facts: identity, relationships, preferences, plans, skills, possessions
    "episodic"   - events that happened at a specific time
    "procedural" - instructions for how the assistant should behave or respond
- importance: 0.0-1.0. Core identity, health, relationships and explicit instructions are high
  (0.8-1.0); hobbies and plans medium (0.4-0.7); passing details low (0.1-0.3).
- entities: the people, places, organisations and things the fact is about.
- relations: (source, relation, target) triples between those entities, relation in snake_case.
  Targets must be entities (people, places, organisations, things, dates), never true/false
  or adjectives: write ("Dhruv", "diet", "vegetarian"), not ("Dhruv", "is_vegetarian", "true").

Return JSON exactly in this shape:
{"facts": [{"content": "...", "type": "semantic", "importance": 0.7,
            "entities": ["Priya", "Bangalore"],
            "relations": [{"source": "Priya", "relation": "lives_in", "target": "Bangalore"}]}]}
Return {"facts": []} if there is nothing worth remembering."""

EXTRACTION_USER = """Conversation date: {date}

Background memories (already stored):
{background}

Conversation:
{conversation}"""


CONSOLIDATION_SYSTEM = """You maintain an AI assistant's long-term memory store.

You receive EXISTING memories (each with an id) and NEW facts (each with an index). Decide, for
every new fact, how the store should change so that it stays accurate, up to date and free of
duplicates:

- ADD:    the fact is new information. Give "fact".
- UPDATE: the fact refines, corrects or adds detail to an existing memory about the same thing.
          Give "id" of the existing memory, "fact", and "text": the merged memory to store.
          The text must keep every detail of both that is still true, including dates.
          Prefer the newer information when they conflict.
- DELETE: the fact proves an existing memory is entirely no longer true (e.g. "moved from
          Delhi to Pune" invalidates "lives in Delhi"). Give "id". If the new fact should also
          be stored, emit a separate ADD for it. If only part of a memory is outdated, UPDATE
          it instead, keeping the parts that are still true ("CS student in Delhi" + moved to
          Pune -> "CS student, previously lived in Delhi").
- NONE:   the fact is already fully captured by an existing memory. Give "fact" and "id".

Check every EXISTING memory against the new facts. A new event can make a lasting fact
outdated: "moved to Pune last week" means "lives in Delhi" must be updated or deleted, even
though the move itself is ADDed. One new fact may need several operations.

Past events stay true: a later event does not make an earlier one false, so never UPDATE or
DELETE an episodic memory because of a newer event, only to correct it.

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
