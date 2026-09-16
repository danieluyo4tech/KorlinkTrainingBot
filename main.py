import os
import json
import datetime
import time
import urllib.request
import urllib.parse
from pathlib import Path
from zoneinfo import ZoneInfo

from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    ""
).strip()

TELEGRAM_CHAT_ID = os.getenv(
    "TELEGRAM_CHAT_ID",
    ""
).strip()

RUN_MODE = os.getenv(
    "RUN_MODE",
    "morning"
).strip().lower()

NIGERIA_TZ = ZoneInfo("Africa/Lagos")

QUESTIONS_FILE = Path("logs/questions.json")
POSTS_FILE = Path("logs/posts.json")

MAX_GENERATION_ATTEMPTS = 5


# ============================================================
# TRAINING TRACKS
# ============================================================
# These are broad training areas only.
# Gemini is free to choose the specific topic.

TRACKS = {
    0: "Cybersecurity",
    1: "Software Engineering",
    2: "Smart Home Automation",
    3: "Network Engineering",
    4: "Solar PV Design and Installation",
}


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_log_files():
    QUESTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not QUESTIONS_FILE.exists():
        QUESTIONS_FILE.write_text(
            "[]",
            encoding="utf-8"
        )

    if not POSTS_FILE.exists():
        POSTS_FILE.write_text(
            "[]",
            encoding="utf-8"
        )


def load_json(path):
    try:
        if not path.exists():
            return []

        content = path.read_text(
            encoding="utf-8"
        ).strip()

        if not content:
            return []

        data = json.loads(content)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_json_atomic(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temp_path.replace(path)


# ============================================================
# ENVIRONMENT
# ============================================================

def validate_environment():
    missing = []

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")

    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")

    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
        )


# ============================================================
# GEMINI
# ============================================================

def get_gemini_client():
    return genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# TEXT HELPERS
# ============================================================

def word_count(text):
    return len(
        str(text).split()
    )


def normalize_text(text):
    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )


def clean_json_response(text):
    text = str(text).strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================
# DAILY CHALLENGE VALIDATION
# ============================================================

def validate_poll(poll):
    if not isinstance(poll, dict):
        raise ValueError(
            "Generated response is not an object."
        )

    required_fields = [
        "question",
        "options",
        "correct_option",
        "explanation",
    ]

    for field in required_fields:
        if field not in poll:
            raise ValueError(
                f"Missing field: {field}"
            )

    question = str(
        poll["question"]
    ).strip()

    options = poll["options"]

    explanation = str(
        poll["explanation"]
    ).strip()

    practical_challenge = str(
        poll.get(
            "practical_challenge",
            ""
        )
    ).strip()

    bonus_challenge = str(
        poll.get(
            "bonus_challenge",
            ""
        )
    ).strip()

    if not question:
        raise ValueError(
            "Question is empty."
        )

    if word_count(question) < 6:
        raise ValueError(
            "Question is too short."
        )

    if word_count(question) > 25:
        raise ValueError(
            "Question is too long."
        )

    # Prevent exam-style questions.
    forbidden_phrases = [
        "which of the following",
        "what is the correct answer",
        "select the correct",
        "choose the correct",
        "according to the definition",
        "define ",
        "what does ",
        "which statement is true",
        "all of the above",
        "none of the above",
    ]

    normalized_question = normalize_text(
        question
    )

    for phrase in forbidden_phrases:
        if phrase in normalized_question:
            raise ValueError(
                "Question sounds like an exam question."
            )

    if not isinstance(options, list):
        raise ValueError(
            "Options must be a list."
        )

    if len(options) != 4:
        raise ValueError(
            "Exactly four options are required."
        )

    cleaned_options = []

    for option in options:
        option = str(option).strip()

        if not option:
            raise ValueError(
                "Option cannot be empty."
            )

        if word_count(option) > 6:
            raise ValueError(
                "Option is too long."
            )

        cleaned_options.append(option)

    normalized_options = [
        normalize_text(option)
        for option in cleaned_options
    ]

    if len(set(normalized_options)) != 4:
        raise ValueError(
            "Options must be unique."
        )

    try:
        correct_option = int(
            poll["correct_option"]
        )
    except Exception:
        raise ValueError(
            "correct_option must be a number from 1 to 4."
        )

    if correct_option not in [1, 2, 3, 4]:
        raise ValueError(
            "correct_option must be between 1 and 4."
        )

    if word_count(explanation) < 15:
        raise ValueError(
            "Explanation is too short."
        )

    if word_count(explanation) > 70:
        raise ValueError(
            "Explanation is too long."
        )

    if practical_challenge:
        if word_count(practical_challenge) > 30:
            raise ValueError(
                "Practical challenge is too long."
            )

    if bonus_challenge:
        if word_count(bonus_challenge) > 25:
            raise ValueError(
                "Bonus challenge is too long."
            )

        if bonus_challenge.endswith("?"):
            raise ValueError(
                "Bonus challenge must be an action, not a question."
            )

    poll["question"] = question
    poll["options"] = cleaned_options
    poll["correct_option"] = correct_option
    poll["explanation"] = explanation
    poll["practical_challenge"] = practical_challenge
    poll["bonus_challenge"] = bonus_challenge

    return poll


# ============================================================
# GENERATE DAILY CHALLENGE
# ============================================================

def generate_poll(track, recent_questions):
    client = get_gemini_client()

    recent_context = []

    for item in recent_questions[-60:]:
        if not isinstance(item, dict):
            continue

        recent_context.append({
            "question": item.get(
                "question",
                ""
            ),
            "options": item.get(
                "options",
                []
            ),
            "explanation": item.get(
                "explanation",
                ""
            ),
            "practical_challenge": item.get(
                "practical_challenge",
                ""
            ),
            "bonus_challenge": item.get(
                "bonus_challenge",
                ""
            ),
        })

    history_text = json.dumps(
        recent_context,
        ensure_ascii=False,
        indent=2
    )

    prompt = f"""
You are the daily training content instructor for
Korlink Technologies Ltd.

TODAY'S BROAD TRAINING TRACK:
{track}

IMPORTANT:

The track above is ONLY the broad subject area.

You have full freedom to choose the specific topic.

Do NOT use a fixed topic list.

Do NOT follow a predefined sequence.

Do NOT restrict yourself to examples from previous instructions.

You may choose any appropriate concept, situation, technology,
problem, behaviour, tool, process or real-world scenario that
belongs naturally to the day's track.

The goal is to expose learners to different parts of the subject
naturally over time.

============================================================
THE DAILY CHALLENGE
============================================================

Create ONE short, practical daily challenge.

This is NOT an exam.

It should feel like a situation a normal person could actually
encounter in everyday life.

The learner does not need to be a technical professional.

A person with little or no technical background should still be
able to understand the situation and make a reasonable choice.

The challenge should:

- be natural
- be interesting
- be practical
- encourage people to think
- be easy to read quickly
- relate to real life
- use simple language
- be 8–22 words ideally
- never exceed 25 words

The question should NOT test memorised definitions.

Do not use:

- Which of the following
- What is the correct answer
- Select the correct answer
- Choose the correct option
- According to the definition
- Define
- What does X mean
- Which statement is true
- All of the above
- None of the above

============================================================
OPTIONS
============================================================

Create exactly four options.

Options should be short.

Normally use 1–4 words.

Never exceed 6 words.

The learner should be able to read all four options quickly.

Do not put explanations inside the options.

============================================================
EXPLANATION
============================================================

The explanation is where the technical lesson should happen.

Explain the correct answer in simple, natural language.

It should:

- teach something useful
- explain why the answer makes sense
- introduce the relevant technical idea naturally
- be understandable to a beginner
- sound like an experienced instructor
- be 15–70 words

Do not make it sound like a textbook.

============================================================
PRACTICAL CHALLENGE
============================================================

Add ONE short practical follow-up.

It should allow learners to safely observe, check, compare,
practise or try something related to the lesson.

It must NOT be another multiple-choice question.

Keep it simple.

============================================================
BONUS CHALLENGE
============================================================

Add ONE small bonus challenge.

This bonus will NOT appear in the morning.

It will be revealed with the answer in the evening.

The bonus should give learners one additional simple thing to try.

It must:

- be practical
- be safe
- be connected to the day's lesson
- be different from the practical challenge
- ideally be 5–20 words
- never exceed 25 words
- be an action or observation
- NOT be written as a question
- NOT require special equipment
- NOT require paid software
- NOT require another person's account
- NOT involve offensive hacking
- NOT involve dangerous activity

For Solar PV, keep the bonus strictly observation-based.
Learners must not touch wiring, terminals, batteries,
exposed conductors or live electrical equipment.

============================================================
VARIETY
============================================================

Use the recent challenges below to avoid repetition.

Do not simply change a few words from an old challenge.

Avoid unnecessary repetition of:

- the same situation
- the same concept
- the same technology
- the same action
- the same learning point
- the same scenario
- the same question structure
- the same practical activity
- the same bonus activity

However, do NOT permanently exclude a subject simply because it
appeared recently.

A concept can appear again when the new challenge approaches it
from a genuinely different angle.

There is NO fixed topic sequence.

Choose naturally based on usefulness, freshness and learner interest.

============================================================
SAFETY
============================================================

Cybersecurity content must remain defensive, educational and safe.

Do not give instructions for hacking, credential theft,
bypassing security or attacking systems.

Solar activities must remain safe and observation-based.

For all tracks, the learner activity must be appropriate for
ordinary learners.

============================================================
RECENT CHALLENGES
============================================================

{history_text}

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "question": "Short practical daily challenge",
  "options": [
    "Short option",
    "Short option",
    "Short option",
    "Short option"
  ],
  "correct_option": 1,
  "explanation": "Short useful explanation.",
  "practical_challenge": "Short safe practical follow-up.",
  "bonus_challenge": "Short extra challenge learners can try."
}}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw_text = response.text or ""

    cleaned = clean_json_response(
        raw_text
    )

    poll = json.loads(cleaned)

    return validate_poll(poll)


# ============================================================
# QUESTION HISTORY
# ============================================================

def question_is_valid(poll):
    try:
        validate_poll(poll)
        return True
    except Exception:
        return False


def get_today_question(track):
    ensure_log_files()

    questions = load_json(
        QUESTIONS_FILE
    )

    today = datetime.datetime.now(
        NIGERIA_TZ
    ).date().isoformat()

    for item in questions:
        if not isinstance(item, dict):
            continue

        if item.get("date") != today:
            continue

        if item.get("track") != track:
            continue

        if question_is_valid(item):
            return item

    recent_same_track = [
        item
        for item in questions
        if isinstance(item, dict)
        and item.get("track") == track
    ]

    for attempt in range(
        MAX_GENERATION_ATTEMPTS
    ):
        try:
            poll = generate_poll(
                track,
                recent_same_track
            )

            new_question = normalize_text(
                poll["question"]
            )

            duplicate = False

            for old in recent_same_track:
                old_question = normalize_text(
                    old.get(
                        "question",
                        ""
                    )
                )

                if new_question == old_question:
                    duplicate = True
                    break

            if duplicate:
                print(
                    "Generated duplicate. Retrying..."
                )
                continue

            poll["date"] = today
            poll["track"] = track

            questions.append(poll)

            save_json_atomic(
                QUESTIONS_FILE,
                questions
            )

            return poll

        except Exception as exc:
            print(
                f"Generation attempt "
                f"{attempt + 1} failed: {exc}"
            )

            if attempt < (
                MAX_GENERATION_ATTEMPTS - 1
            ):
                time.sleep(2)

    raise RuntimeError(
        "Unable to generate a valid daily challenge."
    )


# ============================================================
# TELEGRAM
# ============================================================

def telegram_request(method, payload):
    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/{method}"
    )

    data = urllib.parse.urlencode(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        raw = response.read().decode(
            "utf-8"
        )

        result = json.loads(raw)

        if not result.get("ok"):
            raise RuntimeError(
                f"Telegram API error: {result}"
            )

        return result


def send_message(text):
    return telegram_request(
        "sendMessage",
        {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }
    )


# ============================================================
# MORNING MESSAGE
# ============================================================

def format_poll(poll, track):
    options = poll["options"]

    return (
        "*KORLINK TECHNOLOGIES*\n\n"
        "*DAILY CHALLENGE*\n\n"
        f"*Track:* {track}\n\n"
        f"{poll['question']}\n\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}\n\n"
        "What would you do in this situation?"
    )


# ============================================================
# EVENING ANSWER
# ============================================================

def format_answer(poll, track):
    options = poll["options"]

    correct_index = (
        poll["correct_option"] - 1
    )

    correct_answer = options[
        correct_index
    ]

    explanation = poll[
        "explanation"
    ].strip()

    practical = poll.get(
        "practical_challenge",
        ""
    ).strip()

    bonus = poll.get(
        "bonus_challenge",
        ""
    ).strip()

    parts = [
        "*KORLINK TECHNOLOGIES*",
        "",
        "*DAILY CHALLENGE — ANSWER*",
        "",
        f"*Track:* {track}",
        "",
        f"*Correct Answer:* {correct_answer}",
        "",
        f"*Why?* {explanation}",
    ]

    if practical:
        parts.extend([
            "",
            "*Practical Challenge:*",
            practical
        ])

    if bonus:
        parts.extend([
            "",
            "*Bonus Challenge:*",
            bonus
        ])

    return "\n".join(parts)


# ============================================================
# AI-GENERATED WEEKEND CONTENT
# ============================================================

def generate_weekend_message(
    day_type,
    recent_posts
):
    client = get_gemini_client()

    recent_context = []

    for item in recent_posts[-30:]:
        if not isinstance(item, dict):
            continue

        if item.get("mode") not in [
            "saturday",
            "sunday"
        ]:
            continue

        recent_context.append({
            "date": item.get(
                "date",
                ""
            ),
            "mode": item.get(
                "mode",
                ""
            ),
            "message": item.get(
                "message",
                ""
            ),
        })

    history_text = json.dumps(
        recent_context,
        ensure_ascii=False,
        indent=2
    )

    if day_type == "saturday":

        prompt = f"""
You are writing the Saturday message for
Korlink Technologies Ltd.

Generate a fresh motivational message for the training community.

The audience consists of learners and people developing their
technology skills.

The message can naturally touch on learning, discipline,
consistency, practical experience, personal development,
career growth, patience or progress.

You have freedom to choose the message and angle.

Do not follow a fixed motivational template.

Do not repeat previous messages.

The message must:

- sound like a real professional training organization
- be sincere
- be practical
- be concise
- be encouraging without exaggeration
- use natural business English

Do not sound like an AI.

Avoid exaggerated motivational clichés.

Avoid phrases such as:

"Never give up"
"You can achieve anything"
"The sky is the limit"
"Believe in yourself and conquer the world"

Do not make it childish.

Do not use excessive emojis.

Return ONLY valid JSON:

{{
  "title": "Short title",
  "message": "Short motivational message."
}}

Previous weekend messages:

{history_text}
"""

    else:

        prompt = f"""
You are writing the Sunday message for
Korlink Technologies Ltd.

Generate a fresh Gospel-based inspiration for the NEW WEEK.

The purpose is to encourage learners as they prepare for
the coming week.

You have freedom to choose the Biblical theme and message.

The message can naturally focus on themes such as wisdom,
strength, diligence, faith, patience, purpose, guidance,
peace, courage or responsibility, but do not follow a fixed
topic list or template.

Choose a suitable Bible passage yourself.

The message must:

- be genuinely Christian
- include ONE Bible verse reference
- connect naturally with the new week
- be concise
- be sincere
- be respectful
- be suitable for a professional training community
- sound natural rather than AI-generated

Do not write a sermon.

Do not reproduce a long Bible passage.

Briefly paraphrase the lesson of the verse in your own words.

Do not use excessive emojis.

Do not repeat recent Sunday messages.

Return ONLY valid JSON:

{{
  "title": "Short title",
  "verse_reference": "Book Chapter:Verse",
  "message": "Short Gospel-based inspiration for the new week."
}}

Previous weekend messages:

{history_text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw_text = response.text or ""

    cleaned = clean_json_response(
        raw_text
    )

    content = json.loads(cleaned)

    if not isinstance(content, dict):
        raise ValueError(
            "Weekend response is not an object."
        )

    title = str(
        content.get(
            "title",
            ""
        )
    ).strip()

    message = str(
        content.get(
            "message",
            ""
        )
    ).strip()

    if not title:
        raise ValueError(
            "Weekend title is empty."
        )

    if not message:
        raise ValueError(
            "Weekend message is empty."
        )

    if word_count(message) > 100:
        raise ValueError(
            "Weekend message is too long."
        )

    if day_type == "sunday":

        verse_reference = str(
            content.get(
                "verse_reference",
                ""
            )
        ).strip()

        if not verse_reference:
            raise ValueError(
                "Sunday message has no Bible reference."
            )

        return {
            "title": title,
            "verse_reference": verse_reference,
            "message": message
        }

    return {
        "title": title,
        "message": message
    }


# ============================================================
# WEEKEND FORMATTING
# ============================================================

def format_saturday_message(content):
    return (
        "*KORLINK TECHNOLOGIES*\n\n"
        "*SATURDAY NOTE*\n\n"
        f"*{content['title']}*\n\n"
        f"{content['message']}"
    )


def format_sunday_message(content):
    return (
        "*KORLINK TECHNOLOGIES*\n\n"
        "*SUNDAY INSPIRATION*\n\n"
        f"*{content['title']}*\n\n"
        f"{content['message']}\n\n"
        f"*Bible Reference:* "
        f"{content['verse_reference']}"
    )


# ============================================================
# POST LOGGING
# ============================================================

def log_post(
    mode,
    poll=None,
    track=None,
    message=None
):
    ensure_log_files()

    posts = load_json(
        POSTS_FILE
    )

    now = datetime.datetime.now(
        NIGERIA_TZ
    )

    entry = {
        "timestamp": now.isoformat(),
        "date": now.date().isoformat(),
        "mode": mode,
        "track": track
    }

    if poll:
        entry["question"] = poll.get(
            "question",
            ""
        )

    if message:
        entry["message"] = message

    posts.append(entry)

    save_json_atomic(
        POSTS_FILE,
        posts
    )


# ============================================================
# WEEKDAY
# ============================================================

def get_today_track():
    now = datetime.datetime.now(
        NIGERIA_TZ
    )

    weekday = now.weekday()

    return TRACKS.get(
        weekday
    )


def run_morning():
    track = get_today_track()

    if not track:
        print(
            "No weekday track is scheduled for today."
        )
        return

    poll = get_today_question(
        track
    )

    message = format_poll(
        poll,
        track
    )

    send_message(
        message
    )

    log_post(
        "morning",
        poll,
        track
    )

    print(
        f"Morning challenge sent: {track}"
    )


def run_evening():
    track = get_today_track()

    if not track:
        print(
            "No weekday track is scheduled for today."
        )
        return

    poll = get_today_question(
        track
    )

    message = format_answer(
        poll,
        track
    )

    send_message(
        message
    )

    log_post(
        "evening",
        poll,
        track
    )

    print(
        f"Evening answer sent: {track}"
    )


# ============================================================
# WEEKEND
# ============================================================

def run_weekend():
    now = datetime.datetime.now(
        NIGERIA_TZ
    )

    weekday = now.weekday()

    if weekday == 5:
        day_type = "saturday"

    elif weekday == 6:
        day_type = "sunday"

    else:
        print(
            "Weekend mode can only run on Saturday or Sunday."
        )
        return

    posts = load_json(
        POSTS_FILE
    )

    for attempt in range(
        MAX_GENERATION_ATTEMPTS
    ):
        try:

            content = generate_weekend_message(
                day_type,
                posts
            )

            if day_type == "saturday":
                message = format_saturday_message(
                    content
                )
            else:
                message = format_sunday_message(
                    content
                )

            send_message(
                message
            )

            log_post(
                day_type,
                message=message
            )

            print(
                f"{day_type.capitalize()} "
                f"message sent."
            )

            return

        except Exception as exc:

            print(
                f"{day_type.capitalize()} "
                f"generation attempt "
                f"{attempt + 1} failed: {exc}"
            )

            if attempt < (
                MAX_GENERATION_ATTEMPTS - 1
            ):
                time.sleep(2)

    raise RuntimeError(
        f"Unable to generate {day_type} message."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        validate_environment()

        ensure_log_files()

        print(
            f"Korlink Training Update started "
            f"in {RUN_MODE} mode."
        )

        if RUN_MODE == "morning":
            run_morning()

        elif RUN_MODE == "evening":
            run_evening()

        elif RUN_MODE == "weekend":
            run_weekend()

        else:
            raise ValueError(
                f"Unknown RUN_MODE: {RUN_MODE}"
            )

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        raise


if __name__ == "__main__":
    main()
