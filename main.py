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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

RUN_MODE = os.getenv("RUN_MODE", "morning").strip().lower()

NIGERIA_TZ = ZoneInfo("Africa/Lagos")

QUESTIONS_FILE = Path("logs/questions.json")
POSTS_FILE = Path("logs/posts.json")

MAX_GENERATION_ATTEMPTS = 5


# ============================================================
# WEEKDAY TRACKS
# ============================================================

TRACKS = {
    0: {
        "name": "Cybersecurity",
        "focus": "safe cybersecurity awareness and everyday digital protection",
    },
    1: {
        "name": "Software Engineering",
        "focus": "everyday software, apps, websites and how they behave",
    },
    2: {
        "name": "Smart Home Automation",
        "focus": "smart devices, automation and connected-home situations",
    },
    3: {
        "name": "Network Engineering",
        "focus": "Wi-Fi, internet connectivity, routers and everyday networks",
    },
    4: {
        "name": "Solar PV Design and Installation",
        "focus": "safe everyday observation and understanding of solar power systems",
    },
}


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_log_files():
    QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not QUESTIONS_FILE.exists():
        QUESTIONS_FILE.write_text("[]", encoding="utf-8")

    if not POSTS_FILE.exists():
        POSTS_FILE.write_text("[]", encoding="utf-8")


def load_json(path):
    try:
        if not path.exists():
            return []

        content = path.read_text(encoding="utf-8").strip()

        if not content:
            return []

        data = json.loads(content)

        return data if isinstance(data, list) else []

    except Exception:
        return []


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_suffix(".tmp")

    temp_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


# ============================================================
# ENVIRONMENT VALIDATION
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
    return len(str(text).split())


def normalize_text(text):
    return " ".join(
        str(text).lower().strip().split()
    )


def clean_json_response(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================
# POLL VALIDATION
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
            "",
        )
    ).strip()

    bonus_challenge = str(
        poll.get(
            "bonus_challenge",
            "",
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
                "Options must be short."
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
                "",
            ),
            "options": item.get(
                "options",
                [],
            ),
            "explanation": item.get(
                "explanation",
                "",
            ),
            "practical_challenge": item.get(
                "practical_challenge",
                "",
            ),
            "bonus_challenge": item.get(
                "bonus_challenge",
                "",
            ),
        })

    history_text = json.dumps(
        recent_context,
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are writing the daily training challenge for Korlink Technologies Ltd.

TRACK:
{track["name"]}

TRAINING FOCUS:
{track["focus"]}

This is NOT an examination.

The challenge is for a general training group. Some members may have
little or no technical background.

The question must therefore be based on a simple, realistic,
everyday situation that almost anyone can understand.

The technical learning should come mainly from the explanation.

============================================================
QUESTION
============================================================

Create ONE practical daily challenge.

The question must:

- sound natural
- feel like a real-life situation
- be interesting enough to make people want to answer
- be understandable by a complete beginner
- require simple thinking rather than memorisation
- be 8–22 words ideally
- never exceed 25 words
- NOT sound like an examination question

Do not use:

- Which of the following
- What is the correct answer
- Select the correct answer
- Choose the correct option
- According to the definition
- Define
- What does X mean
- Which statement is true

The question should feel like something that could happen at home,
at work, in school, while using a phone, or while using everyday technology.

============================================================
OPTIONS
============================================================

Create exactly FOUR short options.

Each option must:

- be easy to read quickly
- normally be 1–4 words
- never exceed 6 words
- be clearly different
- sound natural

Do not put long explanations inside the options.

============================================================
EXPLANATION
============================================================

After the learner answers, explain the correct answer clearly.

The explanation should:

- be 15–70 words
- teach the technical idea in simple language
- explain why the correct answer makes sense
- be useful to someone with no technical background
- sound like an experienced instructor explaining it to learners

Do not make the explanation sound like a textbook.

============================================================
PRACTICAL FOLLOW-UP
============================================================

Add ONE short practical follow-up.

It should be something the learner can safely:

- observe
- check
- compare
- practise
- or try

It must connect directly to the lesson.

It must NOT be another multiple-choice question.

Keep it short.

============================================================
BONUS CHALLENGE
============================================================

Add ONE small bonus challenge for learners to try AFTER the answer
is revealed in the evening.

The bonus challenge must:

- be simple
- be practical
- be safe
- reinforce today's concept
- be different from the practical follow-up
- ideally be 5–20 words
- never exceed 25 words
- be an action or observation
- NOT be written as a question
- NOT require special tools
- NOT require paid software
- NOT require accessing another person's account
- NOT involve hacking or offensive activity
- NOT involve dangerous electrical work

For Solar PV Design and Installation, the bonus must be
observation-only.

Learners must not touch wiring, terminals, batteries,
exposed conductors or live electrical equipment.

============================================================
TRACK GUIDANCE
============================================================

Cybersecurity:

Use safe awareness and protection situations.

Examples:

- suspicious messages
- passwords
- updates
- links
- account protection
- device security
- privacy
- public Wi-Fi awareness

Do not provide hacking instructions or offensive security activities.

Software Engineering:

Use familiar software situations.

Examples:

- an app freezing
- an app update
- login problems
- a website loading
- search behaviour
- saving information
- notifications
- restarting an app

Smart Home Automation:

Use familiar smart-home situations.

Do not repeatedly use:

- smart lights
- motion sensors
- arriving home
- lights turning on
- phone connection

Use different situations involving:

- smart plugs
- appliances
- schedules
- voice control
- sensors
- automation routines
- energy awareness
- device communication

Network Engineering:

Use familiar connectivity situations.

Examples:

- weak Wi-Fi
- router placement
- devices connecting
- internet interruptions
- sharing a connection
- network range
- connected devices

Solar PV Design and Installation:

Keep activities safe and observation-based.

Use everyday situations such as:

- sunlight
- energy use
- panels
- batteries
- charging
- daytime energy
- basic system behaviour

Do not ask learners to touch or inspect live electrical components.

============================================================
AVOID REPETITION
============================================================

Do not repeat recent questions.

Do not simply change a few words from an old question.

Avoid repeating the same:

- situation
- device
- action
- learning point
- scenario family
- opening phrase
- question structure
- practical activity
- bonus activity

Create a genuinely different challenge.

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
        contents=prompt,
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

        if item.get("date") == today:
            if item.get("track") == track["name"]:
                if question_is_valid(item):
                    return item

    recent_same_track = [
        item
        for item in questions
        if isinstance(item, dict)
        and item.get("track") == track["name"]
    ]

    for attempt in range(
        MAX_GENERATION_ATTEMPTS
    ):
        try:
            poll = generate_poll(
                track,
                recent_same_track,
            )

            normalized_new_question = normalize_text(
                poll["question"]
            )

            duplicate = False

            for old in recent_same_track:
                old_question = normalize_text(
                    old.get(
                        "question",
                        "",
                    )
                )

                if (
                    normalized_new_question
                    == old_question
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            poll["date"] = today
            poll["track"] = track["name"]

            questions.append(poll)

            save_json_atomic(
                QUESTIONS_FILE,
                questions,
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
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
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
        },
    )


# ============================================================
# MESSAGE FORMATTING
# ============================================================

def format_poll(poll, track):
    options = poll["options"]

    return (
        "*KORLINK TECHNOLOGIES*\n\n"
        "*DAILY CHALLENGE*\n\n"
        f"*Track:* {track['name']}\n\n"
        f"{poll['question']}\n\n"
        f"1. {options[0]}\n"
        f"2. {options[1]}\n"
        f"3. {options[2]}\n"
        f"4. {options[3]}\n\n"
        "Don't be afraid to get it wrong.\n"
        "The goal is to learn!\n"
    )


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
        "",
    ).strip()

    bonus = poll.get(
        "bonus_challenge",
        "",
    ).strip()

    parts = [
        "*KORLINK TECHNOLOGIES*",
        "",
        "*DAILY CHALLENGE — ANSWER*",
        "",
        f"*Track:* {track['name']}",
        "",
        f"*Correct Answer:* {correct_answer}",
        "",
        f"*Why?* {explanation}",
    ]

    if practical:
        parts.extend([
            "",
            "*Practical Challenge:*",
            practical,
        ])

    if bonus:
        parts.extend([
            "",
            "*Bonus Challenge:*",
            bonus,
        ])

    return "\n".join(parts)


# ============================================================
# AI-GENERATED WEEKEND CONTENT
# ============================================================

def generate_weekend_message(day_type, recent_posts):
    client = get_gemini_client()

    recent_context = []

    for item in recent_posts[-30:]:
        if not isinstance(item, dict):
            continue

        if item.get("mode") not in [
            "saturday",
            "sunday",
        ]:
            continue

        recent_context.append({
            "date": item.get(
                "date",
                "",
            ),
            "mode": item.get(
                "mode",
                "",
            ),
            "message": item.get(
                "message",
                "",
            ),
        })

    history_text = json.dumps(
        recent_context,
        ensure_ascii=False,
        indent=2,
    )

    if day_type == "saturday":
        prompt = f"""
You are writing the Saturday message for
Korlink Technologies Ltd training community.

Generate a fresh, natural and professional motivational message.

The audience consists of people learning technology and developing
their skills. The message should encourage:

- consistency
- learning
- discipline
- practical improvement
- patience
- personal development
- career growth

Do not sound like an AI.

Do not use exaggerated motivational language.

Do not use clichés such as:
"Never give up"
"You can achieve anything"
"Sky is the limit"
"Believe in yourself and conquer the world"

Do not make it childish.

Keep it concise enough for a Telegram training group.

The message should feel like something a real professional training
organization would send to its learners on a Saturday.

Do not use excessive emojis.

Return ONLY valid JSON:

{{
  "title": "Short title",
  "message": "Short motivational message."
}}

Recent weekend messages to avoid repeating:

{history_text}
"""

    else:
        prompt = f"""
You are writing the Sunday message for
Korlink Technologies Ltd training community.

Generate a fresh Gospel-based inspiration for the NEW WEEK.

The message should:

- be Christian and faith-based
- encourage wisdom, strength, diligence, peace and purposeful living
- connect naturally with starting a new week
- be respectful and suitable for a professional training community
- be concise
- sound natural and sincere
- avoid excessive religious language
- avoid sounding like an AI-generated sermon

Include ONE Bible verse reference.

Do not reproduce a long Bible passage.

You may briefly paraphrase the message of the verse in your own words.

The message should encourage learners as they prepare for
the coming week.

Do not use excessive emojis.

Return ONLY valid JSON:

{{
  "title": "Short title",
  "verse_reference": "Book Chapter:Verse",
  "message": "Short Gospel-based inspiration for the new week."
}}

Recent Sunday messages to avoid repeating:

{history_text}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
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
            "",
        )
    ).strip()

    message = str(
        content.get(
            "message",
            "",
        )
    ).strip()

    if not title or not message:
        raise ValueError(
            "Weekend content is incomplete."
        )

    if word_count(message) > 100:
        raise ValueError(
            "Weekend message is too long."
        )

    if day_type == "sunday":
        verse_reference = str(
            content.get(
                "verse_reference",
                "",
            )
        ).strip()

        if not verse_reference:
            raise ValueError(
                "Sunday message has no Bible reference."
            )

        return {
            "title": title,
            "verse_reference": verse_reference,
            "message": message,
        }

    return {
        "title": title,
        "message": message,
    }


# ============================================================
# WEEKEND MESSAGE FORMATTING
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
    message=None,
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
        "track": (
            track["name"]
            if track
            else None
        ),
    }

    if poll:
        entry["question"] = poll.get(
            "question",
            "",
        )

    if message:
        entry["message"] = message

    posts.append(entry)

    save_json_atomic(
        POSTS_FILE,
        posts,
    )


# ============================================================
# WEEKDAY RUNS
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
        track,
    )

    send_message(message)

    log_post(
        "morning",
        poll,
        track,
    )

    print(
        f"Morning challenge sent: "
        f"{track['name']}"
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
        track,
    )

    send_message(message)

    log_post(
        "evening",
        poll,
        track,
    )

    print(
        f"Evening answer sent: "
        f"{track['name']}"
    )


# ============================================================
# WEEKEND RUN
# ============================================================

def run_weekend():
    now = datetime.datetime.now(
        NIGERIA_TZ
    )

    weekday = now.weekday()

    posts = load_json(
        POSTS_FILE
    )

    if weekday == 5:
        day_type = "saturday"

    elif weekday == 6:
        day_type = "sunday"

    else:
        print(
            "Weekend mode can only run on Saturday or Sunday."
        )
        return

    for attempt in range(
        MAX_GENERATION_ATTEMPTS
    ):
        try:
            content = generate_weekend_message(
                day_type,
                posts,
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
                message=message,
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
        f"Unable to generate "
        f"{day_type} message."
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
