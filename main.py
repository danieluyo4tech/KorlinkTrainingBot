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
# KORLINK TRAINING UPDATE
# KORLINK TECHNOLOGIES LTD
# ============================================================

APP_NAME = "Korlink Daily Challenge"

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RUN_MODE = os.getenv("RUN_MODE", "morning").strip().lower()

NIGERIA_TZ = ZoneInfo("Africa/Lagos")

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"

QUESTIONS_FILE = LOG_DIR / "questions.json"
POSTS_FILE = LOG_DIR / "posts.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# WEEKLY TRAINING TRACKS
# ============================================================

WEEKDAY_TRACKS = {
    0: {
        "school": "School of Computing",
        "name": "Cybersecurity",
        "description": (
            "cybersecurity, phishing, social engineering, "
            "password security, privacy, threats, malware, "
            "digital safety and security awareness"
        ),
    },

    1: {
        "school": "School of Computing",
        "name": "Software Engineering",
        "description": (
            "software development, programming, debugging, "
            "testing, databases, APIs, version control and "
            "software development practices"
        ),
    },

    2: {
        "school": "School of Technology",
        "name": "Smart Home Automation",
        "description": (
            "smart homes, IoT devices, sensors, lighting, "
            "security systems, controllers, automation and "
            "practical home automation"
        ),
    },

    3: {
        "school": "School of Technology",
        "name": "Network Engineering",
        "description": (
            "computer networks, routers, switches, Wi-Fi, "
            "IP addressing, connectivity, troubleshooting "
            "and network security"
        ),
    },

    4: {
        "school": "School of Technology",
        "name": "Solar PV Design and Installation",
        "description": (
            "solar panels, batteries, charge controllers, "
            "inverters, system sizing, installation, "
            "maintenance and troubleshooting"
        ),
    },
}


# ============================================================
# TIME
# ============================================================

def nigeria_now():
    return datetime.datetime.now(NIGERIA_TZ)


def today_string():
    return nigeria_now().date().isoformat()


# ============================================================
# ENVIRONMENT
# ============================================================

def validate_environment():
    missing = []

    if not API_KEY:
        missing.append("GEMINI_API_KEY")

    if not GEMINI_MODEL:
        missing.append("GEMINI_MODEL")

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
# JSON STORAGE
# ============================================================

def load_json(file_path, default):
    if not file_path.exists():
        return default

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    except (json.JSONDecodeError, OSError) as error:
        print(
            f"Warning: Could not read {file_path}: {error}"
        )
        return default


def save_json(file_path, data):
    """
    Atomic save to reduce the possibility of
    corrupted JSON files.
    """

    temp_file = file_path.with_suffix(".tmp")

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temp_file.replace(file_path)


def load_questions():
    return load_json(QUESTIONS_FILE, [])


def save_questions(questions):
    save_json(
        QUESTIONS_FILE,
        questions[-300:]
    )


def load_posts():
    return load_json(POSTS_FILE, [])


def save_posts(posts):
    save_json(
        POSTS_FILE,
        posts[-300:]
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)


def clean_json_response(text):
    """
    Remove markdown code fences or accidental text around JSON.
    """

    if not text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def normalize_correct_option(poll):
    """
    Gemini may occasionally use another name for the answer field.
    Normalize common alternatives.
    """

    if "correct_option" not in poll:

        aliases = [
            "correct_answer",
            "answer",
            "correct",
            "correct_choice",
            "correct_index",
        ]

        for alias in aliases:
            if alias in poll:
                poll["correct_option"] = poll[alias]
                break

    if "correct_option" not in poll:
        return poll

    value = poll["correct_option"]

    if isinstance(value, str):

        value = value.strip()

        if value.isdigit():
            value = int(value)

        else:
            lowered = value.lower()

            if lowered.startswith("option "):

                number = lowered.replace(
                    "option ",
                    ""
                ).strip()

                if number.isdigit():
                    value = int(number)

            elif "options" in poll:

                for index, option in enumerate(
                    poll["options"],
                    start=1
                ):

                    if (
                        lowered
                        == str(option).strip().lower()
                    ):
                        value = index
                        break

    poll["correct_option"] = value

    return poll


# ============================================================
# POLL VALIDATION
# ============================================================

def validate_poll(poll):

    if not isinstance(poll, dict):
        raise ValueError(
            "Gemini response is not a JSON object."
        )

    required = [
        "question",
        "options",
        "correct_option",
        "explanation",
    ]

    for field in required:

        if field not in poll:
            raise ValueError(
                f"Gemini response is missing: {field}"
            )

    if (
        not isinstance(poll["question"], str)
        or not poll["question"].strip()
    ):
        raise ValueError(
            "Question is empty."
        )

    options = poll["options"]

    if not isinstance(options, list):
        raise ValueError(
            "Options must be a list."
        )

    if len(options) != 4:
        raise ValueError(
            "Exactly four options are required."
        )

    for option in options:

        if (
            not isinstance(option, str)
            or not option.strip()
        ):
            raise ValueError(
                "Every option must contain text."
            )

    try:
        correct_option = int(
            poll["correct_option"]
        )

    except (TypeError, ValueError):

        raise ValueError(
            "correct_option must be a number."
        )

    if correct_option not in (1, 2, 3, 4):

        raise ValueError(
            "correct_option must be between 1 and 4."
        )

    if (
        not isinstance(poll["explanation"], str)
        or not poll["explanation"].strip()
    ):

        raise ValueError(
            "Explanation is empty."
        )

    practical = poll.get(
        "practical_challenge",
        ""
    )

    if not isinstance(practical, str):
        practical = str(practical)

    poll["correct_option"] = correct_option
    poll["practical_challenge"] = practical.strip()

    return poll


# ============================================================
# GEMINI GENERATION
# ============================================================

def gemini_generate(prompt, attempts=5):

    if client is None:
        raise RuntimeError(
            "Gemini client is not configured."
        )

    last_error = None

    for attempt in range(1, attempts + 1):

        try:

            print(
                f"Gemini attempt "
                f"{attempt}/{attempts}..."
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if not response:
                raise RuntimeError(
                    "Gemini returned no response."
                )

            text = getattr(
                response,
                "text",
                None
            )

            if not text:
                raise RuntimeError(
                    "Gemini response contained no text."
                )

            return text

        except Exception as error:

            last_error = error

            print(
                f"Gemini attempt {attempt} failed: "
                f"{error}"
            )

            if attempt < attempts:
                time.sleep(2 * attempt)

    raise RuntimeError(
        "Gemini generation failed after "
        f"{attempts} attempts: {last_error}"
    )


# ============================================================
# DAILY CHALLENGE GENERATOR
# ============================================================

def generate_poll(track):

    prompt = f"""
You are the official content writer for Korlink Technologies Ltd.

You are creating the daily learning challenge for the
Korlink Training Update WhatsApp group.

PROGRAM:
Korlink Daily Challenge

SCHOOL:
{track["school"]}

TRAINING TRACK:
{track["name"]}

TOPIC AREA:
{track["description"]}

Create ONE engaging multiple-choice challenge.

AUDIENCE:

The group contains learners from different backgrounds and
different levels of technical knowledge.

Some are beginners.
Some are currently learning.
Some already have technical experience.

The challenge must therefore be easy to understand while still
testing a genuine technical idea.

------------------------------------------------------------
HOOK AND ENGAGEMENT
------------------------------------------------------------

The challenge MUST start with a natural, interesting hook.

The hook should make the reader curious enough to continue.

Use a realistic situation, observation or problem.

Examples of the STYLE we want:

"You could be hacked without clicking a single link."

"Here's something many people overlook about Wi-Fi."

"One small change can make a working application fail."

"Imagine your house could respond automatically when you arrive."

"Your solar system is receiving sunlight, but something is wrong."

These are examples only.

Create a fresh hook that fits the day's subject.

DO NOT copy these examples.

The hook must not be exaggerated clickbait.

------------------------------------------------------------
IMPORTANT WRITING RULES
------------------------------------------------------------

The content must feel like it was written by a real instructor
for a professional training community.

Do NOT make it sound like AI-generated social media content.

Avoid:

- childish wording
- excessive motivation
- exaggerated hype
- unnecessary emojis
- long introductions
- textbook definitions
- complicated jargon
- fake excitement
- "Let's see who gets this!"
- "Only geniuses can answer!"
- "Are you ready?"
- "Test your IQ!"

Keep it professional, practical and interesting.

------------------------------------------------------------
LENGTH
------------------------------------------------------------

The complete challenge must be concise.

Aim for approximately 20-40 words for the situation and question.

One or two short paragraphs maximum.

It should take only a few seconds to read.

DO NOT write a long story.

DO NOT write a mini-article.

DO NOT explain the answer inside the question.

------------------------------------------------------------
QUESTION STYLE
------------------------------------------------------------

Prefer realistic situations over direct definitions.

For example, avoid:

"What is a router?"

Instead, create a practical situation where the learner needs
to understand what a router does.

The learner should need to think before choosing an answer.

------------------------------------------------------------
OPTIONS
------------------------------------------------------------

Provide exactly FOUR options.

Each option should be short.

All four options should sound believable.

Avoid obviously ridiculous answers.

Do not make the correct answer too obvious.

------------------------------------------------------------
ANSWER
------------------------------------------------------------

correct_option MUST be a number:

1
2
3
or
4

------------------------------------------------------------
EVENING EXPLANATION
------------------------------------------------------------

Write a useful explanation of approximately 30-55 words.

Explain WHY the correct answer is correct.

Do not simply repeat the answer.

Make the explanation understandable to someone who did not know
the concept before seeing the challenge.

------------------------------------------------------------
PRACTICAL CHALLENGE
------------------------------------------------------------

Add one short follow-up question that encourages the learner
to think about the concept in real life.

Keep it short.

------------------------------------------------------------
MORNING CLOSING
------------------------------------------------------------

The morning post will end with:

"What would you do in this situation?"

Do NOT put this sentence inside the JSON question.

------------------------------------------------------------
CORPORATE TONE
------------------------------------------------------------

This is an official Korlink Technologies training community.

The content should be:

Professional
Natural
Practical
Educational
Confident
Concise

No emojis.

No unnecessary formatting.

------------------------------------------------------------
OUTPUT
------------------------------------------------------------

Return ONLY valid JSON.

Use exactly this structure:

{{
  "question": "Engaging hook, realistic situation and question",
  "options": [
    "First option",
    "Second option",
    "Third option",
    "Fourth option"
  ],
  "correct_option": 1,
  "explanation": "Short practical explanation.",
  "practical_challenge": "One short follow-up question."
}}

Do not rename the fields.

Do not omit any field.

Do not put markdown around the JSON.

Do not write anything before or after the JSON.
"""

    for generation_attempt in range(1, 4):

        print(
            f"Question generation attempt "
            f"{generation_attempt}/3..."
        )

        try:

            raw = gemini_generate(
                prompt,
                attempts=5
            )

            cleaned = clean_json_response(
                raw
            )

            poll = json.loads(cleaned)

            poll = normalize_correct_option(
                poll
            )

            poll = validate_poll(
                poll
            )

            return poll

        except Exception as error:

            print(
                f"Invalid poll generated: {error}"
            )

            if generation_attempt < 3:
                time.sleep(2)

    raise RuntimeError(
        "Gemini failed to produce a valid "
        "daily challenge."
    )


# ============================================================
# QUESTION HISTORY
# ============================================================

def question_already_used(
    question,
    questions
):

    normalized = (
        question.strip().lower()
    )

    for item in questions:

        if not isinstance(item, dict):
            continue

        old_question = item.get(
            "question",
            ""
        )

        if (
            old_question.strip().lower()
            == normalized
        ):
            return True

    return False


def question_is_valid(item):

    if not isinstance(item, dict):
        return False

    required = [
        "question",
        "options",
        "correct_option",
        "explanation",
    ]

    if any(
        field not in item
        for field in required
    ):
        return False

    if (
        not isinstance(item["options"], list)
        or len(item["options"]) != 4
    ):
        return False

    try:
        answer = int(
            item["correct_option"]
        )

    except (TypeError, ValueError):
        return False

    if answer not in (1, 2, 3, 4):
        return False

    return True


def get_today_question(questions):

    today = today_string()

    for item in reversed(questions):

        if item.get("date") != today:
            continue

        # Ignore old or incomplete records.
        if question_is_valid(item):
            return item

        print(
            "Ignoring incomplete question "
            "record found in today's log."
        )

    return None


# ============================================================
# TELEGRAM
# ============================================================

def telegram_request(
    method,
    params
):

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/{method}"
    )

    encoded = urllib.parse.urlencode(
        params
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "Content-Type":
                "application/x-www-form-urlencoded"
        },
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:

        data = json.loads(
            response.read().decode(
                "utf-8"
            )
        )

        if not data.get("ok"):
            raise RuntimeError(
                f"Telegram API error: {data}"
            )

        return data


def send_telegram_message(text):

    return telegram_request(
        "sendMessage",
        {
            "chat_id":
                TELEGRAM_CHAT_ID,

            "text":
                text,

            "parse_mode":
                "Markdown",

            "disable_web_page_preview":
                "true",
        }
    )


# ============================================================
# MORNING MESSAGE
# ============================================================

def format_poll(
    poll,
    track
):

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
        "What would you do in this situation?/n/n"
        "Don't be afraid to get it wrong. The goal is to learn!/n"
    )


# ============================================================
# EVENING ANSWER
# ============================================================

def format_answer(
    poll,
    track
):

    correct_number = int(
        poll["correct_option"]
    )

    correct_answer = poll[
        "options"
    ][correct_number - 1]

    explanation = (
        poll["explanation"]
        .strip()
    )

    practical = (
        poll.get(
            "practical_challenge",
            ""
        )
        .strip()
    )

    message = (
        "*KORLINK TECHNOLOGIES*\n\n"
        "*DAILY CHALLENGE*\n\n"
        "*Answer & Explanation*\n"
        f"*Track:* {track['name']}\n\n"
        "*Correct Answer:*\n"
        f"{correct_number}. "
        f"{correct_answer}\n\n"
        "*Why?*\n"
        f"{explanation}"
    )

    if practical:

        message += (
            "\n\n"
            "*Practical Challenge:*\n"
            f"{practical}"
        )

    return message


# ============================================================
# SATURDAY
# ============================================================

def generate_saturday_message():

    prompt = """
Write a short professional Saturday message for the
Korlink Technologies training community.

Encourage learners to use some of their free time to review,
practise or reflect on what they learned during the week.

Keep it:
- professional
- natural
- concise
- practical
- encouraging without excessive motivation

Do not use emojis.
Do not use clichés.
Do not sound like AI-generated social media content.

Return only the message.
"""

    return gemini_generate(
        prompt
    ).strip()


# ============================================================
# SUNDAY
# ============================================================

def generate_sunday_message():

    prompt = """
Write a short professional Sunday reflection for the
Korlink Technologies training community.

Connect learning, consistency and personal development naturally.

The message should:
- be thoughtful
- be positive
- be professional
- be concise
- avoid excessive religious language
- avoid clichés
- avoid emojis
- sound naturally written by a real training organization

Return only the message.
"""

    return gemini_generate(
        prompt
    ).strip()


# ============================================================
# MORNING
# ============================================================

def run_morning():

    now = nigeria_now()

    print(
        "Nigeria time: "
        f"{now.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(
        f"Gemini model: {GEMINI_MODEL}"
    )

    print(
        "Run mode: morning"
    )

    weekday = now.weekday()

    if weekday not in WEEKDAY_TRACKS:

        print(
            "Today is a weekend. "
            "Morning weekday challenge skipped."
        )

        return

    track = WEEKDAY_TRACKS[
        weekday
    ]

    print(
        f"Generating {track['name']} "
        "challenge..."
    )

    questions = load_questions()

    existing = get_today_question(
        questions
    )

    if existing:

        print(
            "A valid challenge already "
            "exists for today."
        )

        poll = existing

    else:

        poll = None

        for attempt in range(1, 6):

            candidate = generate_poll(
                track
            )

            if not question_already_used(
                candidate["question"],
                questions
            ):

                poll = candidate
                break

            print(
                "Duplicate challenge detected. "
                "Generating another."
            )

        if poll is None:

            raise RuntimeError(
                "Could not generate a unique "
                "challenge."
            )

        questions.append(
            {
                "date":
                    today_string(),

                "weekday":
                    now.strftime("%A"),

                "school":
                    track["school"],

                "track":
                    track["name"],

                "question":
                    poll["question"],

                "options":
                    poll["options"],

                "correct_option":
                    poll["correct_option"],

                "explanation":
                    poll["explanation"],

                "practical_challenge":
                    poll.get(
                        "practical_challenge",
                        ""
                    ),

                "created_at":
                    now.isoformat(),
            }
        )

        save_questions(
            questions
        )

    message = format_poll(
        poll,
        track
    )

    send_telegram_message(
        message
    )

    print(
        "Morning challenge sent successfully."
    )


# ============================================================
# EVENING
# ============================================================

def run_evening():

    now = nigeria_now()

    print(
        "Nigeria time: "
        f"{now.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(
        f"Gemini model: {GEMINI_MODEL}"
    )

    print(
        "Run mode: evening"
    )

    weekday = now.weekday()

    if weekday not in WEEKDAY_TRACKS:

        print(
            "Today is a weekend. "
            "Evening answer skipped."
        )

        return

    track = WEEKDAY_TRACKS[
        weekday
    ]

    questions = load_questions()

    poll = get_today_question(
        questions
    )

    if not poll:

        raise RuntimeError(
            "No valid morning challenge "
            "was found for today."
        )

    message = format_answer(
        poll,
        track
    )

    send_telegram_message(
        message
    )

    posts = load_posts()

    posts.append(
        {
            "date":
                today_string(),

            "weekday":
                now.strftime("%A"),

            "type":
                "evening_answer",

            "track":
                track["name"],

            "message":
                message,

            "created_at":
                now.isoformat(),
        }
    )

    save_posts(
        posts
    )

    print(
        "Evening answer sent successfully."
    )


# ============================================================
# WEEKEND
# ============================================================

def run_weekend():

    now = nigeria_now()

    print(
        "Nigeria time: "
        f"{now.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(
        f"Gemini model: {GEMINI_MODEL}"
    )

    print(
        "Run mode: weekend"
    )

    weekday = now.weekday()

    if weekday == 5:

        print(
            "Generating Saturday message..."
        )

        body = generate_saturday_message()

        heading = "Saturday Boost"

    elif weekday == 6:

        print(
            "Generating Sunday message..."
        )

        body = generate_sunday_message()

        heading = "Sunday Reflection"

    else:

        print(
            "Weekend mode can only run "
            "on Saturday or Sunday."
        )

        return

    message = (
        "*KORLINK TECHNOLOGIES*\n\n"
        f"*{heading}*\n\n"
        f"{body}"
    )

    send_telegram_message(
        message
    )

    posts = load_posts()

    posts.append(
        {
            "date":
                today_string(),

            "weekday":
                now.strftime("%A"),

            "type":
                "weekend",

            "message":
                message,

            "created_at":
                now.isoformat(),
        }
    )

    save_posts(
        posts
    )

    print(
        f"{heading} sent successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    validate_environment()

    print("=" * 60)
    print("Korlink Technologies Ltd")
    print("Korlink Daily Challenge")
    print("=" * 60)

    if RUN_MODE == "morning":

        run_morning()

    elif RUN_MODE == "evening":

        run_evening()

    elif RUN_MODE == "weekend":

        run_weekend()

    else:

        raise ValueError(
            f"Invalid RUN_MODE: {RUN_MODE}. "
            "Use morning, evening or weekend."
        )


if __name__ == "__main__":
    main()
