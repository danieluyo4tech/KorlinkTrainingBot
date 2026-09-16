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
# TEXT HELPERS
# ============================================================

def word_count(text):
    return len(
        str(text).strip().split()
    )


def normalize_text(text):
    return " ".join(
        str(text).strip().lower().split()
    )


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

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    if (
        not isinstance(poll["question"], str)
        or not poll["question"].strip()
    ):
        raise ValueError(
            "Question is empty."
        )

    question = poll["question"].strip()

    # Practical challenges must be short.
    if word_count(question) < 6:
        raise ValueError(
            "Challenge is too short."
        )

    if word_count(question) > 25:
        raise ValueError(
            "Challenge is too long."
        )

    # Reject common exam-style wording.
    forbidden_phrases = [
        "which of the following",
        "what is the definition",
        "define ",
        "what does ",
        "which protocol",
        "which component",
        "which technology",
        "best describes",
        "select the correct",
        "choose the correct",
        "what is meant by",
        "according to the definition",
    ]

    lowered_question = question.lower()

    for phrase in forbidden_phrases:
        if phrase in lowered_question:
            raise ValueError(
                "Challenge sounds like an exam question."
            )

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    options = poll["options"]

    if not isinstance(options, list):
        raise ValueError(
            "Options must be a list."
        )

    if len(options) != 4:
        raise ValueError(
            "Exactly four options are required."
        )

    normalized_options = []

    for option in options:

        if (
            not isinstance(option, str)
            or not option.strip()
        ):
            raise ValueError(
                "Every option must contain text."
            )

        clean_option = option.strip()

        # Keep poll options short enough to scan quickly.
        if word_count(clean_option) > 6:
            raise ValueError(
                "An option is too long."
            )

        normalized_option = normalize_text(
            clean_option
        )

        if normalized_option in normalized_options:
            raise ValueError(
                "Duplicate poll options found."
            )

        normalized_options.append(
            normalized_option
        )

    # --------------------------------------------------------
    # CORRECT ANSWER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------------

    if (
        not isinstance(poll["explanation"], str)
        or not poll["explanation"].strip()
    ):

        raise ValueError(
            "Explanation is empty."
        )

    explanation = poll["explanation"].strip()

    if word_count(explanation) < 15:
        raise ValueError(
            "Explanation is too short."
        )

    if word_count(explanation) > 70:
        raise ValueError(
            "Explanation is too long."
        )

    # --------------------------------------------------------
    # PRACTICAL CHALLENGE
    # --------------------------------------------------------

    practical = poll.get(
        "practical_challenge",
        ""
    )

    if not isinstance(practical, str):
        practical = str(practical)

    practical = practical.strip()

    if practical and word_count(practical) > 30:
        raise ValueError(
            "Practical challenge is too long."
        )

    poll["question"] = question
    poll["options"] = [
        option.strip()
        for option in options
    ]
    poll["correct_option"] = correct_option
    poll["explanation"] = explanation
    poll["practical_challenge"] = practical

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
# DAILY PRACTICAL CHALLENGE GENERATOR
# ============================================================

def generate_poll(track, questions=None):

    if questions is None:
        questions = []

    # --------------------------------------------------------
    # RECENT HISTORY
    # --------------------------------------------------------
    #
    # We send recent challenges from this track to Gemini.
    # This helps it avoid repeating the same situation,
    # concept or structure.
    #

    recent_items = []

    for item in questions[-180:]:

        if not isinstance(item, dict):
            continue

        if item.get("track") != track["name"]:
            continue

        question = item.get(
            "question",
            ""
        ).strip()

        if not question:
            continue

        recent_items.append(
            {
                "question": question,
                "options": item.get(
                    "options",
                    []
                ),
                "explanation": item.get(
                    "explanation",
                    ""
                ),
            }
        )

    recent_items = recent_items[-60:]

    if recent_items:

        recent_challenges_text = []

        for index, item in enumerate(
            recent_items,
            start=1
        ):

            options = item.get(
                "options",
                []
            )

            options_text = " | ".join(
                str(option)
                for option in options
            )

            recent_challenges_text.append(
                f"{index}. "
                f"{item['question']} "
                f"[Options: {options_text}]"
            )

        recent_questions_text = "\n".join(
            recent_challenges_text
        )

    else:

        recent_questions_text = (
            "No previous challenges are available "
            "for this training track."
        )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are the practical instructor and official daily training
content writer for Korlink Technologies Ltd.

Create ONE short multiple-choice DAILY CHALLENGE for:

School: {track["school"]}
Training Track: {track["name"]}
Training Area: {track["description"]}

The challenge will be posted in a professional Korlink training
community.

============================================================
CORE IDEA
============================================================

THIS IS NOT AN EXAM.

Do not write a textbook question.

Do not write a classroom examination question.

Create a small, realistic situation that people can understand
quickly and relate to from everyday life.

A person with little or no technical background should be able
to understand the challenge without needing technical knowledge
before reading the explanation.

The challenge should make someone stop for a few seconds and
think:

"What would I do?"

"What would I check?"

"What makes sense here?"

"What would help?"

"What should happen?"

The technical lesson should come mainly from the explanation.

============================================================
REAL-LIFE CHALLENGE
============================================================

Base the challenge on something that could genuinely happen in
ordinary life.

It can involve technology being used at home, at work, at school,
in a small business or during normal daily activities.

Keep the situation simple.

Do not create a long story.

Do not create a complicated technical setup.

Do not assume the learner is an engineer.

Do not require specialist knowledge just to understand the
question.

The learner should be able to read the challenge and options
within a few seconds.

============================================================
QUESTION STYLE
============================================================

Write ONE short challenge.

Aim for approximately 8-22 words.

Maximum 25 words.

Use natural everyday language.

Good style:

"Your phone connects to Wi-Fi, but nothing loads. What would you check?"

"Your smart light stops responding. What would you check first?"

"You receive a message saying your account will be blocked today. What should you do?"

"An app suddenly stops working after an update. What would you try?"

"You want a light to come on automatically when someone enters. What could help?"

Bad style:

"Which protocol is responsible for..."

"Which of the following best describes..."

"What is the definition of..."

"Which component is primarily responsible for..."

"Select the correct answer."

"According to networking principles..."

Do not use examination language.

============================================================
OPTIONS
============================================================

Provide exactly FOUR options.

Each option must normally be 1-4 words.

Maximum 6 words.

Keep them extremely easy to scan.

Do not write sentences as options.

Do not make one option obviously longer or more technical than
the others.

Wrong answers should still make reasonable sense to an ordinary
person.

There must be ONE clearly correct answer.

Example style:

"Check Wi-Fi"
"Change wallpaper"
"Restart the clock"
"Move the chair"

Do not copy this example. Create options that actually fit
today's challenge.

============================================================
TECHNICAL LEARNING
============================================================

The challenge itself should stay simple.

The underlying technical idea can be meaningful and accurate.

Use the evening explanation to introduce the technical concept
naturally.

For example, a simple everyday situation can teach:

- why a network connection matters
- why software updates can affect an app
- how a sensor supports automation
- why account messages can be suspicious
- how a battery and inverter behave
- why a device may lose communication
- why a system needs a particular condition before acting

Do not force technical jargon into the challenge.

============================================================
VARIETY
============================================================

Do NOT use a fixed topic list.

Do NOT follow a topic sequence.

Do NOT create a rotation inside the training track.

Think independently about the entire training track.

Every day should feel genuinely different.

Read the recent challenges below before creating today's one.

Do not repeat:

- the same situation
- the same device
- the same action
- the same learning point
- the same scenario family
- the same opening
- the same question structure

Changing only a few words is NOT enough.

If a recent challenge was about a device losing connection,
do not simply create another device-loss question with a
different device.

If a recent challenge was about checking Wi-Fi, deliberately
look for another useful concept.

If recent challenges repeatedly use "What would you check
first?", use a naturally different structure when appropriate.

Variety is required.

============================================================
SMART HOME AUTOMATION
============================================================

For Smart Home Automation, DO NOT repeatedly write about:

- smart lights
- motion sensors
- arriving home
- lights turning on
- phone connection

Those are only examples and must not become a rotation.

Think broadly and independently about Smart Home Automation.

A challenge may naturally come from device behaviour,
automation logic, communication, security, reliability,
troubleshooting, comfort, energy use, sensors, controllers,
scheduling or another relevant area.

But do NOT try to cover these areas one by one.

Choose a genuinely different idea each day.

The learner should still understand the situation without
specialist knowledge.

============================================================
OTHER TRACKS
============================================================

Cybersecurity:

Use ordinary digital situations such as messages, accounts,
passwords, links, devices, privacy or suspicious activity.

Do not teach offensive hacking.

Keep the challenge focused on safe awareness and protection.

Software Engineering:

Use everyday software behaviour such as an app failing,
an update causing a problem, information not saving, repeated
errors or a feature behaving unexpectedly.

The challenge should not require programming knowledge just to
understand it.

Network Engineering:

Use familiar connectivity situations involving phones,
computers, Wi-Fi, internet access or connected devices.

Keep networking concepts underneath the simple situation.

Solar PV Design and Installation:

Use safe everyday observations involving solar power, battery
behaviour, charging, energy use, inverter behaviour or system
performance.

Do not ask learners to touch electrical wiring, terminals,
batteries, exposed conductors or live equipment.

Do not give dangerous installation instructions.

============================================================
HOOK
============================================================

The challenge itself should be the hook.

Do not add unnecessary introductions such as:

"Here is today's challenge..."

"Are you ready?"

"Let's test your knowledge!"

"Only experts can answer!"

"Can you crack this?"

"Imagine..."

"You are working as..."

Avoid repeated "Imagine..." openings.

Start naturally.

============================================================
EXPLANATION
============================================================

Write a concise explanation of approximately 25-55 words.

Explain why the correct answer makes sense.

Teach the underlying technical idea naturally.

Do not simply repeat the correct option.

Do not turn the explanation into a textbook.

Write so that a beginner learns something useful.

============================================================
PRACTICAL CHALLENGE
============================================================

Add one short SAFE practical action or observation related to the
lesson.

It should be something the learner can safely notice, check,
think through or practise in everyday life.

It should NOT be another multiple-choice question.

Examples of the style:

"Check how many devices use your home Wi-Fi."

"Look at one recent message and check who sent it."

"Notice what happens when an app is reopened after closing."

"Observe whether your smart device responds when its connection
changes."

For solar or electrical topics, keep this strictly observational.
Do not instruct the learner to open, wire, touch or modify
electrical equipment.

============================================================
KORLINK CORPORATE STYLE
============================================================

Write like an experienced instructor working for a real
technology training organization.

Use:

- professional language
- natural business English
- simple wording
- practical situations
- concise presentation
- clear technical reasoning
- confident instructor tone

Avoid:

- excessive emojis
- hype
- childish wording
- fake excitement
- motivational clichés
- clickbait
- long introductions
- unnecessary technical jargon
- AI-style social media language

Do not say:

"Let's see who gets this!"

"Are you ready?"

"Test your IQ!"

"Only experts can answer!"

"Can you crack this?"

"Let's challenge your brain!"

============================================================
RECENT CHALLENGES
============================================================

These are recent challenges from this same training track:

{recent_questions_text}

Today's challenge MUST be meaningfully different from them.

============================================================
FINAL QUALITY CHECK
============================================================

Before returning the JSON, silently check:

1. Can a non-technical person understand the challenge immediately?
2. Is it based on a realistic everyday situation?
3. Is the challenge short?
4. Are all four options short?
5. Is there only one clearly correct answer?
6. Does the challenge avoid exam-style wording?
7. Does it avoid repeating recent situations?
8. Does it avoid repeating the same underlying concept?
9. Is the explanation where the technical teaching happens?
10. Is the practical follow-up safe and useful?

If any answer is NO, rewrite the challenge before returning it.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "question": "Short practical challenge",
  "options": [
    "Short option",
    "Short option",
    "Short option",
    "Short option"
  ],
  "correct_option": 1,
  "explanation": "Short useful explanation.",
  "practical_challenge": "Short safe practical action or observation."
}}

Do not rename fields.

Do not omit fields.

Do not add fields.

Do not use markdown.

Do not write anything before or after the JSON.
"""

    # --------------------------------------------------------
    # GENERATE AND VALIDATE
    # --------------------------------------------------------

    for generation_attempt in range(1, 4):

        print(
            f"Challenge generation attempt "
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

            poll = json.loads(
                cleaned
            )

            poll = normalize_correct_option(
                poll
            )

            poll = validate_poll(
                poll
            )

            if question_already_used(
                poll["question"],
                questions
            ):

                print(
                    "Generated challenge already exists. "
                    "Requesting another."
                )

                continue

            return poll

        except Exception as error:

            print(
                f"Invalid challenge generated: {error}"
            )

            if generation_attempt < 3:
                time.sleep(2)

    raise RuntimeError(
        "Gemini failed to produce a valid and unique "
        f"{track['name']} daily challenge."
    )


# ============================================================
# QUESTION HISTORY
# ============================================================

def question_already_used(
    question,
    questions
):

    normalized = normalize_text(
        question
    )

    for item in questions:

        if not isinstance(item, dict):
            continue

        old_question = item.get(
            "question",
            ""
        )

        old_normalized = normalize_text(
            old_question
        )

        if old_normalized == normalized:
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
        not isinstance(item["question"], str)
        or not item["question"].strip()
    ):
        return False

    if (
        not isinstance(item["options"], list)
        or len(item["options"]) != 4
    ):
        return False

    for option in item["options"]:

        if (
            not isinstance(option, str)
            or not option.strip()
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

    if (
        not isinstance(item["explanation"], str)
        or not item["explanation"].strip()
    ):
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
            "Ignoring incomplete challenge "
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
        "What would you do in this situation?\n"
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
        "daily challenge..."
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
                track,
                questions
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
        "Morning daily challenge sent successfully."
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
