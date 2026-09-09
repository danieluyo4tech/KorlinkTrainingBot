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
# KORLINK TECHNOLOGIES
# TRAINING UPDATE AI
# ============================================================

# Nigeria operates on West Africa Time (UTC+1).
# GitHub Actions itself runs on UTC, but all content dates
# and scheduling logic inside this application use Nigeria time.

NIGERIA_TZ = ZoneInfo("Africa/Lagos")


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")
TEXT_MODEL = os.getenv("GEMINI_MODEL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

if not TEXT_MODEL:
    raise RuntimeError("GEMINI_MODEL is not configured.")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not configured.")

if not TELEGRAM_CHAT_ID:
    raise RuntimeError("TELEGRAM_CHAT_ID is not configured.")


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# FILE DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LOG_DIR = BASE_DIR / "logs"

QUESTIONS_FILE = LOG_DIR / "questions.json"
POSTS_FILE = LOG_DIR / "posts.json"

LOG_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# WEEKLY TRAINING SCHEDULE
# ============================================================

WEEKDAY_TRACKS = {

    0: {
        "school": "School of Computing",
        "name": "Cybersecurity",
        "description": (
            "online safety, phishing, passwords, privacy, "
            "malware, scams and practical cybersecurity"
        ),
    },

    1: {
        "school": "School of Computing",
        "name": "Software Engineering",
        "description": (
            "websites, applications, coding, databases, "
            "debugging and practical software development"
        ),
    },

    2: {
        "school": "School of Technology",
        "name": "Smart Home Automation",
        "description": (
            "smart homes, IoT devices, sensors, smart lighting, "
            "security systems, controllers and practical home automation"
        ),
    },

    3: {
        "school": "School of Technology",
        "name": "Network Engineering",
        "description": (
            "Wi-Fi, routers, switches, IP addresses, "
            "Internet connections and network troubleshooting"
        ),
    },

    4: {
        "school": "School of Technology",
        "name": "Solar PV Design and Installation",
        "description": (
            "solar panels, batteries, inverters, "
            "charge controllers, solar system design "
            "and installation"
        ),
    },
}


# ============================================================
# NIGERIA DATE / TIME HELPERS
# ============================================================

def nigeria_now():
    """
    Return the current date and time in Nigeria.
    """

    return datetime.datetime.now(
        NIGERIA_TZ
    )


def nigeria_today():
    """
    Return today's date according to Nigeria time.
    """

    return nigeria_now().date()


def today_string():
    """
    Return today's Nigeria date as YYYY-MM-DD.
    """

    return str(
        nigeria_today()
    )


# ============================================================
# JSON STORAGE
# ============================================================

def load_json(file_path):

    if not file_path.exists():
        return []

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

            if isinstance(data, list):
                return data

            return []

    except Exception as error:

        print(
            f"Could not read {file_path}: {error}"
        )

        return []


def save_json(file_path, data):

    temporary_file = file_path.with_suffix(
        file_path.suffix + ".tmp"
    )

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Replace the old file only after the new file
    # has been written successfully.
    temporary_file.replace(
        file_path
    )


# ============================================================
# QUESTION STORAGE
# ============================================================

def load_questions():

    return load_json(
        QUESTIONS_FILE
    )


def save_question(question_data):

    questions = load_questions()

    # Prevent duplicate records for the same date.
    # This is useful if a manual workflow is accidentally
    # run twice on the same morning.
    questions = [
        item
        for item in questions
        if not (
            item.get("date") == question_data.get("date")
            and item.get("type") == question_data.get("type")
        )
    ]

    questions.append(
        question_data
    )

    # Keep the history manageable while retaining
    # enough information for duplicate checking.
    questions = questions[-100:]

    save_json(
        QUESTIONS_FILE,
        questions
    )


# ============================================================
# POST STORAGE
# ============================================================

def save_post(post_data):

    posts = load_json(
        POSTS_FILE
    )

    posts.append(
        post_data
    )

    # Keep the most recent 200 posts.
    posts = posts[-200:]

    save_json(
        POSTS_FILE,
        posts
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram_message(message):

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    data = urllib.parse.urlencode({
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": "true",
    }).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        if result.get("ok"):

            print(
                "Telegram message sent successfully."
            )

            return True

        print(
            "Telegram API error:"
        )

        print(result)

        return False

    except Exception as error:

        print(
            f"Telegram connection error: {error}"
        )

        return False


# ============================================================
# GEMINI RETRY HANDLER
# ============================================================

def generate_with_retry(
    prompt,
    max_attempts=5
):
    """
    Generate Gemini content with automatic retries
    for temporary API errors such as 429, 500 and 503.
    """

    delays = [
        5,
        10,
        20,
        40,
        60
    ]

    for attempt in range(
        1,
        max_attempts + 1
    ):

        try:

            print(
                f"Gemini attempt "
                f"{attempt}/{max_attempts}..."
            )

            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt
            )

            if not response:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            if not getattr(
                response,
                "text",
                None
            ):

                raise RuntimeError(
                    "Gemini returned no text."
                )

            return response

        except Exception as error:

            error_text = str(error).upper()

            temporary_error = any(
                code in error_text
                for code in [
                    "503",
                    "429",
                    "500",
                    "502",
                    "504",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                    "INTERNAL",
                    "TIMEOUT",
                    "OVERLOADED"
                ]
            )

            print(
                f"Gemini error: {error}"
            )

            if (
                not temporary_error
                or attempt == max_attempts
            ):

                print(
                    "Gemini generation failed."
                )

                raise

            delay = delays[
                min(
                    attempt - 1,
                    len(delays) - 1
                )
            ]

            print(
                "Gemini is temporarily unavailable."
            )

            print(
                f"Retrying in {delay} seconds..."
            )

            time.sleep(
                delay
            )

    raise RuntimeError(
        "Gemini generation failed after "
        "all retry attempts."
    )


# ============================================================
# CLEAN GEMINI JSON RESPONSE
# ============================================================

def clean_json_response(text):

    text = text.strip()

    # Remove Markdown code fences if Gemini adds them.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    return text


# ============================================================
# GENERATE WEEKDAY POLL
# ============================================================

def generate_poll(track):

    questions = load_questions()

    recent_questions = questions[-50:]

    history = "\n".join(
        f"- {item.get('question', '')}"
        for item in recent_questions
    )

    prompt = f"""
You are the official content editor and practical instructor
for Korlink Technologies Training Update.

Korlink Technologies is a professional technology training
company. Its training communication must sound authentic,
clear and professional.

TODAY'S TRAINING TRACK

School:
{track['school']}

Course:
{track['name']}

Focus:
{track['description']}

Create ONE practical multiple-choice question for a Telegram
learning community.

The audience includes beginners and developing learners.

The main purpose is STUDENT INTERACTION and practical learning.

EDITORIAL STYLE:

- Write like an experienced human instructor.
- Use natural, professional English.
- Keep the tone confident, approachable and educational.
- Make the situation realistic.
- Use practical examples from everyday life, work or training.
- Keep the question reasonably short.
- Avoid unnecessary technical jargon.
- Explain technical ideas in language beginners can understand.
- Make the content useful rather than promotional.
- Do not make it sound like an examination.
- Do not use childish language.
- Do not use exaggerated marketing language.
- Do not mention AI, Gemini, prompts or content generation.
- Do not use hashtags.
- Do not use unnecessary emojis.
- Do not use phrases such as:
  "Let's see who gets this!"
  "Tech warriors!"
  "Are you ready?"
  "Test your brain!"
  "Level up!"
  "Crush this!"
  or similar promotional phrases.

QUESTION RULES:

- Exactly four options.
- Only one correct answer.
- Do not reveal the answer in the question.
- Avoid trick questions.
- Avoid boring definition questions.
- Do not begin every question with "What is..."
- Prefer realistic situations.
- Vary the scenarios.
- Do not repeat previous questions.
- Do not create a substantially similar question.
- Make all four options plausible.
- Make the correct answer technically accurate.
- Make the question comfortable for beginners to attempt.

Examples of useful scenarios:

Cybersecurity:
A staff member receives a suspicious email asking them
to urgently confirm their account details.

Software Engineering:
A developer changes a piece of code and an existing feature
stops working.

Smart Home Automation:
A homeowner wants the lights to turn on automatically when
someone enters a room.

Network Engineering:
A laptop connects to Wi-Fi but cannot access the Internet.

Solar PV:
A solar system has adequate sunlight but the battery is
not charging properly.

PREVIOUS QUESTIONS:

{history}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "question": "...",
    "option_1": "...",
    "option_2": "...",
    "option_3": "...",
    "option_4": "...",
    "correct_option": 1,
    "simple_explanation": "...",
    "bonus_challenge": "..."
}}
"""

    response = generate_with_retry(
        prompt
    )

    text = clean_json_response(
        response.text
    )

    try:

        poll = json.loads(
            text
        )

    except json.JSONDecodeError as error:

        print(
            "Gemini returned invalid JSON:"
        )

        print(text)

        raise RuntimeError(
            f"Gemini returned invalid JSON: {error}"
        )

    required_fields = [
        "question",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "correct_option",
        "simple_explanation",
        "bonus_challenge"
    ]

    for field in required_fields:

        if field not in poll:

            raise RuntimeError(
                f"Gemini response is missing: {field}"
            )

    # Ensure correct_option is a valid integer.
    try:

        poll["correct_option"] = int(
            poll["correct_option"]
        )

    except (TypeError, ValueError):

        raise RuntimeError(
            "correct_option must be an integer."
        )

    if poll["correct_option"] not in [1, 2, 3, 4]:

        raise RuntimeError(
            "correct_option must be between 1 and 4."
        )

    return poll


# ============================================================
# DUPLICATE QUESTION CHECK
# ============================================================

def is_duplicate_question(
    new_question
):

    questions = load_questions()

    if not questions:
        return False

    previous = "\n".join(
        f"- {item.get('question', '')}"
        for item in questions[-50:]
    )

    prompt = f"""
Compare this new question with the previous questions.

NEW QUESTION:

{new_question}

PREVIOUS QUESTIONS:

{previous}

Determine whether the new question is substantially similar
to any previous question.

Consider:
- meaning
- scenario
- learning objective
- situation
- expected reasoning

Do not mark a question as duplicate merely because it covers
the same general course.

Return ONLY one word:

DUPLICATE

or

UNIQUE
"""

    response = generate_with_retry(
        prompt
    )

    result = response.text.strip().upper()

    return result == "DUPLICATE" or result.startswith(
        "DUPLICATE"
    )


# ============================================================
# MORNING POLL MESSAGE
# ============================================================

def format_poll(
    track,
    poll
):

    return f"""*KORLINK TECHNOLOGIES*

*Daily Challenge*

*Track:* {track['name']}

{poll['question']}

1. {poll['option_1']}
2. {poll['option_2']}
3. {poll['option_3']}
4. {poll['option_4']}

Share the option you consider correct and, if possible,
briefly explain your reasoning.
"""


# ============================================================
# FIND TODAY'S QUESTION
# ============================================================

def get_today_question():

    today = today_string()

    questions = load_questions()

    for question in reversed(
        questions
    ):

        if (
            question.get("date") == today
            and
            question.get("type") == "weekday_poll"
        ):

            return question

    return None


# ============================================================
# EVENING ANSWER MESSAGE
# ============================================================

def format_answer(
    question
):

    correct = question[
        "correct_option"
    ]

    option_key = (
        f"option_{correct}"
    )

    correct_text = question[
        option_key
    ]

    return f"""*KORLINK TECHNOLOGIES*

*Daily Challenge — Answer*

*Track:* {question['track']}

*Question:*

{question['question']}

*Correct Answer:*

{correct}. {correct_text}

*Explanation:*

{question['explanation']}

*Practical Challenge:*

{question['bonus_challenge']}
"""


# ============================================================
# SATURDAY MOTIVATION
# ============================================================

def generate_saturday():

    prompt = """
You are writing the official Saturday message for
Korlink Technologies Training Update.

Create a short professional weekend message for technology
students and aspiring professionals.

Maximum 70 words.

Focus on:
- consistency
- learning
- practice
- building projects
- professional growth

STYLE:

- Natural and authentic.
- Warm but professional.
- Sound like a real instructor or training organization.
- Avoid exaggerated motivation.
- Avoid clichés.
- Do not use famous quotes.
- Do not mention AI.
- Do not use hashtags.
- Use no more than one simple emoji, and only if it genuinely
  improves the message.
- End with ONE short question that encourages students to reply.

Use this format:

*KORLINK TECHNOLOGIES*

*Saturday Inspiration Note*

[short message]

*Reflection:* [short question]
"""

    response = generate_with_retry(
        prompt
    )

    return response.text.strip()


# ============================================================
# SUNDAY INSPIRATION
# ============================================================

def generate_sunday():

    prompt = """
You are writing the official Sunday message for
Korlink Technologies Training Update.

Create a short Sunday inspirational message for students.

Maximum 90 words.

The message may be inspired by a Gospel principle or a short
Bible reference.

Connect the message naturally to:
- learning
- wisdom
- discipline
- purpose
- using skills to help others
- preparing for a new week

STYLE:

- Warm and respectful.
- Professional and authentic.
- Suitable for a company training community.
- Do not preach harshly.
- Do not use excessive religious language.
- Do not reproduce a long Bible passage.
- Do not mention AI.
- Do not use hashtags.
- Avoid unnecessary emojis.
- End with ONE simple reflection question.

Use this format:

*KORLINK TECHNOLOGIES*

*Sunday Reflection*

[short message]

*Reflection:* [one short question]
"""

    response = generate_with_retry(
        prompt
    )

    return response.text.strip()


# ============================================================
# MORNING ENGINE
# ============================================================

def run_morning():

    today = nigeria_today()

    weekday = today.weekday()

    track = WEEKDAY_TRACKS.get(
        weekday
    )

    if not track:

        raise RuntimeError(
            "No training track configured for today."
        )

    print(
        f"Generating {track['name']} poll..."
    )

    poll = None

    for attempt in range(3):

        print(
            f"Question generation attempt "
            f"{attempt + 1}/3..."
        )

        candidate = generate_poll(
            track
        )

        if not is_duplicate_question(
            candidate["question"]
        ):

            poll = candidate

            break

        print(
            "A similar question was detected."
        )

        print(
            "Generating another question..."
        )

    if poll is None:

        raise RuntimeError(
            "Could not generate a unique "
            "question after three attempts."
        )

    message = format_poll(
        track,
        poll
    )

    record = {

        "date": str(today),

        "type": "weekday_poll",

        "school": track["school"],

        "track": track["name"],

        "question": poll["question"],

        "option_1": poll["option_1"],

        "option_2": poll["option_2"],

        "option_3": poll["option_3"],

        "option_4": poll["option_4"],

        "correct_option": poll[
            "correct_option"
        ],

        "explanation": poll[
            "simple_explanation"
        ],

        "bonus_challenge": poll[
            "bonus_challenge"
        ],
    }

    # Save the question before Telegram delivery.
    # This ensures the evening engine can retrieve
    # today's question even if the Telegram operation
    # is delayed or interrupted.
    save_question(
        record
    )

    save_post({

        "date": str(today),

        "type": "weekday_poll",

        "track": track["name"],

        "message": message,

    })

    print()
    print("=" * 70)
    print(
        "MORNING TELEGRAM MESSAGE"
    )
    print("=" * 70)

    print(message)

    print()
    print(
        "Sending to Telegram..."
    )

    if not send_telegram_message(
        message
    ):

        raise RuntimeError(
            "Telegram message could not be sent."
        )

    print()
    print(
        "Morning poll completed successfully."
    )


# ============================================================
# EVENING ENGINE
# ============================================================

def run_evening():

    question = get_today_question()

    if not question:

        raise RuntimeError(
            "No poll was found for today. "
            "The morning poll may not have completed successfully."
        )

    message = format_answer(
        question
    )

    save_post({

        "date": today_string(),

        "type": "weekday_answer",

        "track": question["track"],

        "message": message,

    })

    print()
    print("=" * 70)
    print(
        "EVENING TELEGRAM MESSAGE"
    )
    print("=" * 70)

    print(message)

    print()
    print(
        "Sending to Telegram..."
    )

    if not send_telegram_message(
        message
    ):

        raise RuntimeError(
            "Telegram message could not be sent."
        )

    print()
    print(
        "Evening answer completed successfully."
    )


# ============================================================
# WEEKEND ENGINE
# ============================================================

def run_weekend():

    today = nigeria_today()

    if today.weekday() == 5:

        print(
            "Generating Saturday learning message..."
        )

        message = generate_saturday()

        post_type = (
            "saturday_motivation"
        )

    elif today.weekday() == 6:

        print(
            "Generating Sunday reflection..."
        )

        message = generate_sunday()

        post_type = (
            "sunday_inspiration"
        )

    else:

        raise RuntimeError(
            "Weekend mode can only run on Saturday or Sunday."
        )

    save_post({

        "date": str(today),

        "type": post_type,

        "message": message,

    })

    print()
    print("=" * 70)
    print(
        "WEEKEND TELEGRAM MESSAGE"
    )
    print("=" * 70)

    print(message)

    print()
    print(
        "Sending to Telegram..."
    )

    if not send_telegram_message(
        message
    ):

        raise RuntimeError(
            "Telegram message could not be sent."
        )

    print()
    print(
        "Weekend message completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    mode = os.getenv(
        "RUN_MODE",
        "morning"
    ).lower().strip()

    current_time = nigeria_now()

    today = current_time.date()

    print()
    print("=" * 70)
    print(
        "KORLINK TECHNOLOGIES"
    )
    print(
        "Training Update AI"
    )
    print("=" * 70)

    print(
        f"Nigeria time: "
        f"{current_time.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(
        f"Gemini model: {TEXT_MODEL}"
    )

    print(
        f"Run mode: {mode}"
    )

    print()

    # --------------------------------------------------------
    # WEEKDAY MORNING
    # --------------------------------------------------------

    if mode == "morning":

        if today.weekday() <= 4:

            run_morning()

        else:

            print(
                "Morning weekday poll is not scheduled "
                "for Saturday or Sunday."
            )

    # --------------------------------------------------------
    # WEEKDAY EVENING
    # --------------------------------------------------------

    elif mode == "evening":

        if today.weekday() <= 4:

            run_evening()

        else:

            print(
                "Evening weekday answer is not scheduled "
                "for Saturday or Sunday."
            )

    # --------------------------------------------------------
    # WEEKEND
    # --------------------------------------------------------

    elif mode == "weekend":

        if today.weekday() >= 5:

            run_weekend()

        else:

            print(
                "Weekend mode is only available "
                "on Saturday or Sunday."
            )

    # --------------------------------------------------------
    # INVALID MODE
    # --------------------------------------------------------

    else:

        raise RuntimeError(
            f"Unknown RUN_MODE: {mode}. "
            f"Expected morning, evening or weekend."
        )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
