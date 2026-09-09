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
# KORLINK TRAINING UPDATE AI
# ============================================================

APP_NAME = "Korlink Daily Challenge"

API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
RUN_MODE = os.getenv("RUN_MODE", "morning").lower().strip()

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
            "cybersecurity awareness, phishing, passwords, "
            "social engineering, privacy, threats and safe digital practices"
        ),
    },

    1: {
        "school": "School of Computing",
        "name": "Software Engineering",
        "description": (
            "software development, programming, debugging, "
            "testing, databases, APIs and software development practices"
        ),
    },

    2: {
        "school": "School of Technology",
        "name": "Smart Home Automation",
        "description": (
            "smart homes, IoT devices, sensors, lighting, "
            "security systems, controllers and practical home automation"
        ),
    },

    3: {
        "school": "School of Technology",
        "name": "Network Engineering",
        "description": (
            "computer networks, routers, switches, Wi-Fi, IP addressing, "
            "connectivity, troubleshooting and network security"
        ),
    },

    4: {
        "school": "School of Technology",
        "name": "Solar PV Design and Installation",
        "description": (
            "solar panels, batteries, charge controllers, inverters, "
            "system sizing, installation and practical solar troubleshooting"
        ),
    },
}


# ============================================================
# TIME HELPERS
# ============================================================

def nigeria_now():
    return datetime.datetime.now(NIGERIA_TZ)


def nigeria_today():
    return nigeria_now().date()


def today_string():
    return nigeria_today().isoformat()


# ============================================================
# VALIDATION
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

    except (json.JSONDecodeError, OSError):
        print(f"Warning: Could not read {file_path}. Starting fresh.")
        return default


def save_json(file_path, data):
    """
    Atomic JSON write to reduce the chance of corrupted log files.
    """
    temporary_file = file_path.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_file.replace(file_path)


def load_questions():
    return load_json(QUESTIONS_FILE, [])


def save_questions(questions):
    # Keep the log manageable.
    save_json(QUESTIONS_FILE, questions[-300:])


def load_posts():
    return load_json(POSTS_FILE, [])


def save_posts(posts):
    save_json(POSTS_FILE, posts[-300:])


# ============================================================
# GEMINI
# ============================================================

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)


def clean_json_response(text):
    """
    Cleans Gemini output when it returns JSON wrapped in markdown.
    """

    if not text:
        raise ValueError("Gemini returned an empty response.")

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Handle accidental text before/after the JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text


def normalize_correct_option(poll):
    """
    Gemini sometimes returns a different field name.
    Normalize common alternatives to correct_option.
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

    # Handle numeric strings.
    if isinstance(value, str):
        value = value.strip()

        if value.isdigit():
            value = int(value)

        else:
            # Handle "Option 2"
            lowered = value.lower()

            if lowered.startswith("option "):
                number = lowered.replace("option ", "").strip()

                if number.isdigit():
                    value = int(number)

            # Handle answer text matching one of the choices.
            if isinstance(value, str) and "options" in poll:
                for index, option in enumerate(poll["options"], start=1):
                    if value.lower() == str(option).lower():
                        value = index
                        break

    poll["correct_option"] = value

    return poll


def validate_poll(poll):
    """
    Ensures Gemini produced the structure required by the bot.
    """

    if not isinstance(poll, dict):
        raise ValueError("Gemini response is not a JSON object.")

    required_fields = [
        "question",
        "options",
        "correct_option",
        "explanation",
    ]

    for field in required_fields:
        if field not in poll:
            raise ValueError(
                f"Gemini response is missing: {field}"
            )

    question = poll["question"]
    options = poll["options"]
    correct_option = poll["correct_option"]
    explanation = poll["explanation"]

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question is empty.")

    if not isinstance(options, list) or len(options) != 4:
        raise ValueError(
            "Poll must contain exactly four options."
        )

    for option in options:
        if not isinstance(option, str) or not option.strip():
            raise ValueError("Poll contains an invalid option.")

    try:
        correct_option = int(correct_option)
    except (TypeError, ValueError):
        raise ValueError(
            "correct_option must be a number from 1 to 4."
        )

    if correct_option not in [1, 2, 3, 4]:
        raise ValueError(
            "correct_option must be between 1 and 4."
        )

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Explanation is empty.")

    poll["correct_option"] = correct_option

    # Optional field.
    if "practical_challenge" not in poll:
        poll["practical_challenge"] = ""

    if not isinstance(poll["practical_challenge"], str):
        poll["practical_challenge"] = str(
            poll["practical_challenge"]
        )

    return poll


def gemini_generate(prompt, attempts=5):
    """
    Calls Gemini with retries.
    """

    if client is None:
        raise RuntimeError("Gemini client is not configured.")

    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            print(
                f"Gemini attempt {attempt}/{attempts}..."
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            if not response:
                raise RuntimeError(
                    "Gemini returned no response."
                )

            text = getattr(response, "text", None)

            if not text:
                raise RuntimeError(
                    "Gemini response contained no text."
                )

            return text

        except Exception as error:
            last_error = error

            print(
                f"Gemini attempt {attempt} failed: {error}"
            )

            if attempt < attempts:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"Gemini generation failed after {attempts} attempts: "
        f"{last_error}"
    )


# ============================================================
# POLL GENERATION
# ============================================================

def generate_poll(track):
    """
    Generates a realistic, engaging Korlink Daily Challenge.

    The question should be:
    - Interesting from the first sentence
    - Based on a familiar real-world situation
    - Easy to understand
    - Useful to both beginners and technical learners
    - Medium length
    - Not a mini-article
    """

    prompt = f"""
You are writing a professional daily technology challenge for
Korlink Technologies Ltd.

PROGRAM:
Korlink Daily Challenge

TRAINING TRACK:
{track["name"]}

SCHOOL:
{track["school"]}

TOPIC AREA:
{track["description"]}

Create ONE multiple-choice challenge.

IMPORTANT WRITING STYLE:

The challenge must feel like something an experienced instructor
would naturally write for a real training community.

Do NOT make it sound like AI-generated content.

Do NOT use:
- excessive motivational language
- childish wording
- exaggerated hype
- unnecessary emojis
- long introductions
- textbook-style definitions
- complicated technical jargon
- artificial phrases such as "Let's see who gets this!"

The challenge should be interesting to someone who has no technical
background while still teaching a real technical concept.

Use this structure:

REAL-LIFE SITUATION -> QUESTION -> FOUR CHOICES

The opening should create curiosity.

Good openings can sound like:
"You receive..."
"You notice..."
"Imagine you are..."
"Your customer reports..."
"Your Wi-Fi suddenly..."
"You want your..."
"After making a small change..."

QUESTION LENGTH:

Aim for approximately 18-35 words.

Use one or two sentences maximum.

Give enough context to make the situation interesting, but do not
write a story or mini-article.

The reader should be able to understand the question quickly.

OPTIONS:

Provide exactly four choices.

Each option should be short, natural and believable.

Avoid obviously silly answers.

The choices should be numbered conceptually from 1 to 4.

EXPLANATION:

Explain the correct answer in approximately 25-50 words.

Keep the explanation practical and easy to understand.

PRACTICAL CHALLENGE:

Provide one short follow-up question that can encourage discussion.
Do not make it complicated.

IMPORTANT:

The correct answer MUST be represented by a number from 1 to 4.

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
  "question": "The challenge question",
  "options": [
    "First option",
    "Second option",
    "Third option",
    "Fourth option"
  ],
  "correct_option": 1,
  "explanation": "Short practical explanation.",
  "practical_challenge": "Short follow-up question."
}}

Do not rename any of these fields.
Do not omit correct_option.
Do not include markdown.
Do not include text outside the JSON.
"""

    # Generate several times if Gemini produces invalid structure.
    for generation_attempt in range(1, 4):

        print(
            f"Question generation attempt "
            f"{generation_attempt}/3..."
        )

        try:
            raw_response = gemini_generate(
                prompt,
                attempts=5,
            )

            cleaned = clean_json_response(raw_response)

            poll = json.loads(cleaned)

            poll = normalize_correct_option(poll)

            poll = validate_poll(poll)

            return poll

        except Exception as error:
            print(
                f"Invalid poll generated: {error}"
            )

            if generation_attempt < 3:
                time.sleep(2)

    raise RuntimeError(
        "Gemini failed to produce a valid poll after multiple attempts."
    )


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def question_already_used(question, questions):
    normalized = question.strip().lower()

    for item in questions:
        if not isinstance(item, dict):
            continue

        old_question = item.get("question", "")

        if old_question.strip().lower() == normalized:
            return True

    return False


def get_today_question(questions):
    today = today_string()

    for item in reversed(questions):
        if item.get("date") == today:
            return item

    return None


# ============================================================
# TELEGRAM
# ============================================================

def telegram_request(method, params):
    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/{method}"
    )

    encoded = urllib.parse.urlencode(params).encode("utf-8")

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
        timeout=30,
    ) as response:

        response_data = response.read().decode("utf-8")

        result = json.loads(response_data)

        if not result.get("ok"):
            raise RuntimeError(
                f"Telegram API error: {result}"
            )

        return result


def send_telegram_message(text):
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
# POLL FORMATTING
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
        "What would you choose? "
        "If possible, share your reason."
    )


# ============================================================
# ANSWER FORMATTING
# ============================================================

def format_answer(poll, track):
    correct_number = poll["correct_option"]
    correct_answer = poll["options"][correct_number - 1]

    explanation = poll["explanation"].strip()
    practical = poll.get(
        "practical_challenge",
        ""
    ).strip()

    message = (
        "*KORLINK TECHNOLOGIES*\n\n"
        "*DAILY CHALLENGE*\n\n"
        "*Answer & Explanation*\n"
        f"*Track:* {track['name']}\n\n"
        "*Correct Answer:*\n"
        f"{correct_number}. {correct_answer}\n\n"
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
# WEEKEND CONTENT
# ============================================================

def generate_saturday_message():
    prompt = """
Write a short professional Saturday message for the Korlink
Technologies training community.

The message should:
- Be natural and authentic
- Sound like a real training organization
- Encourage learners to review what they learned during the week
- Be useful to both beginners and experienced learners
- Avoid excessive motivational language
- Avoid emojis
- Avoid clichés
- Be concise

Return only the message text.
"""

    return gemini_generate(prompt).strip()


def generate_sunday_message():
    prompt = """
Write a short professional Sunday reflection for the Korlink
Technologies training community.

The message should:
- Have a thoughtful and positive tone
- Connect personal growth with learning technology
- Be suitable for a professional training community
- Be concise
- Avoid excessive religious preaching
- Avoid hype
- Avoid emojis
- Sound naturally written by a real organization

Return only the message text.
"""

    return gemini_generate(prompt).strip()


# ============================================================
# MORNING RUN
# ============================================================

def run_morning():
    now = nigeria_now()

    print(
        f"Nigeria time: "
        f"{now.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(f"Gemini model: {GEMINI_MODEL}")
    print("Run mode: morning")

    weekday = now.weekday()

    # Monday-Friday.
    if weekday not in WEEKDAY_TRACKS:
        print(
            "Today is a weekend. "
            "Morning challenge is not required."
        )
        return

    track = WEEKDAY_TRACKS[weekday]

    print(
        f"Generating {track['name']} poll..."
    )

    questions = load_questions()

    existing_today = get_today_question(questions)

    if existing_today:
        print(
            "A question already exists for today. "
            "Using the existing question."
        )

        poll = existing_today

    else:
        poll = None

        # Generate a new question and reject duplicates.
        for attempt in range(1, 6):

            candidate = generate_poll(track)

            if not question_already_used(
                candidate["question"],
                questions,
            ):
                poll = candidate
                break

            print(
                "Generated question already exists. "
                "Generating another one..."
            )

        if poll is None:
            raise RuntimeError(
                "Could not generate a unique question."
            )

        questions.append(
            {
                "date": today_string(),
                "weekday": now.strftime("%A"),
                "school": track["school"],
                "track": track["name"],
                "question": poll["question"],
                "options": poll["options"],
                "correct_option": poll["correct_option"],
                "explanation": poll["explanation"],
                "practical_challenge":
                    poll.get("practical_challenge", ""),
                "created_at": now.isoformat(),
            }
        )

        save_questions(questions)

    message = format_poll(
        poll,
        track,
    )

    send_telegram_message(message)

    print(
        "Morning challenge sent successfully."
    )


# ============================================================
# EVENING RUN
# ============================================================

def run_evening():
    now = nigeria_now()

    print(
        f"Nigeria time: "
        f"{now.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(f"Gemini model: {GEMINI_MODEL}")
    print("Run mode: evening")

    weekday = now.weekday()

    if weekday not in WEEKDAY_TRACKS:
        print(
            "Today is a weekend. "
            "Evening answer is not required."
        )
        return

    track = WEEKDAY_TRACKS[weekday]

    questions = load_questions()

    poll = get_today_question(questions)

    if not poll:
        raise RuntimeError(
            "No morning challenge was found for today."
        )

    message = format_answer(
        poll,
        track,
    )

    send_telegram_message(message)

    posts = load_posts()

    posts.append(
        {
            "date": today_string(),
            "weekday": now.strftime("%A"),
            "type": "evening_answer",
            "track": track["name"],
            "message": message,
            "created_at": now.isoformat(),
        }
    )

    save_posts(posts)

    print(
        "Evening answer sent successfully."
    )


# ============================================================
# WEEKEND RUN
# ============================================================

def run_weekend():
    now = nigeria_now()

    print(
        f"Nigeria time: "
        f"{now.strftime('%A, %d %B %Y %H:%M:%S WAT')}"
    )

    print(f"Gemini model: {GEMINI_MODEL}")
    print("Run mode: weekend")

    weekday = now.weekday()

    if weekday == 5:
        print("Generating Saturday message...")

        message_body = generate_saturday_message()

        heading = "Saturday Boost"

    elif weekday == 6:
        print("Generating Sunday message...")

        message_body = generate_sunday_message()

        heading = "Sunday Reflection"

    else:
        print(
            "Weekend mode can only run on Saturday or Sunday."
        )
        return

    message = (
        "*KORLINK TECHNOLOGIES*\n\n"
        f"*{heading}*\n\n"
        f"{message_body}"
    )

    send_telegram_message(message)

    posts = load_posts()

    posts.append(
        {
            "date": today_string(),
            "weekday": now.strftime("%A"),
            "type": "weekend",
            "message": message,
            "created_at": now.isoformat(),
        }
    )

    save_posts(posts)

    print(
        f"{heading} sent successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():
    validate_environment()

    print("=" * 60)
    print("Korlink Daily Challenge")
    print("Korlink Technologies Ltd")
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
