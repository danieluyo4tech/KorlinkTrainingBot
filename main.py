import os
import json
import datetime
import time
import urllib.request
import urllib.parse
from pathlib import Path

from google import genai


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
        "name": "Network Engineering",
        "description": (
            "Wi-Fi, routers, switches, IP addresses, "
            "Internet connections and network troubleshooting"
        ),
    },

    3: {
        "school": "School of Technology",
        "name": "Smart Tech and Automation",
        "description": (
            "IoT, sensors, smart devices, automation, "
            "controllers and practical automation"
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
            f"⚠️ Could not read {file_path}: {error}"
        )

        return []


def save_json(file_path, data):

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
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

    questions.append(
        question_data
    )

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
                "✅ Telegram message sent successfully."
            )

            return True

        print(
            "❌ Telegram API error:"
        )

        print(result)

        return False

    except Exception as error:

        print(
            f"❌ Telegram connection error: {error}"
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
                f"🤖 Gemini attempt "
                f"{attempt}/{max_attempts}..."
            )

            # IMPORTANT:
            # Actually call Gemini here.
            # Do NOT call generate_with_retry()
            # from inside itself.

            response = client.models.generate_content(
                model=TEXT_MODEL,
                contents=prompt
            )

            if not response:

                raise RuntimeError(
                    "Gemini returned an empty response."
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
                f"⚠️ Gemini error: {error}"
            )

            if (
                not temporary_error
                or attempt == max_attempts
            ):

                print(
                    "❌ Gemini generation failed."
                )

                raise

            delay = delays[
                min(
                    attempt - 1,
                    len(delays) - 1
                )
            ]

            print(
                "⚠️ Gemini temporarily unavailable."
            )

            print(
                f"🔄 Retrying in {delay} seconds..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "Gemini generation failed after "
        "all retry attempts."
    )


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
You are the friendly practical instructor for
Korlink Technologies Training Update.

TODAY'S TRAINING TRACK:

School:
{track['school']}

Course:
{track['name']}

Focus:
{track['description']}

Create ONE interesting practical Telegram poll.

The audience contains beginners.

The main goal is STUDENT INTERACTION.

Do NOT make it sound like an examination.

Do NOT ask boring definition questions such as:

"What is..."
"Define..."
"Which of the following defines..."

Instead, create a realistic everyday situation.

Examples:

A student receives a suspicious message.

Someone connects a laptop to Wi-Fi but has no Internet.

Someone is building a simple school website.

A classroom wants lights to turn on automatically.

A house has solar panels but the battery is not charging.

RULES:

- Use simple everyday English.
- Keep the question short.
- Make it practical.
- Make it technically correct.
- Exactly four options.
- Only one correct answer.
- Do not reveal the answer.
- Avoid unnecessary jargon.
- Vary the scenarios.
- Do not repeatedly use the same scenario.
- Do not repeat previous questions.
- Do not create a substantially similar question.
- Make beginners comfortable answering.
- Make the scenario realistic.
- Encourage students to think before answering.

PREVIOUS QUESTIONS:

{history}

Return ONLY valid JSON.

Use exactly:

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

    text = response.text.strip()

    # Remove Markdown code fences if Gemini adds them

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    try:

        poll = json.loads(
            text
        )

    except json.JSONDecodeError as error:

        print(
            "❌ Gemini returned invalid JSON:"
        )

        print(text)

        raise RuntimeError(
            f"Gemini returned invalid JSON: {error}"
        )

    # Validate required fields

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

Return ONLY one word:

DUPLICATE

or

UNIQUE

Consider the meaning and scenario,
not just identical wording.

If the question is substantially similar
to an old question, return DUPLICATE.
"""

    response = generate_with_retry(
        prompt
    )

    result = response.text.strip().upper()

    return "DUPLICATE" in result


# ============================================================
# MORNING POLL MESSAGE
# ============================================================

def format_poll(
    track,
    poll
):

    return f"""☀️ *KORLINK DAILY TECH POLL*

🎓 *{track['name']}*

👀 *Let's see who gets this!*

👉 *Question:*

{poll['question']}

1️⃣ {poll['option_1']}
2️⃣ {poll['option_2']}
3️⃣ {poll['option_3']}
4️⃣ {poll['option_4']}

👇 *Drop your answer number and why below.*

💡 _Don't be afraid to get it wrong. The goal is to learn!_
"""


# ============================================================
# FIND TODAY'S QUESTION
# ============================================================

def get_today_question():

    today = str(
        datetime.date.today()
    )

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

    return f"""🌙 *KORLINK ANSWER*

🎓 *{question['track']}*

❓ *Today's Question:*

{question['question']}

✅ *Correct Answer:*

{correct}️⃣ {correct_text}

🧠 *Why?*

{question['explanation']}

🚀 *BONUS CHALLENGE:*

{question['bonus_challenge']}

🔥 _Keep learning. Keep practicing. Keep building!_
"""


# ============================================================
# SATURDAY MOTIVATION
# ============================================================

def generate_saturday():

    prompt = """
Create a short motivational message for
Korlink Technologies Training Update.

Maximum 70 words.

Focus on:

- consistency
- learning
- practice
- building projects
- career growth

Use simple everyday English.

Do not use a famous quote.

Make it personal, warm and encouraging.

End with ONE short question that encourages
students to reply.

Use this format:

🚀 *SATURDAY MOTIVATION*

[short message]

💬 [short question]
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
Create a short Sunday inspirational message for
Korlink Technologies Training Update.

Maximum 90 words.

The message should be inspired by a Gospel principle
or short Bible reference.

Connect it to:

- learning
- wisdom
- discipline
- purpose
- using skills to help others
- preparing for a new week

Keep it warm and encouraging.

Do NOT reproduce a long Bible passage.

Do NOT preach harshly.

End with ONE simple reflection question.

Use this format:

✝️ *SUNDAY INSPIRATION*

[short message]

💭 *Reflection:*

[one short question]
"""

    response = generate_with_retry(
        prompt
    )

    return response.text.strip()


# ============================================================
# MORNING ENGINE
# ============================================================

def run_morning():

    today = datetime.date.today()

    weekday = today.weekday()

    track = WEEKDAY_TRACKS.get(
        weekday
    )

    if not track:

        raise RuntimeError(
            "No training track configured for today."
        )

    print(
        f"🎓 Generating "
        f"{track['name']} poll..."
    )

    poll = None

    for attempt in range(3):

        print(
            f"Attempt {attempt + 1}/3..."
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
            "⚠️ Similar question detected."
        )

        print(
            "🔄 Generating another question..."
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

        "correct_option": poll["correct_option"],

        "explanation": poll[
            "simple_explanation"
        ],

        "bonus_challenge": poll[
            "bonus_challenge"
        ],
    }

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
        "📱 MORNING TELEGRAM MESSAGE"
    )
    print("=" * 70)

    print(message)

    print()
    print(
        "📤 Sending to Telegram..."
    )

    if not send_telegram_message(
        message
    ):

        raise RuntimeError(
            "Telegram message could not be sent."
        )

    print()
    print(
        "✅ Morning poll completed."
    )


# ============================================================
# EVENING ENGINE
# ============================================================

def run_evening():

    question = get_today_question()

    if not question:

        raise RuntimeError(
            "No poll was found for today. "
            "Make sure the 7:50 AM job ran successfully."
        )

    message = format_answer(
        question
    )

    save_post({

        "date": str(
            datetime.date.today()
        ),

        "type": "weekday_answer",

        "track": question["track"],

        "message": message,

    })

    print()
    print("=" * 70)
    print(
        "📱 EVENING TELEGRAM MESSAGE"
    )
    print("=" * 70)

    print(message)

    print()
    print(
        "📤 Sending to Telegram..."
    )

    if not send_telegram_message(
        message
    ):

        raise RuntimeError(
            "Telegram message could not be sent."
        )

    print()
    print(
        "✅ Evening answer completed."
    )


# ============================================================
# WEEKEND ENGINE
# ============================================================

def run_weekend():

    today = datetime.date.today()

    if today.weekday() == 5:

        print(
            "🚀 Generating Saturday Motivation..."
        )

        message = generate_saturday()

        post_type = (
            "saturday_motivation"
        )

    else:

        print(
            "✝️ Generating Sunday Inspiration..."
        )

        message = generate_sunday()

        post_type = (
            "sunday_inspiration"
        )

    save_post({

        "date": str(today),

        "type": post_type,

        "message": message,

    })

    print()
    print("=" * 70)
    print(
        "📱 WEEKEND TELEGRAM MESSAGE"
    )
    print("=" * 70)

    print(message)

    print()
    print(
        "📤 Sending to Telegram..."
    )

    if not send_telegram_message(
        message
    ):

        raise RuntimeError(
            "Telegram message could not be sent."
        )

    print()
    print(
        "✅ Weekend message completed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    mode = os.getenv(
        "RUN_MODE",
        "morning"
    ).lower().strip()

    today = datetime.date.today()

    print()
    print("=" * 70)
    print(
        "🚀 KORLINK TRAINING UPDATE AI"
    )
    print("=" * 70)

    print(
        f"📅 Today: "
        f"{today.strftime('%A, %d %B %Y')}"
    )

    print(
        f"🤖 Gemini model: {TEXT_MODEL}"
    )

    print(
        f"⚙️ Run mode: {mode}"
    )

    print()

    if mode == "morning":

        if today.weekday() <= 4:

            run_morning()

        else:

            print(
                "⏭️ Morning poll is only "
                "for Monday-Friday."
            )

    elif mode == "evening":

        if today.weekday() <= 4:

            run_evening()

        else:

            print(
                "⏭️ Evening answer is only "
                "for Monday-Friday."
            )

    elif mode == "weekend":

        if today.weekday() >= 5:

            run_weekend()

        else:

            print(
                "⏭️ Weekend mode is only "
                "for Saturday/Sunday."
            )

    else:

        raise RuntimeError(
            f"Unknown RUN_MODE: {mode}"
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
