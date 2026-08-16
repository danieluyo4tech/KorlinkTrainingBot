import os
import json
import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# ============================================================
# KORLINK TRAINING UPDATE AI ENGINE
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured."
    )

client = genai.Client(api_key=API_KEY)

# Gemini model
TEXT_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)


# ============================================================
# FILE LOCATIONS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

QUESTIONS_FILE = DATA_DIR / "questions.json"
POSTS_FILE = DATA_DIR / "posts.json"
STATUS_FILE = DATA_DIR / "status.json"


# ============================================================
# TEST SETTINGS
# ============================================================

# While testing locally:
#
# TEST_MODE = True
# TEST_DATE = "2026-08-17"
#
# When deployed to GitHub:
#
# TEST_MODE = False

TEST_MODE = False

TEST_DATE = "2026-08-17"


def get_today():

    if TEST_MODE:
        return datetime.date.fromisoformat(TEST_DATE)

    return datetime.date.today()


# ============================================================
# WEEKLY TRAINING SCHEDULE
# ============================================================

WEEKDAY_TRACKS = {

    # MONDAY
    0: {
        "school": "School of Computing",
        "name": "Cybersecurity",
        "description": (
            "online safety, phishing, passwords, privacy, "
            "scams, malware and practical cybersecurity"
        ),
    },

    # TUESDAY
    1: {
        "school": "School of Computing",
        "name": "Software Engineering",
        "description": (
            "websites, applications, coding, databases, "
            "debugging, programming and software development"
        ),
    },

    # WEDNESDAY
    2: {
        "school": "School of Technology",
        "name": "Network Engineering",
        "description": (
            "Wi-Fi, routers, switches, IP addresses, "
            "Internet connections and network troubleshooting"
        ),
    },

    # THURSDAY
    3: {
        "school": "School of Technology",
        "name": "Smart Tech and Automation",
        "description": (
            "IoT, sensors, smart devices, automation, "
            "controllers, robotics and practical automation"
        ),
    },

    # FRIDAY
    4: {
        "school": "School of Technology",
        "name": "Solar PV Design and Installation",
        "description": (
            "solar panels, batteries, inverters, "
            "charge controllers, solar system design "
            "and practical installation"
        ),
    },
}


# ============================================================
# JSON FUNCTIONS
# ============================================================

def load_json(file_path, default):

    if not file_path.exists():
        return default

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return default


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
# QUESTION HISTORY
# ============================================================

def load_questions():

    return load_json(
        QUESTIONS_FILE,
        []
    )


def save_question(question):

    questions = load_questions()

    questions.append(question)

    save_json(
        QUESTIONS_FILE,
        questions
    )


# ============================================================
# POST HISTORY
# ============================================================

def save_post(post):

    posts = load_json(
        POSTS_FILE,
        []
    )

    posts.append(post)

    save_json(
        POSTS_FILE,
        posts
    )


# ============================================================
# STATUS
# ============================================================

def load_status():

    return load_json(
        STATUS_FILE,
        {}
    )


def save_status(status):

    save_json(
        STATUS_FILE,
        status
    )


# ============================================================
# CLEAN GEMINI JSON
# ============================================================

def clean_json(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# ============================================================
# GENERATE QUESTION
# ============================================================

def generate_poll(track):

    questions = load_questions()

    recent_questions = questions[-50:]

    if recent_questions:

        history = "\n".join(
            f"- {q.get('question', '')}"
            for q in recent_questions
        )

    else:

        history = "No previous questions."


    prompt = f"""
You are the friendly instructor for
Korlink Technologies Training Update.

Create ONE beginner-friendly interactive
technology poll.

COURSE:

{track['school']} → {track['name']}

TOPIC AREA:

{track['description']}

The audience contains beginners.

The question should feel like a friendly
real-life situation, not an examination.

For example:

A student receives a suspicious message.

Someone connects to Wi-Fi but has no Internet.

Someone is building a simple website.

Someone wants a classroom light to turn on
automatically.

Someone is designing a small solar system.

Do NOT make the question simply ask for a
definition.

RULES:

1. Use simple everyday English.
2. Keep the question short.
3. Make it practical.
4. Make it educational.
5. Give exactly four options.
6. Only one option is correct.
7. Make all four options believable.
8. Do not reveal the answer.
9. Avoid unnecessary technical jargon.
10. Vary the real-life scenarios.
11. Do not repeat previous questions.
12. Do not repeat the same underlying learning point.
13. Make beginners comfortable participating.
14. Make it suitable for a WhatsApp group.
15. Do not make it sound like an examination.

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
    "practical_tip": "...",
    "bonus_challenge": "..."
}}
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt
    )

    text = clean_json(
        response.text
    )

    try:

        poll = json.loads(text)

    except json.JSONDecodeError:

        raise RuntimeError(
            "Gemini returned invalid JSON:\n\n"
            + response.text
        )

    required = [
        "question",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "correct_option",
        "simple_explanation",
        "practical_tip",
        "bonus_challenge"
    ]

    for field in required:

        if field not in poll:

            raise RuntimeError(
                f"Missing field: {field}"
            )

    return poll


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate(question):

    questions = load_questions()

    if not questions:
        return False

    previous = questions[-50:]

    previous_text = "\n".join(
        f"{i + 1}. {q.get('question', '')}"
        for i, q in enumerate(previous)
    )

    prompt = f"""
Check whether this new question is substantially
similar to any previous question.

NEW QUESTION:

{question}

PREVIOUS QUESTIONS:

{previous_text}

If it is essentially the same scenario or learning
point, respond:

DUPLICATE

Otherwise respond:

UNIQUE

Return only one word.
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt
    )

    result = response.text.strip().upper()

    return result.startswith("DUPLICATE")


# ============================================================
# UNIQUE QUESTION GENERATOR
# ============================================================

def generate_unique_poll(track):

    for attempt in range(1, 6):

        print(
            f"🧠 Generating question "
            f"{attempt}/5..."
        )

        poll = generate_poll(track)

        print(
            f"👉 {poll['question']}"
        )

        if not is_duplicate(
            poll["question"]
        ):

            print(
                "✅ Question is unique."
            )

            return poll

        print(
            "⚠️ Similar question detected."
        )

    raise RuntimeError(
        "Unable to generate a sufficiently "
        "different question."
    )


# ============================================================
# MORNING MESSAGE
# ============================================================

def format_morning_poll(
    track,
    poll
):

    return f"""☀️ *DAILY TECH POLL*

🎓 *{poll['question']}*

1️⃣ {poll['option_1']}
2️⃣ {poll['option_2']}
3️⃣ {poll['option_3']}
4️⃣ {poll['option_4']}

👇 *Drop your answer number below.*

💡 _Everyone can participate — beginners included!_
"""


# ============================================================
# EVENING ANSWER
# ============================================================

def format_evening_answer(
    poll
):

    correct_number = poll[
        "correct_option"
    ]

    correct_text = poll[
        f"option_{correct_number}"
    ]

    return f"""🎯 *ANSWER TIME!*

The correct answer is:

*{correct_number}️⃣ {correct_text}*

📚 *Why?*

{poll['simple_explanation']}

💡 *Practical Tip:*

{poll['practical_tip']}

🔥 *Bonus Challenge:*

{poll['bonus_challenge']}

👏 Thanks to everyone who participated today!
"""


# ============================================================
# SATURDAY MOTIVATION
# ============================================================

def generate_saturday():

    prompt = """
Create a short motivational message for
a technology training WhatsApp group.

Maximum 60 words.

Focus on consistency, learning, practice,
building projects and career growth.

Use simple everyday English.

Do not use a famous person's quotation.

End with one short question encouraging
students to reply.

Format:

🚀 *SATURDAY MOTIVATION*

[message]

💬 [question]
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# SUNDAY INSPIRATION
# ============================================================

def generate_sunday():

    prompt = """
Create a short Sunday inspirational message
for a technology training WhatsApp group.

Maximum 70 words.

Use a Gospel principle or short Bible reference.

Connect it to learning, wisdom, discipline,
purpose, using skills to help others and
preparing for a new week.

Keep it warm and encouraging.

Do not reproduce a long Bible passage.

End with one simple reflection question.

Format:

✝️ *SUNDAY INSPIRATION*

[message]

💭 *Reflection:* [question]
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=prompt
    )

    return response.text.strip()


# ============================================================
# MORNING JOB
# ============================================================

def morning_job(today):

    if today.weekday() > 4:
        return None

    date_key = str(today)

    status = load_status()

    if status.get(date_key, {}).get(
        "morning_generated"
    ):

        print(
            "ℹ️ Morning poll already generated."
        )

        return None

    track = WEEKDAY_TRACKS[
        today.weekday()
    ]

    print()
    print("=" * 70)
    print("☀️ MORNING POLL")
    print("=" * 70)

    print(
        f"🎓 {track['school']} → "
        f"{track['name']}"
    )

    poll = generate_unique_poll(
        track
    )

    message = format_morning_poll(
        track,
        poll
    )

    question_record = {

        "date": date_key,

        "school": track["school"],

        "track": track["name"],

        "question": poll["question"],

        "option_1": poll["option_1"],

        "option_2": poll["option_2"],

        "option_3": poll["option_3"],

        "option_4": poll["option_4"],

        "correct_option":
            poll["correct_option"],

        "simple_explanation":
            poll["simple_explanation"],

        "practical_tip":
            poll["practical_tip"],

        "bonus_challenge":
            poll["bonus_challenge"],

        "morning_message":
            message
    }

    save_question(
        question_record
    )

    save_post({

        "date": date_key,

        "type": "morning_poll",

        "message": message
    })

    status[date_key] = {

        "morning_generated": True,

        "evening_generated": False
    }

    save_status(status)

    print()
    print("=" * 70)
    print("📱 MORNING MESSAGE")
    print("=" * 70)

    print(message)

    print()
    print(
        "✅ Morning poll generated."
    )

    return message


# ============================================================
# EVENING JOB
# ============================================================

def evening_job(today):

    if today.weekday() > 4:
        return None

    date_key = str(today)

    status = load_status()

    if status.get(date_key, {}).get(
        "evening_generated"
    ):

        print(
            "ℹ️ Evening answer already generated."
        )

        return None

    questions = load_questions()

    today_questions = [

        q for q in questions

        if q.get("date") == date_key
    ]

    if not today_questions:

        print(
            "⚠️ No poll was found for today."
        )

        return None

    poll = today_questions[-1]

    message = format_evening_answer(
        poll
    )

    save_post({

        "date": date_key,

        "type": "evening_answer",

        "message": message
    })

    status.setdefault(
        date_key,
        {}
    )

    status[date_key][
        "evening_generated"
    ] = True

    save_status(status)

    print()
    print("=" * 70)
    print("🎯 EVENING ANSWER")
    print("=" * 70)

    print(message)

    print()
    print(
        "✅ Evening answer generated."
    )

    return message


# ============================================================
# WEEKEND
# ============================================================

def weekend_job(today):

    date_key = str(today)

    status = load_status()

    if status.get(date_key, {}).get(
        "weekend_generated"
    ):

        print(
            "ℹ️ Weekend content already generated."
        )

        return None

    if today.weekday() == 5:

        message = generate_saturday()

        post_type = (
            "saturday_motivation"
        )

    else:

        message = generate_sunday()

        post_type = (
            "sunday_inspiration"
        )

    save_post({

        "date": date_key,

        "type": post_type,

        "message": message
    })

    status[date_key] = {

        "weekend_generated": True
    }

    save_status(status)

    print()
    print("=" * 70)
    print("📱 WEEKEND MESSAGE")
    print("=" * 70)

    print(message)

    return message


# ============================================================
# MAIN
# ============================================================

def main():

    today = get_today()

    print()
    print(
        "🚀 KORLINK TRAINING UPDATE AI ENGINE"
    )

    print(
        f"📅 {today.strftime('%A, %d %B %Y')}"
    )

    if today.weekday() <= 4:

        morning_job(today)

        evening_job(today)

    else:

        weekend_job(today)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
