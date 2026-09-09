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

    # Retain enough history for duplicate checking.
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
    for temporary API errors.
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

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    text = text.strip()

    # Remove Markdown code fences.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    # Handle accidental surrounding text by extracting
    # the outermost JSON object.
    if not text.startswith("{"):

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:

            text = text[
                start:end + 1
            ]

    return text.strip()


# ============================================================
# VALIDATE GENERATED POLL
# ============================================================

def validate_poll(poll):

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

    if not isinstance(
        poll,
        dict
    ):

        raise RuntimeError(
            "Gemini response is not a JSON object."
        )

    for field in required_fields:

        if field not in poll:

            raise RuntimeError(
                f"Gemini response is missing: {field}"
            )

        if poll[field] is None:

            raise RuntimeError(
                f"Gemini returned an empty field: {field}"
            )

    # Convert numeric strings such as "3" to integers.
    try:

        poll["correct_option"] = int(
            poll["correct_option"]
        )

    except (
        TypeError,
        ValueError
    ):

        raise RuntimeError(
            "correct_option must be a number from 1 to 4."
        )

    if poll["correct_option"] not in [
        1,
        2,
        3,
        4
    ]:

        raise RuntimeError(
            "correct_option must be between 1 and 4."
        )

    # Ensure all text fields are strings.
    text_fields = [
        "question",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "simple_explanation",
        "bonus_challenge"
    ]

    for field in text_fields:

        if not isinstance(
            poll[field],
            str
        ):

            poll[field] = str(
                poll[field]
            )

        poll[field] = poll[field].strip()

        if not poll[field]:

            raise RuntimeError(
                f"Generated field is empty: {field}"
            )

    return poll


# ============================================================
# GENERATE WEEKDAY CHALLENGE
# ============================================================

def generate_poll(track):

    questions = load_questions()

    recent_questions = questions[-50:]

    history = "\n".join(
        f"- {item.get('question', '')}"
        for item in recent_questions
    )

    prompt = f"""
You are the official content writer and practical instructor
for Korlink Technologies.

Korlink Technologies runs a professional technology training
community called "Korlink Daily Challenge".

Your task is to create today's challenge for the community.

TODAY'S TRAINING TRACK

School:
{track['school']}

Course:
{track['name']}

Training focus:
{track['description']}

PURPOSE

The challenge should encourage people to stop, think and
participate.

It must feel like something a knowledgeable human instructor
would naturally post in a professional training community.

The audience can include:
- complete beginners
- students
- working professionals
- people changing careers
- technically experienced learners

Therefore, the challenge must be understandable even to someone
who is not yet familiar with the technical subject.

CONTENT STYLE

Write in natural, clear English.

Make the challenge engaging without sounding childish.

Use a realistic situation from everyday life, work, business,
school, home or technology use.

The scenario should provide enough context to make the question
interesting, but it must not become a long story.

The ideal question can normally be read in about 15 to 25 seconds.

Do NOT make the question extremely short.

Do NOT make the question excessively long.

Aim for approximately 25 to 55 words for the question.

OPTIONS

Create exactly four options.

Each option should normally be short enough to read quickly.

All four options should be plausible.

Only one option must be correct.

The correct answer must be technically accurate.

Avoid trick questions.

LEARNING VALUE

The challenge should test practical understanding rather than
memorisation.

Whenever possible, make the learner think about what they would
actually do in a real situation.

Avoid repeatedly asking simple definitions such as:
"What is..."
"Define..."
"Which of these is..."

A definition-based question is acceptable only when it is
genuinely useful and presented in a practical context.

EXAMPLES OF THE RIGHT STYLE

Cybersecurity:

A staff member receives an email claiming that their company
account will be suspended unless they confirm their password
through a link. What should they do before taking any action?

Software Engineering:

A developer adds a new feature to an application, but an older
feature suddenly stops working. What should the developer check
first?

Smart Home Automation:

A homeowner wants the corridor light to turn on automatically
when someone enters at night. Which device would best detect
the person's movement?

Network Engineering:

A laptop connects successfully to the office Wi-Fi, but websites
will not open while other devices are working normally. What is
the most useful first check?

Solar PV:

A solar system receives good sunlight during the day, but the
battery is not charging as expected. Which part of the system
should be checked first?

These examples show the desired level of detail. Do not copy
them or create questions substantially similar to them.

PROFESSIONAL STYLE

Do:
- Sound like an experienced instructor.
- Be practical.
- Be clear.
- Be interesting.
- Use natural language.
- Make people curious enough to answer.
- Make the content useful.

Do not:
- Mention AI.
- Mention Gemini.
- Mention prompts.
- Use hashtags.
- Use excessive emojis.
- Use hype.
- Use slang.
- Use childish expressions.
- Use exaggerated motivational phrases.
- Use "Let's see who gets this!"
- Use "Are you ready?"
- Use "Tech warriors!"
- Use "Test your brain!"
- Use "Level up!"
- Use "Crush this!"
- Turn the question into an advertisement.

EXPLANATION

The explanation should be approximately 25 to 60 words.

It should clearly explain why the correct answer is correct.

Write it so that someone who selected the wrong answer can
still learn something useful.

BONUS CHALLENGE

The bonus challenge should be one short practical question or
task related to the same topic.

It should encourage further thinking without becoming another
long lesson.

QUESTION HISTORY

Do not repeat or substantially recreate any of these previous
questions:

{history}

IMPORTANT JSON REQUIREMENTS

Your entire response MUST be valid JSON.

Do not write anything before the JSON.

Do not write anything after the JSON.

Do not use Markdown code fences.

The JSON must contain ALL of these fields:

{{
    "question": "string",
    "option_1": "string",
    "option_2": "string",
    "option_3": "string",
    "option_4": "string",
    "correct_option": 1,
    "simple_explanation": "string",
    "bonus_challenge": "string"
}}

"correct_option" MUST be a number, not a word and not a string.

It MUST be exactly one of:

1
2
3
4

Before returning the response, verify that:
1. There are exactly four options.
2. There is exactly one correct answer.
3. correct_option matches the correct option.
4. The question is realistic.
5. The question is not too long.
6. The explanation is useful but concise.
7. The bonus challenge is short.
8. The question is not substantially similar to the previous questions.
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

    return validate_poll(
        poll
    )


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
Compare this new training challenge with the previous challenges.

NEW CHALLENGE:

{new_question}

PREVIOUS CHALLENGES:

{previous}

Determine whether the new challenge is substantially similar
to any previous challenge.

Consider:
- the scenario
- the learning objective
- the practical situation
- the reasoning required
- the subject being tested

Do not mark it as duplicate simply because it belongs to the
same course.

Return ONLY one word:

DUPLICATE

or

UNIQUE
"""

    response = generate_with_retry(
        prompt
    )

    result = response.text.strip().upper()

    return result.startswith(
        "DUPLICATE"
    )


# ============================================================
# MORNING CHALLENGE MESSAGE
# ============================================================

def format_poll(
    track,
    poll
):

    return f"""*KORLINK TECHNOLOGIES*

*DAILY CHALLENGE*

*Track:* {track['name']}

{poll['question']}

1. {poll['option_1']}
2. {poll['option_2']}
3. {poll['option_3']}
4. {poll['option_4']}

Share your answer and, if possible, tell us why you chose it.
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

*DAILY CHALLENGE*

*Answer & Explanation*
*Track:* {question['track']}

*Today's Challenge*

{question['question']}

*Correct Answer:*

{correct}. {correct_text}

*Explanation:*

{question['explanation']}

*Practical Challenge:*

{question['bonus_challenge']}
"""


# ============================================================
# SATURDAY BOOST
# ============================================================

def generate_saturday():

    prompt = """
Write the official Saturday message for Korlink Technologies'
training community.

Programme:
KORLINK DAILY CHALLENGE

Heading:
Saturday Boost

Create a warm, professional message for students and aspiring
technology professionals.

Maximum 80 words.

Focus naturally on:
- consistency
- practice
- learning
- building projects
- professional development

The message should feel like it was written by an experienced
training organisation, not an AI.

Avoid:
- clichés
- excessive motivation
- exaggerated promises
- hashtags
- slang
- childish language
- unnecessary emojis
- famous quotes
- references to AI

End with one simple question that encourages students to reply.

Return only the final message.
"""

    response = generate_with_retry(
        prompt
    )

    return response.text.strip()


# ============================================================
# SUNDAY REFLECTION
# ============================================================

def generate_sunday():

    prompt = """
Write the official Sunday message for Korlink Technologies'
training community.

Programme:
KORLINK DAILY CHALLENGE

Heading:
Sunday Reflection

Create a short, respectful Sunday reflection.

Maximum 100 words.

The message may be inspired by a Gospel principle or a short
Bible reference.

Connect the reflection naturally with:
- wisdom
- discipline
- learning
- purpose
- using skills responsibly
- preparing for the coming week

The tone should be warm, professional and respectful.

Do not preach harshly.

Do not reproduce a long Bible passage.

Avoid excessive religious language, hashtags, slang,
childish wording and unnecessary emojis.

Do not mention AI.

End with one simple reflection question.

Return only the final message.
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
        f"Generating {track['name']} challenge..."
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
            "Generating another challenge..."
        )

    if poll is None:

        raise RuntimeError(
            "Could not generate a unique "
            "challenge after three attempts."
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
        "Morning challenge completed successfully."
    )


# ============================================================
# EVENING ENGINE
# ============================================================

def run_evening():

    question = get_today_question()

    if not question:

        raise RuntimeError(
            "No challenge was found for today. "
            "The morning challenge may not have completed successfully."
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
            "Generating Saturday Boost..."
        )

        message = generate_saturday()

        post_type = (
            "saturday_boost"
        )

    elif today.weekday() == 6:

        print(
            "Generating Sunday Reflection..."
        )

        message = generate_sunday()

        post_type = (
            "sunday_reflection"
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
    # MORNING
    # --------------------------------------------------------

    if mode == "morning":

        if today.weekday() <= 4:

            run_morning()

        else:

            print(
                "Morning weekday challenge is not scheduled "
                "for Saturday or Sunday."
            )

    # --------------------------------------------------------
    # EVENING
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
