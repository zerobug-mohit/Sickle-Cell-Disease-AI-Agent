QA_SYSTEM_PROMPT = """\
You are a friendly training assistant for health workers (ASHAs, MPWs, CHOs, Staff Nurses, and Medical Officers) \
working under India's National Sickle Cell Anaemia Elimination Mission (NSCAEM).
Your users are health workers learning about Sickle Cell Disease on the job -- they may ask vague, comparative, \
or imprecise questions. Your job is to guide them to the right understanding, not to reject their questions.

How to answer:
- Use the provided context as your primary source. Do not invent medical facts, dosages, clinical criteria, or procedures.
- For vague or ambiguous questions, interpret charitably and answer the most likely intent based on the conversation so far.
- For comparison questions ("is HbSS worse than HbAS?", "what is the difference between a crisis and an infection?") \
explain how the two things relate or differ using what the context tells you. Do not refuse just because the context \
does not directly compare them.
- For follow-up questions with pronouns ("it", "that disease", "the same test", "ye", "woh") \
resolve them from the conversation history before answering.
- The question has already been judged to be about Sickle Cell Disease. If the context only partially covers it, \
give your best partial answer from the context and briefly acknowledge the gap -- do NOT refuse an SCD-related question.
- Tailor the answer to the user's role using the user profile below: focus on what THIS cadre should do at their \
facility level; when a task belongs to a different cadre or facility level, briefly say who is responsible and \
where to refer. Keep all clinical facts (dosages, criteria, procedures) accurate regardless of cadre.
- For questions about the programme structure, committees, reporting or coordination at State / District / Block \
level, use the user's state and district to make the answer concrete.
- Only decline if the topic is completely unrelated to Sickle Cell Disease, NSCAEM, or related health \
worker duties (e.g. cooking, politics, sports). In that case say clearly that this bot covers SCD training topics only.

Style:
- Keep answers short -- maximum 3 short paragraphs or a brief bullet list.
- Use simple, encouraging language. These are field health workers, not specialists.
- Write your ENTIRE reply in this language ONLY: {language_directive}
- This applies to the answer text AND every follow-up question -- all of it in that one language and script. \
Never mix languages or scripts within a single reply, and never use any language other than the one specified here.
- Write this reply entirely in that language EVEN IF earlier messages in the conversation used a different \
language or script. The language above is decided from the user's latest message and overrides the history.
- Do NOT use any emojis, emoticons, or decorative icons anywhere in your reply.
- Use single line spacing: separate paragraphs or sections with exactly one blank line, never more.

Follow-up suggestions (the "followups" list, NOT inside the answer text):
- Provide 2-3 follow-up questions ONLY when you have given a substantive answer drawn from the context.
- Write each in the SAME language and script as your answer.
- Ground every suggested question in content that is actually present in the retrieved context below.
- Do NOT suggest questions whose answers are not covered by the context.
- Use an empty list [] for introductory, meta, redirecting, or refusal replies.
- Do NOT write the suggestions or any "You could also ask" heading inside the "answer" string — only in "followups".

Output format:
- Return ONLY a JSON object with exactly these keys: {{"refused": <true|false>, "answer": "<answer text>", "followups": ["...", "..."]}}.
- "answer": your reply text with real newlines and single blank-line spacing, WITHOUT any follow-up suggestions.
- "followups": list of 2-3 short follow-up questions (or [] when not applicable, as above).
- Set "refused" to true ONLY when you are refusing because the question is unrelated to SCD/NSCAEM \
(then "answer" holds your short out-of-scope message and "followups" is []).
- For every genuine answer -- including brief, partial, or gap-acknowledging ones -- set "refused" to false.

{profile_block}
Context from training materials:
{retrieved_chunks}"""


# Maps the stored chunk `source` (the PDF filename) to the document's real,
# human-readable title as printed inside the document — not the file name.
SOURCE_TITLES = {
    "GoI_CHO SCD Training Manual (English).pdf":
        "Training Module for Community Health Officers (NSCAEM 2023)",
    "GoI_Counseling Module (English).pdf":
        "Counseling Modules on Sickle Cell Disease",
    "GoI_MO SCD Training Manual (English).pdf":
        "Training Module for Medical Officers (NSCAEM 2023)",
    "GoI_MPW and Asha SCD Training Manual (English).pdf":
        "Training Module for Multi-Purpose Workers (M/F)/ASHAs (NSCAEM 2023)",
    "GoI_NSCAEM Operational Guidelines.pdf":
        "Guidelines for National Programme for Prevention & Management of Sickle Cell Disease (NSCAEM 2023)",
    "GoI_Staff Nurse SCD Training Manual (English).pdf":
        "Training Module for Staff Nurses (NSCAEM 2023)",
    "Roles and Responsibilities of FLWs in SCD Management.pptx":
        "Roles & Responsibilities of Frontline Workers in SCD Management (NSCAEM)",
}

# (singular_label, plural_label) per language — used only when a grounded answer cites sources.
SOURCE_LABEL = {
    "en":       ("*Source:*", "*Sources:*"),
    "hi":       ("*स्रोत:*", "*स्रोत:*"),
    "hinglish": ("*Source:*", "*Sources:*"),
}

# Bold heading for the follow-up suggestions block (rendered in code, like sources,
# so it is always bold and consistently formatted). WhatsApp bold = *text*.
FOLLOWUP_LABEL = {
    "en":       "*You could also ask:*",
    "hi":       "*आप यह भी पूछ सकते हैं:*",
    "hinglish": "*Aap yeh bhi poochh sakte hain:*",
}


# --- User profile / cadres -----------------------------------------------------
# `code` matches the cadre tags written into the FAISS index by scripts/ingest.py.
CADRES = [
    {"code": "ASHA",       "name": "ASHA",                           "facility": "primary level (SHC/HWC), community", "aliases": ["asha"]},
    {"code": "ANM",        "name": "ANM",                            "facility": "primary level (SHC/HWC)",            "aliases": ["anm", "auxiliary nurse"]},
    {"code": "MPW",        "name": "Multi-Purpose Worker (MPW)",     "facility": "primary level (SHC/HWC)",            "aliases": ["mpw", "multi-purpose", "multipurpose"]},
    {"code": "CHO",        "name": "Community Health Officer (CHO)", "facility": "primary level (SHC/HWC)",            "aliases": ["cho", "community health officer"]},
    {"code": "SN",         "name": "Staff Nurse",                    "facility": "secondary level (PHC/CHC/CH)",       "aliases": ["staff nurse", "nurse"]},
    {"code": "LT",         "name": "Laboratory Technician",          "facility": "secondary/tertiary lab",             "aliases": ["laboratory technician", "lab technician", "lab tech"]},
    {"code": "MO",         "name": "Medical Officer (MO)",           "facility": "secondary level (PHC/CHC/CH)",       "aliases": ["medical officer", "doctor"]},
    {"code": "COUNSELLOR", "name": "Counsellor",                     "facility": "counselling role",                   "aliases": ["counsellor", "counselor"]},
]


def cadre_menu() -> str:
    return "\n".join(f"{i}. {c['name']}" for i, c in enumerate(CADRES, 1))


def cadre_by_code(code: str) -> dict:
    for c in CADRES:
        if c["code"] == code:
            return c
    return {"code": code, "name": code, "facility": ""}


def parse_cadre(text: str) -> dict | None:
    """Resolve a user reply to a cadre: by menu number, exact code, alias, or name."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if t.isdigit():
        i = int(t)
        return CADRES[i - 1] if 1 <= i <= len(CADRES) else None
    for c in CADRES:
        if t == c["code"].lower():
            return c
    for c in CADRES:
        if c["name"].lower() in t or any(a in t for a in c["aliases"]):
            return c
    return None


# Onboarding prompts (bilingual EN + Hinglish, since language isn't known yet).
ONBOARD_ASK_STATE = (
    "Welcome to the *SCD Health Worker Assistant* (NSCAEM).\n"
    "To give you answers relevant to your role, I need a few quick details.\n\n"
    "Which *state* do you work in?\n"
    "(Aap kis rajya mein kaam karte hain?)"
)

ONBOARD_ASK_DISTRICT = (
    "Thank you. Which *district* do you work in?\n"
    "(Aap kis zile mein kaam karte hain?)"
)

ONBOARD_ASK_CADRE = (
    "And what is your *role / cadre*? Reply with the number:\n"
    "(Aapka role kya hai? Neeche se number bhejein:)\n\n"
    + cadre_menu()
)

ONBOARD_INVALID_CADRE = (
    "Please reply with a number from 1 to {n} for your role.\n"
    "(Kripya apne role ke liye 1 se {n} ke beech ka number bhejein.)\n\n"
    + cadre_menu()
).format(n=len(CADRES))


def profile_confirm(cadre_name: str, district: str, state: str) -> str:
    return (
        "Thanks! I've saved your profile:\n"
        f"* Role: {cadre_name}\n"
        f"* District: {district}\n"
        f"* State: {state}\n\n"
        "Now ask me anything about Sickle Cell Disease and I'll answer for your role. "
        "Type /help for topics, or /profile to change these details."
    )


# Generates the tailored, single-message welcome shown after onboarding (and on /help).
WELCOME_PROMPT = """\
A health worker just registered to use a WhatsApp assistant for India's National Sickle Cell Anaemia \
Elimination Mission (NSCAEM). Their profile:
- Role / cadre: {cadre}
- Facility level: {facility}
- District: {district}
- State: {state}

Write ONE WhatsApp message (plain text) showing what they can ask. Structure it EXACTLY like this:

Line 1: a short confirmation, e.g. "You're set as {cadre} in {district}, {state}. Here's what you can ask me:"
Then these five sections, each a bold heading followed by exactly 2 short example questions:
*Disease Basics & Genetics*
*Symptoms & Complications*
*Diagnosis & Screening*
*Treatment & Management*
*Counselling & Community*
Last line, exactly: "Type /help anytime  •  /profile to change details  •  /stop to end"

Rules:
- Every example question must be specific to what a {cadre} actually does for Sickle Cell Disease, and must be \
answerable from GoI NSCAEM SCD training materials. Do not invent topics the materials would not cover.
- Bold each heading with single asterisks (WhatsApp bold). Each question on its own line starting with "* ".
- Keep the WHOLE message under 1100 characters so it fits in one WhatsApp message.
- Simple, encouraging English for a field health worker. No emojis.
Return only the message text, nothing else."""


FALLBACK = {
    "en": (
        "This question is outside the scope of the SCD training modules.\n\n"
        "I can only answer questions related to:\n"
        "* Sickle Cell Disease -- basics, genetics, and inheritance\n"
        "* Symptoms, complications, and when to refer\n"
        "* Diagnosis and community screening methods\n"
        "* Treatment and patient management\n"
        "* Counseling, NSCAEM guidelines, and community outreach\n\n"
        "Please ask a question on one of these topics."
    ),
    "hi": (
        "यह प्रश्न SCD प्रशिक्षण सामग्री के दायरे से बाहर है।\n\n"
        "मैं केवल इन विषयों पर उत्तर दे सकता हूँ:\n"
        "* सिकल सेल रोग -- मूल बातें, आनुवंशिकी और वंशानुक्रम\n"
        "* लक्षण, जटिलताएँ और कब रेफर करें\n"
        "* निदान और सामुदायिक स्क्रीनिंग\n"
        "* उपचार और रोगी प्रबंधन\n"
        "* परामर्श, NSCAEM दिशा-निर्देश और सामुदायिक आउटरीच\n\n"
        "कृपया इन्हीं विषयों से संबंधित प्रश्न पूछें।"
    ),
    "hinglish": (
        "Yeh sawaal SCD training modules ke bahar ka hai.\n\n"
        "Main sirf in topics par jawaab de sakta hoon:\n"
        "* Sickle Cell Disease -- basics aur genetics\n"
        "* Symptoms, complications aur referral\n"
        "* Diagnosis aur community screening\n"
        "* Treatment aur patient management\n"
        "* Counseling aur NSCAEM guidelines\n\n"
        "Kripya inhi topics se related sawaal poochhen."
    ),
}

GOODBYE = {
    "en":       "Goodbye! Thank you for your commitment to fighting Sickle Cell Disease.\n\nNSCAEM -- National Sickle Cell Anaemia Elimination Mission",
    "hi":       "अलविदा! सिकल सेल रोग से लड़ने की आपकी प्रतिबद्धता के लिए धन्यवाद।\n\nNSCAEM -- राष्ट्रीय सिकल सेल एनीमिया उन्मूलन मिशन",
    "hinglish": "Alvida! Sickle Cell Disease se ladne ki aapki commitment ke liye shukriya.\n\nNSCAEM -- National Sickle Cell Anaemia Elimination Mission",
}

EXIT_FOOTER = {
    "en":       "Type /help to return to the main menu anytime.",
    "hi":       "किसी भी समय मुख्य मेनू पर लौटने के लिए /help लिखें।",
    "hinglish": "Kabhi bhi main menu par jaane ke liye /help likhein.",
}

GREETING_PREFIX = {
    "en":       "Hello! Welcome to the SCD Health Worker Training Bot.\n\nHere is what you can ask about:",
    "hi":       "नमस्ते! SCD हेल्थ वर्कर ट्रेनिंग बॉट में आपका स्वागत है।\n\nआप इन विषयों पर प्रश्न पूछ सकते हैं:",
    "hinglish": "Namaste! SCD Health Worker Training Bot mein aapka swagat hai.\n\nAap in topics par sawaal poochh sakte hain:",
}

MENU = {
    "en": """\
SCD Health Worker Training Bot
National Sickle Cell Anaemia Elimination Mission (NSCAEM)

Ask any question about Sickle Cell Disease:

---
Disease Basics and Genetics
* What is sickle cell disease and how is it inherited?
* What is the difference between HbSS and HbAS?
* Can two carriers (HbAS) have a child with SCD?
* What is the difference between a carrier and a patient?
* Why does the red blood cell become sickle-shaped?

Symptoms and Complications
* What are the signs of a sickle cell crisis?
* What is acute chest syndrome and how do I recognise it?
* Why are SCD patients more prone to infections?
* What complications should I watch for in a child with SCD?
* When does a patient with SCD need emergency referral?

Diagnosis and Screening
* How is sickle cell disease diagnosed?
* What is HPLC and when is it used?
* What does a point-of-care (Sickle SCAN) test show?
* At what age should newborn screening for SCD be done?
* How is community-level screening organised under NSCAEM?

Treatment and Management
* What is hydroxyurea and who should take it?
* Why do SCD patients need folic acid daily?
* How do I manage a pain crisis at the community level?
* What vaccines are especially important for SCD patients?
* When should I refer a patient to a higher facility?

Counseling and Community
* How do I counsel a couple before marriage about SCD risk?
* How do I explain a new SCD diagnosis to a family?
* What is my role in community-level SCD screening?
* How do I track and follow up SCD patients in my area?
---

Type /help to return here anytime
Type /stop to end this session.
For support, contact: mchaurasiya@wjcf.in""",

    "hi": """\
SCD हेल्थ वर्कर ट्रेनिंग बॉट
राष्ट्रीय सिकल सेल एनीमिया उन्मूलन मिशन (NSCAEM)

सिकल सेल रोग के बारे में कोई भी प्रश्न पूछें:

---
रोग की मूल बातें और आनुवंशिकी
* सिकल सेल रोग क्या है और यह कैसे फैलता है?
* HbSS और HbAS में क्या अंतर है?
* क्या दो वाहक (HbAS) के बच्चे को SCD हो सकती है?
* वाहक और रोगी में क्या फर्क होता है?
* लाल रक्त कोशिका हँसिया के आकार की क्यों हो जाती है?

लक्षण और जटिलताएँ
* सिकल सेल क्राइसिस के क्या संकेत हैं?
* एक्यूट चेस्ट सिंड्रोम क्या है और इसे कैसे पहचानें?
* SCD मरीजों को संक्रमण अधिक क्यों होता है?
* SCD वाले बच्चे में किन जटिलताओं पर ध्यान दें?
* SCD मरीज को कब तुरंत रेफर करना चाहिए?

निदान और स्क्रीनिंग
* SCD का निदान कैसे होता है?
* HPLC क्या है और कब उपयोग होता है?
* पॉइंट-ऑफ-केयर (Sickle SCAN) टेस्ट क्या बताता है?
* नवजात शिशु की SCD स्क्रीनिंग कब होनी चाहिए?
* NSCAEM के तहत सामुदायिक स्क्रीनिंग कैसे होती है?

उपचार और प्रबंधन
* हाइड्रोक्सीयूरिया क्या है और किसे लेनी चाहिए?
* SCD मरीजों को रोज़ फोलिक एसिड क्यों चाहिए?
* समुदाय स्तर पर दर्द का संकट कैसे संभालें?
* SCD मरीजों के लिए कौन से टीके ज़रूरी हैं?
* मरीज को उच्च अस्पताल कब भेजें?

परामर्श और समुदाय
* शादी से पहले जोड़े को SCD जोखिम के बारे में कैसे समझाएँ?
* नए SCD निदान को परिवार को कैसे बताएँ?
* समुदाय स्तर पर SCD स्क्रीनिंग में मेरी क्या भूमिका है?
* अपने क्षेत्र में SCD मरीजों को कैसे ट्रैक करें?
---

यहाँ वापस आने के लिए कभी भी /help लिखें
इस सत्र को समाप्त करने के लिए /stop लिखें।
सहायता के लिए संपर्क करें: mchaurasiya@wjcf.in""",

    "hinglish": """\
SCD Health Worker Training Bot
National Sickle Cell Anaemia Elimination Mission (NSCAEM)

Sickle Cell Disease ke baare mein koi bhi sawaal poochhen:

---
Bimari ki basics aur genetics
* Sickle cell disease kya hai aur kaise milti hai?
* HbSS aur HbAS mein kya fark hai?
* Kya do carrier (HbAS) ke bacche ko SCD ho sakti hai?
* Carrier aur patient mein kya fark hota hai?
* Red blood cell haansia jaisi kyun ho jaati hai?

Symptoms aur Complications
* Sickle cell crisis ke kya signs hain?
* Acute Chest Syndrome kya hai aur kaise pehchaanen?
* SCD patients ko infection zyada kyun hota hai?
* SCD wale bacche mein kaunsi complications par dhyan den?
* SCD patient ko kab emergency mein refer karna chahiye?

Diagnosis aur Screening
* SCD ka diagnosis kaise hota hai?
* HPLC kya hai aur kab use hota hai?
* Point-of-care (Sickle SCAN) test kya batata hai?
* Naye shishu ki SCD screening kab honi chahiye?
* NSCAEM ke tahat community screening kaise hoti hai?

Ilaaj aur Management
* Hydroxyurea kya hai aur kise leni chahiye?
* SCD patients ko roz Folic Acid kyun chahiye?
* Community level par dard ka crisis kaise handle karein?
* SCD patients ke liye kaunse vaccine zaroori hain?
* Patient ko uchch hospital kab bhejna chahiye?

Counseling aur Community
* Shaadi se pehle couple ko SCD risk kaise samjayen?
* Naye SCD diagnosis ko family ko kaise batayein?
* Community SCD screening mein meri kya bhumika hai?
* Apne area mein SCD patients ko kaise track karein?
---

Kabhi bhi wapas aane ke liye /help likhein
Session khatam karne ke liye /stop likhein.
Support ke liye: mchaurasiya@wjcf.in""",
}
