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
- Only decline if the topic is completely unrelated to Sickle Cell Disease, NSCAEM, or related health \
worker duties (e.g. cooking, politics, sports). In that case say clearly that this bot covers SCD training topics only.

Style:
- Keep answers short -- maximum 3 short paragraphs or a brief bullet list.
- Use simple, encouraging language. These are field health workers, not specialists.
- Detect the language of the user's question and reply in the SAME language and script:
    * Hindi (Devanagari script) -> reply in Hindi Devanagari
    * Hinglish (Roman script Hindi) -> reply in Hinglish Roman script
    * English -> reply in English
- Do NOT mix scripts in a single reply.
- Do NOT use any emojis, emoticons, or decorative icons anywhere in your reply.
- Use single line spacing: separate paragraphs or sections with exactly one blank line, never more.

Follow-up suggestions:
- Only include these when you have given a substantive answer drawn from the training material context.
- Do NOT include them if your response is introductory, meta, or redirecting.
- After a substantive answer, add exactly one blank line, then suggest 2-3 follow-up questions the user could ask next.
- Ground every suggested question in content that is actually present in the retrieved context below.
- Do NOT suggest questions whose answers are not covered by the context.
- Format exactly like this (match the user's language/script), with no icon before the heading:
  English  -> *You could also ask:*\n* Q1\n* Q2\n* Q3
  Hindi    -> *aap yeh bhi poochh sakte hain:*\n* Q1\n* Q2\n* Q3
  Hinglish -> *Aap yeh bhi poochh sakte hain:*\n* Q1\n* Q2\n* Q3

Output format:
- Return ONLY a JSON object with exactly two keys: {{"refused": <true|false>, "answer": "<your full reply>"}}.
- Put your ENTIRE reply -- the answer text and any follow-up suggestions -- inside the "answer" string, with real newlines.
- Set "refused" to true ONLY when you are refusing to answer because the question is unrelated to SCD/NSCAEM \
(the "answer" then holds your short out-of-scope message).
- For every genuine answer -- including brief, partial, or gap-acknowledging ones -- set "refused" to false.

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
}

# (singular_label, plural_label) per language — used only when a grounded answer cites sources.
SOURCE_LABEL = {
    "en":       ("*Source:*", "*Sources:*"),
    "hi":       ("*स्रोत:*", "*स्रोत:*"),
    "hinglish": ("*Source:*", "*Sources:*"),
}


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
        "yeh prashn SCD prasikshan samagri ke dayre se bahar hai.\n\n"
        "main keval in vishyon par uttar de sakta hoon:\n"
        "* Sickle Cell Disease -- mool baaten aur aanuvamshikta\n"
        "* Lakshan, jatiltaen aur referral\n"
        "* Nidan aur samudayik screening\n"
        "* Upchar aur rogi prabandhan\n"
        "* Paramarsh aur NSCAEM disha-nirdesh\n\n"
        "Kripya inhi vishyon se sambandhit prashn poochhen."
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
    "hi":       "Alvida! Sickle Cell Disease se ladne ki aapki pratibaddhat ke liye dhanyavaad.\n\nNSCAEM -- Rashtriya Sickle Cell Anaemia Unmulan Mission",
    "hinglish": "Alvida! Sickle Cell Disease se ladne ki aapki commitment ke liye shukriya.\n\nNSCAEM -- National Sickle Cell Anaemia Elimination Mission",
}

EXIT_FOOTER = {
    "en":       "Type /help to return to the main menu anytime.",
    "hi":       "Kabhi bhi mukhya menu par wapas jaane ke liye /help likhein.",
    "hinglish": "Kabhi bhi main menu par jaane ke liye /help likhein.",
}

GREETING_PREFIX = {
    "en":       "Hello! Welcome to the SCD Health Worker Training Bot.\n\nHere is what you can ask about:",
    "hi":       "Namaste! SCD Health Worker Training Bot mein aapka swagat hai.\n\nAap in vishyon par prashn poochh sakte hain:",
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
SCD Health Worker Training Bot
Rashtriya Sickle Cell Anaemia Unmulan Mission (NSCAEM)

Sickle Cell Disease ke baare mein koi bhi prashn poochhen:

---
Rog ki mool baaten aur aanuvamshikta
* Sickle cell disease kya hai aur yeh kaise phailta hai?
* HbSS aur HbAS mein kya antar hai?
* Kya do vahak (HbAS) ke bacche ko SCD ho sakti hai?
* Vahak aur marij mein kya fark hota hai?
* Lal rakt koshika hansia ke aakaar ki kyun ho jaati hai?

Lakshan aur Jatiltaen
* Sickle cell crisis ke kya sanket hain?
* Acute Chest Syndrome kya hai aur ise kaise pehchaanen?
* SCD mareezon ko sankraman adhik kyun hota hai?
* SCD wale bacche mein kaun si jatiltaon par dhyan den?
* SCD mareej ko kab turant refer karna chahiye?

Nidan aur Screening
* SCD ka nidan kaise hota hai?
* HPLC kya hai aur kab upyog hota hai?
* Point-of-care (Sickle SCAN) test kya batata hai?
* Navjat shishu ki SCD screening kab honi chahiye?
* NSCAEM ke tahat samudayik screening kaise hoti hai?

Upchar aur Prabandhan
* Hydroxyurea kya hai aur kise leni chahiye?
* SCD mareezon ko roz Folic Acid kyun chahiye?
* Samudaay star par dard ka sankat kaise sambhaalen?
* SCD mareezon ke liye kaun se tike zaroori hain?
* Mareej ko uchch aspatal kab bhejen?

Paramarsh aur Samudaay
* Shaadi se pehle jode ko SCD jokhim ke baare mein kaise samjayen?
* Naye SCD nidan ko parivar ko kaise batayen?
* Samudaay star par SCD screening mein meri kya bhumika hai?
* Apne kshetr mein SCD mareezon ko kaise track karen?
---

Wapas aane ke liye kabhi bhi /help likhein
Session khatam karne ke liye /stop likhein.
Sahayata ke liye: mchaurasiya@wjcf.in""",

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
