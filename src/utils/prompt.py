def get_refine_arabic_prompt_deepseek(raw_text):
    return f"""
Act as a senior medical transcription editor specializing in Arabic healthcare documentation.

**ORIGINAL TRANSCRIPTION:**
{raw_text}

**EDITING TASKS:**
- Correct grammatical errors and awkward phrasing
- Improve sentence structure and flow
- Maintain all medical facts and clinical details
- De-identify speaker references when possible
- Ensure professional medical Arabic standards
- Enhance readability for medical professionals

**CRITICAL RULES:**
→ Output ONLY the corrected Arabic text
→ No additional commentary or explanations
→ Preserve the original medical meaning completely
→ Use formal Arabic appropriate for medical records
→ Ensure refining any mentioned medications

“⚠️ Absolutely forbidden to use any asterisks (*), markdown symbols, or additional text.
The output must contain only plain text.
If asterisks or additional text appear anywhere in your response, it is considered incorrect output.”

**CORRECTED MEDICAL TEXT:**
"""


def get_refine_english_prompt_deepseek(translated_text):
    return f"""
    **EDITING TASKS:**
    - Correct grammatical errors and awkward phrasing
    - Improve sentence structure and flow
    - Maintain all medical facts and clinical details
    - De-identify speaker references when possible
    - Ensure professional medical English standards
    - Enhance readability for medical professionals

    **CRITICAL RULES:**
    → Output ONLY the corrected English text
    → No additional commentary or explanations
    → Preserve the original medical meaning completely
    → Use formal English appropriate for medical records
    → Ensure refining any mentioned medications
    
    “⚠️ Absolutely forbidden to use any asterisks (*), markdown symbols, or additional text.
    The output must contain only plain text.
    If asterisks or additional text appear anywhere in your response, it is considered incorrect output.”

    ORIGINAL TEXT:
    \"\"\"{translated_text}\"\"\"
    **ONLY return the refinment no more**
    """


# --- Conversation Mode Prompts ---
def get_refine_arabic_prompt_deepseek_conversation(raw_text):
    return f"""
    Act as a clinical transcription editor. Refine this Arabic medical conversation for accuracy and clarity.

    **CONVERSATION TO PROCESS:**
    {raw_text}

    **SPEAKER IDENTIFICATION:**
    - **الدكتور:** Medical professional (asks questions, examines, diagnoses, prescribes)
    - **المريض:** Patient (describes symptoms, answers questions, shares history)

    **QUALITY STANDARDS:**
    1. **Accuracy:** Preserve all medical content, terminology, and clinical context
    2. **Clarity:** Fix grammatical errors and improve sentence flow
    3. **Format:** Clearly label each speaker turn with **الدكتور:** or **المريض:**
    4. **Anonymization:** Minimize personal identifiers while keeping dialogue intact
    5. **Professionalism:** Use formal medical Arabic appropriate for clinical documentation

    **CRITICAL:**
    → Output ONLY the refined conversation in Arabic
    → No additional text, explanations, or commentary
    → Maintain original dialogue sequence and medical meaning
    → Each speaker turn must be properly labeled
    → Keep brand and generic drug names unchanged, do not lose it, you must mention it
    → Ensure refining any mentioned medications
    
    “⚠️ Absolutely forbidden to use any asterisks (*), markdown symbols, or additional text.
    The output must contain only plain text.
    If asterisks or additional text appear anywhere in your response, it is considered incorrect output.”

    **REFINED CLINICAL CONVERSATION:**
"""


def get_refine_english_prompt_deepseek_conversation(translated_text):
    return f"""
Refine this English medical conversation while preserving the dialogue structure.

Instructions:
1. Correct grammar, punctuation, and phrasing for natural spoken English.
2. Ensure medical terminology and tone are appropriate for a clinical dialogue.
3. Keep brand and generic drug names unchanged, do not lose it, you must mention it. 
4. Ensure refining any mentioned medications
5. Maintain the natural conversational flow.
6. Use **Doctor:** and **Patient:** exactly as shown below (capitalize only the first letter).
7. Return ONLY the refined conversation in dialogue format — no introductions, explanations, or extra text.

Example format:
Doctor: Good morning. What brings you in today?
Patient: I’ve had chest pain for a few days.

Doctor: What kind of pain?
Patient: A pressure in the center of my chest, worse when I climb stairs.

ORIGINAL TEXT:
\"\"\"{translated_text}\"\"\"

“⚠️ Absolutely forbidden to use any asterisks (*), markdown symbols, or additional text.
The output must contain only plain text.
If asterisks appear anywhere in your response or additional comments appear at the begining, it is considered incorrect output.”

Only return the dialogue no more **Without any additional text or astrisks**
"""


def get_translation_prompt_deepseek(refined_text):
    return f"""
**TASK:** Translate Arabic medical text to English

**SOURCE TEXT (Arabic):**
{refined_text}

**TRANSLATION REQUIREMENTS:**
- Translate ALL Arabic text to English
- Preserve medical terminology accurately
- Maintain professional medical language
- Keep the structure and meaning identical
- Keep brand and generic drug names unchanged, do not lose it, you must mention it

**CRITICAL RULES:**
→ Output MUST be in ENGLISH only
→ No Arabic characters or words in the output
→ Return ONLY the English translation
→ No additional commentary or explanations
→ If the text is already English, return it as-is

“⚠️ Absolutely forbidden to use any asterisks (*), markdown symbols, or additional text.
The output must contain only plain text.
If asterisks or additional text appear anywhere in your response, it is considered incorrect output.”

**ENGLISH TRANSLATION:**
"""


def get_translation_prompt_deepseek_conversation(refined_text):
    return f"""
**TASK:** Translate Arabic medical conversation to English

**SOURCE CONVERSATION (Arabic):**
{refined_text}

**TRANSLATION REQUIREMENTS:**
- Translate ALL Arabic dialogue to English
- Preserve speaker labels exactly: **الدكتور:** → **Doctor:** and **المريض:** → **Patient:**
- Maintain accurate medical terminology
- Keep conversational flow and structure
- Keep brand and generic drug names unchanged, do not lose it, you must mention it

**SPEAKER LABEL MAPPING:**
- **الدكتور:** must become **Doctor:**
- **المريض:** must become **Patient:**

**CRITICAL RULES:**
→ Output MUST be in ENGLISH only
→ No Arabic characters in the final output
→ Preserve the dialogue format with English speaker labels
→ Return ONLY the translated conversation
→ No additional text or explanations

“⚠️ Absolutely forbidden to use any asterisks (*), markdown symbols, or additional text.
The output must contain only plain text.
If asterisks or additional text appear anywhere in your response, it is considered incorrect output.”

**ENGLISH TRANVERSATION:**
"""


def get_dynamic_extraction_prompt_llama(translated_text, features):
    return f"""
You are a medical expert Given the following medical text, extract relevant medical features and provide reasoning for the extraction. Return a JSON object with two fields:
- "json_data": A dictionary containing this medical features:
  {features}
- "reasoning": A string explaining the rationale behind the extracted features.

Leave fields empty ("" for strings, [] for lists) if no relevant information is found in the text.

Text: {translated_text}

Example output:
{{
  "json_data": {{
    "chief_complaint": "Persistent cough and fever",
    "icd10_codes": [
      "J11.1 - Influenza with respiratory manifestations",
      "R05 - Cough"
    ],
    "history_of_illness": "Patient has a history of asthma and seasonal allergies.",
    "current_medication": "Albuterol inhaler, Oseltamivir 75mg twice daily",
    "imaging_results": "Chest X-ray shows no consolidation.",
    "plan": "Continue Oseltamivir for 5 days, use Albuterol as needed.",
    "assessment": "Influenza with acute respiratory symptoms",
    "follow_up": "Return in 7 days or sooner if symptoms worsen."
  }},
  "reasoning": "The text describes a patient with cough and fever, leading to a diagnosis of influenza. ICD-10 codes J11.1 and R05 are assigned based on the symptoms. The history of asthma and allergies is noted. Current medications include Oseltamivir for influenza and Albuterol for asthma. Chest X-ray is normal, supporting a viral etiology. The plan includes antiviral treatment and symptom management, with a follow-up in 7 days."
}}
"""

# --- Conversation Mode Dynamic Extraction ---
def get_dynamic_extraction_prompt_llama_conversation(translated_text, features):
    return f"""
You are a medical expert. Given the following medical conversation between a doctor and patient, extract relevant medical features and provide reasoning for the extraction. Return a JSON object with two fields:
- "json_data": A dictionary containing these medical features:
  {features}
- "reasoning": A string explaining the rationale behind the extracted features and how they were identified from the conversation.

Leave fields empty ("" for strings, [] for lists) if no relevant information is found in the conversation.

Instructions:
- Extract information from both doctor's statements and patient's responses
- Chief complaint should come from patient's initial description
- Medical history should be gathered from patient's answers
- Treatment plans and assessments should come from doctor's recommendations
- Add "conversation_summary" field if not in features list

Conversation: {translated_text}

Example output:
{{
  "json_data": {{
    "chief_complaint": "Persistent cough and fever for 5 days",
    "icd10_codes": [
      "J11.1 - Influenza with respiratory manifestations",
      "R05 - Cough"
    ],
    "history_of_illness": "Patient reports history of asthma and seasonal allergies.",
    "current_medication": "Albuterol inhaler, Doctor prescribed Oseltamivir 75mg twice daily",
    "imaging_results": "Chest X-ray shows no consolidation.",
    "plan": "Continue Oseltamivir for 5 days, use Albuterol as needed.",
    "assessment": "Influenza with acute respiratory symptoms",
    "follow_up": "Return in 7 days or sooner if symptoms worsen.",
    "conversation_summary": "Patient presented with 5-day history of cough and fever. Examination and chest X-ray ruled out pneumonia. Diagnosed with influenza and prescribed antiviral treatment."
  }},
  "reasoning": "Chief complaint identified from patient's opening description. Medical history extracted from patient's responses about previous conditions. Doctor's diagnosis informed ICD-10 code selection. Treatment plan based on doctor's recommendations during consultation. Follow-up instructions from doctor's closing remarks."
}}
"""


# --- Question Generation Prompts ---
def get_question_generation_prompt_llama(translated_text):
    return f"""
    You are a **clinical reasoning expert** assisting a doctor in ensuring a thorough patient evaluation.

    Based on the following **medical dictation**, analyze the patient's condition and generate a **context-specific list of essential medical questions** that the doctor:
    - **has already addressed**, or
    - **should have asked** to complete the evaluation.

    Your goal is to ensure the questions are **clinically relevant**, **non-repetitive**, and **adapted to the patient’s symptoms, diagnosis, and context** — not generic.

    ---

    ### **INSTRUCTIONS**

    For each question:
    1. **Infer context**: Identify what condition or system (e.g., respiratory, gastrointestinal, cardiovascular) the dictation focuses on.
    2. **Generate only relevant questions** for that context — avoid repeating standard questions unrelated to the case.
    3. **Determine if the question was answered** in the dictation:
      - If answered → extract the exact answer (verbatim or paraphrased).
      - If not → mark as `"needs_asking": true`.
    4. Assign each question a **category** from:
      - "chief_complaint"
      - "history"
      - "medications"
      - "allergies"
      - "vital_signs"
      - "physical_exam"
      - "assessment"
      - "plan"

    ---

    ### **OUTPUT FORMAT (JSON)**
    Return a JSON object with:
    - **questions**: A list of objects, each containing:
      - `"question"`: The question text
      - `"answer"`: The answer if found, or `null` if not
      - `"needs_asking"`: `true` or `false`
      - `"category"`: One of the categories listed above
    - **reasoning**: A short explanation (2–3 sentences) of how the questions were derived based on the context

    ---

    ### **ADDITIONAL RULES**
    - Tailor questions to the **patient’s symptoms, complaints, or suspected diagnosis**.
    - Avoid repeating similar or generic questions across categories.
    - Use **clinically precise** and **natural** language a real doctor would use.
    - Always ensure full coverage of all key categories, even if some are brief.
    - Do **not** add explanations outside the JSON.

    ---

    ### **INPUT TEXT (Dictation):**
    {translated_text}

    ---

    ### **EXAMPLE OUTPUT**
    {{
      "questions": [
        {{
          "question": "When did the cough and fever start?",
          "answer": "The patient reports both began five days ago.",
          "needs_asking": false,
          "category": "chief_complaint"
        }},
        {{
          "question": "Has the patient experienced any shortness of breath or chest pain?",
          "answer": null,
          "needs_asking": true,
          "category": "history"
        }},
        {{
          "question": "Is the patient currently taking any medications for the fever?",
          "answer": "Paracetamol as needed.",
          "needs_asking": false,
          "category": "medications"
        }},
        {{
          "question": "Does the patient have any drug allergies?",
          "answer": null,
          "needs_asking": true,
          "category": "allergies"
        }}
      ],
      "reasoning": "The dictation suggests an acute respiratory infection. The generated questions focus on symptom duration, associated respiratory signs, medication use, and allergy status to ensure clinical completeness."
    }}
    """


def get_question_generation_prompt_llama_conversation(translated_text):
    return f"""
    You are a **clinical reasoning expert** analyzing a **doctor–patient conversation**.

    Your task: Identify **context-specific** medical questions that were:
    1. **Already asked and answered** during the conversation, or
    2. **Clinically important but missing**, and should have been asked based on the patient’s symptoms and condition.

    ---

    ### **INSTRUCTIONS**

    1. **Comprehend the context:**
      - Identify the main medical issue or system involved (e.g., respiratory, cardiovascular, gastrointestinal, musculoskeletal, neurological, etc.).
      - Understand the patient’s presentation, progression, and any clues from the dialogue.

    2. **Generate only relevant and diverse questions** based on the specific conversation.
      - Avoid repetitive or generic questions that are not tied to the patient's case.
      - Each question should make clinical sense given the symptoms and dialogue content.

    3. **For each question:**
      - `"question"` → Write a precise medical question a doctor would ask in this context.
      - `"answer"` → Extract the answer **verbatim or paraphrased** from the patient’s replies, if available. If not, use `null`.
      - `"needs_asking"` → `false` if it was already asked and answered, `true` if it should still be asked.
      - `"category"` → One of:
        - "chief_complaint"
        - "history"
        - "medications"
        - "allergies"
        - "vital_signs"
        - "physical_exam"
        - "assessment"
        - "plan"

    4. **Ensure full coverage**:
      - Try to include at least one question per relevant category, but only when clinically justified.

    ---

    ### **OUTPUT FORMAT (JSON)**
    Return a JSON object:
    - `"questions"`: List of structured question objects as defined above
    - `"reasoning"`: A concise explanation (2–4 sentences) describing:
      - The clinical context inferred from the conversation
      - Why these questions were chosen
      - How they address any missing clinical information

    ---

    ### **ADDITIONAL RULES**
    - Adapt dynamically — questions must change depending on the condition (e.g., chest pain vs abdominal pain vs fever).
    - Reflect real clinical judgment, not template repetition.
    - Keep the reasoning short, factual, and focused on the dialogue.
    - Return **only** the JSON output. Do not include commentary or extra text.

    ---

    ### **CONVERSATION:**
    {translated_text}

    ---

    ### **EXAMPLE OUTPUT**
    {{
      "questions": [
        {{
          "question": "When did the cough and fever start?",
          "answer": "Five days ago.",
          "needs_asking": false,
          "category": "chief_complaint"
        }},
        {{
          "question": "Have you noticed any shortness of breath or chest tightness?",
          "answer": null,
          "needs_asking": true,
          "category": "history"
        }},
        {{
          "question": "Are you currently using your inhaler?",
          "answer": "Yes, the patient uses an albuterol inhaler as needed.",
          "needs_asking": false,
          "category": "medications"
        }},
        {{
          "question": "Do you have any allergies to medications?",
          "answer": null,
          "needs_asking": true,
          "category": "allergies"
        }},
        {{
          "question": "What is your current temperature?",
          "answer": null,
          "needs_asking": true,
          "category": "vital_signs"
        }}
      ],
      "reasoning": "The conversation indicates a respiratory complaint with cough and fever, likely an acute infection. The doctor covered symptom duration and medication use but missed allergy and vital sign questions, which are clinically relevant to this context."
    }}
    """
