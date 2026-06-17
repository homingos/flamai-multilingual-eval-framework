"""
Instruction Following Dataset Creator
======================================
Builds a template-based instruction following evaluation dataset for the 17
tokenizer-winner languages. Domain: Talking Avatar customer service.

Each language gets exactly 1200 samples across 6 categories (200 each):
  - tone_style           — respond with a specific tone (friendly, formal, empathetic…)
  - length_constraint    — respond within a word/sentence limit
  - language_compliance  — respond only in the target language, no mixing
  - topic_boundary       — stay on context, don't speculate
  - structured_output    — numbered lists, greetings, Q&A format…
  - number_verbalization — TTS normalization: postal/phone/OTP → digit-by-digit,
                           money/measurement/percentage → word form

Output: data/datasets/instructions/<language_slug>/samples.jsonl
        data/datasets/instructions/meta.json  (summary across all languages)

Usage:
  python experiments/create_instruction_dataset.py
  python experiments/create_instruction_dataset.py --language Tamil
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "datasets" / "instructions"

WINNERS = [
    {"name": "Tamil",                "slug": "tamil",                "region": "Indic",       "winner_model": "Tamil-Mistral-7B"},
    {"name": "Marathi",              "slug": "marathi",              "region": "Indic",       "winner_model": "MahaMarathi-7B"},
    {"name": "Kannada",              "slug": "kannada",              "region": "Indic",       "winner_model": "Ambari-7B"},
    {"name": "Gujarati",             "slug": "gujarati",             "region": "Indic",       "winner_model": "Gujju-Llama-7B"},
    {"name": "Arabic",               "slug": "arabic",               "region": "Middle East", "winner_model": "Jais-2-8B"},
    {"name": "Hebrew",               "slug": "hebrew",               "region": "Middle East", "winner_model": "DictaLM-2.0-7B"},
    {"name": "Korean",               "slug": "korean",               "region": "East Asia",   "winner_model": "Polyglot-Ko-12B"},
    {"name": "Malay",                "slug": "malay",                "region": "SEA",         "winner_model": "MaLLaM-5B"},
    {"name": "Swahili",              "slug": "swahili",              "region": "Africa",      "winner_model": "Swahili-Gemma-7B"},
    {"name": "Amharic",              "slug": "amharic",              "region": "Africa",      "winner_model": "Walia-LLM-7B"},
    {"name": "French",               "slug": "french",               "region": "Europe",      "winner_model": "Lucie-7B"},
    {"name": "Swedish",              "slug": "swedish",              "region": "Europe",      "winner_model": "Viking-7B"},
    {"name": "Czech",                "slug": "czech",                "region": "Europe",      "winner_model": "CSMPT-7B"},
    {"name": "Greek",                "slug": "greek",                "region": "Europe",      "winner_model": "Meltemi-7B"},
    {"name": "Brazilian Portuguese", "slug": "brazilian_portuguese", "region": "Americas",    "winner_model": "Tucano-2b4"},
    {"name": "Māori",                "slug": "maori",                "region": "Oceania",     "winner_model": "Goldfish-mri-39M"},
    {"name": "Tok Pisin",            "slug": "tok_pisin",            "region": "Oceania",     "winner_model": "Goldfish-tpi-125M"},
]

# ── Talking Avatar user prompts (40 base prompts, customer-service domain) ────
USER_PROMPTS = [
    "How do I reset my password?",
    "What are the steps to update my account details?",
    "Can you explain the return policy?",
    "I'm having trouble with my payment. What should I do?",
    "How do I contact customer support?",
    "What are your business hours?",
    "How can I track my order?",
    "What documents do I need to submit?",
    "How long does the verification process take?",
    "Can I change my registered email address?",
    "What are the available payment methods?",
    "How do I cancel my subscription?",
    "What happens if I miss a payment?",
    "How do I download the app?",
    "Can I use the service on multiple devices?",
    "What is the refund timeline?",
    "How do I update my phone number?",
    "What are the eligibility criteria?",
    "How do I report a problem?",
    "What languages does the service support?",
    "How do I change my notification preferences?",
    "What is the maximum file size I can upload?",
    "How do I export my data?",
    "Can I add multiple users to my account?",
    "What security measures protect my data?",
    "How do I enable two-factor authentication?",
    "What should I do if I forget my username?",
    "How do I close my account?",
    "Are there any transaction limits?",
    "How do I get a receipt or invoice?",
    "What are the terms and conditions?",
    "How do I provide feedback?",
    "Is my data stored securely?",
    "How do I link my bank account?",
    "What happens if I enter the wrong PIN?",
    "Can I schedule a callback?",
    "How do I access my transaction history?",
    "What do I do if my card is declined?",
    "How do I set up automatic payments?",
    "Where can I find the user manual?",
]

# ── Category templates ─────────────────────────────────────────────────────────

TONE_STYLE_INSTRUCTIONS = [
    "You are a helpful avatar assistant. Respond in {language} with a warm, friendly tone as if speaking to a close friend.",
    "You are a professional customer service avatar. Respond in {language} using formal, respectful language.",
    "You are an empathetic avatar assistant. Respond in {language} with understanding and compassion — the user may be frustrated.",
    "You are a concise avatar assistant. Respond in {language} in a direct, no-nonsense way without pleasantries.",
    "You are a helpful avatar. Respond in {language} using simple, easy-to-understand language suitable for non-technical users.",
    "You are an enthusiastic avatar assistant. Respond in {language} with energy and positivity.",
    "You are a calm, reassuring avatar. Respond in {language} in a soothing, confident tone.",
    "You are a professional avatar. Respond in {language} in a neutral, informative tone without emotional language.",
    "You are a supportive avatar assistant. Respond in {language} with encouragement and patience.",
    "You are a knowledgeable avatar. Respond in {language} in an authoritative but approachable manner.",
]

TONE_CONSTRAINTS = [
    {"tone": "friendly"},
    {"tone": "formal"},
    {"tone": "empathetic"},
    {"tone": "direct"},
    {"tone": "simple"},
    {"tone": "enthusiastic"},
    {"tone": "reassuring"},
    {"tone": "neutral"},
    {"tone": "supportive"},
    {"tone": "authoritative"},
]

LENGTH_INSTRUCTIONS = [
    "You are a helpful avatar assistant. Respond in {language} in exactly 1 sentence.",
    "You are a helpful avatar assistant. Respond in {language} in exactly 2 sentences.",
    "You are a helpful avatar assistant. Respond in {language} in exactly 3 sentences.",
    "You are a helpful avatar assistant. Respond in {language} in under 20 words.",
    "You are a helpful avatar assistant. Respond in {language} in under 50 words.",
    "You are a helpful avatar assistant. Respond in {language} in 3 to 5 sentences.",
    "You are a helpful avatar assistant. Provide a detailed response in {language} with at least 5 sentences.",
    "You are a helpful avatar assistant. Give a one-sentence summary in {language}.",
    "You are a helpful avatar assistant. Respond in {language} with a summary of no more than 2 sentences, followed by up to 3 bullet points.",
    "You are a helpful avatar assistant. Respond in {language} as briefly as possible while still being helpful.",
]

LENGTH_CONSTRAINTS = [
    {"max_sentences": 1},
    {"max_sentences": 2},
    {"max_sentences": 3},
    {"max_words": 20},
    {"max_words": 50},
    {"min_sentences": 3, "max_sentences": 5},
    {"min_sentences": 5},
    {"max_sentences": 1},
    {"max_sentences": 2, "bullet_points": True},
    {"style": "as_brief_as_possible"},
]

LANGUAGE_COMPLIANCE_INSTRUCTIONS = [
    "You are a helpful avatar assistant. Always respond in {language} only. Never use English words or phrases.",
    "You are a helpful avatar assistant. Respond strictly in {language}. Do not mix any other language.",
    "You are a helpful avatar assistant. Your response must be entirely in {language}. If a term has no direct equivalent, describe it in {language} instead of borrowing the English word.",
    "You are a {language}-speaking avatar assistant. Respond only in {language}, even if the user writes in another language.",
    "You are a helpful avatar assistant. The user speaks {language}. Always respond in {language}, no exceptions.",
    "You are a helpful avatar assistant. Reply in {language} and avoid all English loanwords where a native {language} equivalent exists.",
    "You are a helpful avatar assistant. Respond in {language}. Do not switch languages at any point in your response.",
    "You are a bilingual avatar assistant. The user prefers {language}. Always respond in {language}.",
    "You are a helpful avatar assistant. The conversation language is {language}. Maintain this throughout your entire response.",
    "You are a helpful avatar assistant. Translate and respond in {language} regardless of what language the question is asked in.",
]

LANGUAGE_COMPLIANCE_CONSTRAINTS = [
    {"language": "{language}", "no_english": True},
    {"language": "{language}", "no_mixing": True},
    {"language": "{language}", "no_loanwords": True},
    {"language": "{language}"},
    {"language": "{language}", "strict": True},
    {"language": "{language}", "avoid_loanwords": True},
    {"language": "{language}", "no_switching": True},
    {"language": "{language}", "preferred": True},
    {"language": "{language}", "maintain_throughout": True},
    {"language": "{language}", "translate_regardless": True},
]

TOPIC_BOUNDARY_INSTRUCTIONS = [
    "You are a helpful avatar assistant responding in {language}. Answer only based on the following context. If the information is not in the context, say 'I don't have that information.' Context: [USER'S ACCOUNT AND SERVICE INFORMATION]",
    "You are a helpful avatar assistant responding in {language}. Do not speculate or provide information beyond what is explicitly stated in the provided context.",
    "You are a helpful avatar assistant responding in {language}. Stay strictly on the topic of the user's question. Do not provide unrequested additional information.",
    "You are a helpful avatar assistant responding in {language}. Answer only what is asked. Do not volunteer extra information or make assumptions.",
    "You are a helpful avatar assistant responding in {language}. If you are uncertain about the answer, say so clearly. Do not guess.",
    "You are a helpful avatar assistant responding in {language}. Answer only questions related to account management and customer service. Politely decline unrelated topics.",
    "You are a helpful avatar assistant responding in {language}. Provide only factual information. Do not give opinions or personal recommendations.",
    "You are a helpful avatar assistant responding in {language}. If the user's question is outside your knowledge, redirect them to customer support rather than guessing.",
    "You are a helpful avatar assistant responding in {language}. Answer concisely and only address the specific question asked, nothing more.",
    "You are a helpful avatar assistant responding in {language}. Do not make promises or commitments on behalf of the company. Only describe existing features and processes.",
]

TOPIC_BOUNDARY_CONSTRAINTS = [
    {"context_grounded": True, "unknown_fallback": "state_unknown"},
    {"no_speculation": True},
    {"on_topic_only": True},
    {"answer_only_what_asked": True},
    {"acknowledge_uncertainty": True},
    {"domain": "account_management_only"},
    {"factual_only": True},
    {"redirect_if_unknown": True},
    {"no_extra_info": True},
    {"no_promises": True},
]

STRUCTURED_OUTPUT_INSTRUCTIONS = [
    "You are a helpful avatar assistant. Respond in {language} as a numbered list of steps.",
    "You are a helpful avatar assistant. Respond in {language} with: 1) A one-sentence answer, 2) A brief explanation, 3) A next step the user can take.",
    "You are a helpful avatar assistant. Start your response with a greeting, provide the answer, and end with an offer to help further. Respond in {language}.",
    "You are a helpful avatar assistant. Respond in {language} using bullet points for any list of items. Use plain sentences otherwise.",
    "You are a helpful avatar assistant. Structure your {language} response as: [Summary] | [Solution] | [Next Steps].",
    "You are a helpful avatar assistant. Begin with 'Here is what you need to do:' and then provide numbered steps in {language}.",
    "You are a helpful avatar assistant. Respond in {language} with a brief answer first, then a more detailed explanation below.",
    "You are a helpful avatar assistant. Use a Q&A format in {language}: restate the question, then answer it.",
    "You are a helpful avatar assistant. Respond in {language} and end your response with 'Is there anything else I can help you with?'",
    "You are a helpful avatar assistant. Format your {language} response with a header summarizing the topic, followed by the details.",
]

STRUCTURED_OUTPUT_CONSTRAINTS = [
    {"format": "numbered_list"},
    {"format": "three_part", "parts": ["answer", "explanation", "next_step"]},
    {"format": "greeting_answer_offer"},
    {"format": "bullet_points_for_lists"},
    {"format": "summary_solution_nextsteps"},
    {"format": "numbered_steps", "opener": "Here is what you need to do:"},
    {"format": "brief_then_detailed"},
    {"format": "qa_restatement"},
    {"format": "ends_with_offer"},
    {"format": "header_then_details"},
]

# ── Number verbalization prompts (200 samples across 8 number types) ──────────
# Each prompt contains a sentence with numbers; the model must verbalize them
# correctly in the target language using the digit-by-digit or word-form rule.

NUMBER_VERBALIZATION_INSTRUCTIONS = [
    "You are a TTS normalization assistant for a talking avatar. Convert all numbers in the following sentence to their spoken form in {language}. Rule: postal codes, phone numbers, OTPs, PINs, and reference IDs → say each digit individually. Currency, quantities, measurements, percentages, and dates → convert to natural spoken words. Output only the normalized sentence in {language}.",
    "You are a talking avatar that speaks in {language}. When reading the text aloud, apply these rules: postal codes, phone numbers, PIN codes, OTPs → read each digit one by one. Money amounts, measurements, percentages, and large quantities → say them in words. Return the full text as it should be spoken in {language}.",
    "You are a speech normalization engine for a {language}-speaking avatar. Transform the following sentence so that all numbers are written as they would be spoken. Digit-by-digit: postal codes, phone numbers, reference numbers, OTPs. Word form: currency, decimal values, percentages, dates, and quantities. Reply in {language}.",
    "You are a helpful avatar assistant speaking in {language}. Read the following sentence aloud — convert each number to its correct spoken form in {language}. If the number is a postal code, phone number, OTP, PIN, or ID, say each digit separately. If it is a money amount, measurement, date, percentage, or quantity, say it naturally as a word. Return only the spoken version.",
    "You are a TTS preprocessing assistant. The following sentence will be passed to a text-to-speech engine for a {language}-speaking avatar. Rewrite it so that: (a) postal codes, phone numbers, PINs, OTPs, and transaction IDs are expanded digit-by-digit, and (b) currency amounts, measurements, decimals, percentages, and dates are written as {language} words. Return only the rewritten sentence.",
    "You are a {language} avatar assistant. Normalize the numbers in the following sentence for speech output. Speak postal codes, phone numbers, OTPs, and reference IDs as individual digits. Speak money amounts, quantities, measurements, percentages, and dates as full {language} words. Return the normalized sentence in {language}.",
    "You are a helpful avatar. The sentence below contains numbers that must be converted to speech in {language}. Postal codes, phone numbers, PINs, OTPs → digit by digit. Currency, measurement values, percentages, dates → in words. Output the full sentence with all numbers converted, in {language}.",
    "You are a speech-ready avatar assistant for {language}. Rewrite the following sentence replacing every number with its spoken equivalent in {language}. Numbers that are identifiers (postal codes, phone numbers, OTPs, PINs, IDs) → individual digits. Numbers that express quantity or value (money, measurements, percentages, dates) → natural {language} word form.",
    "You are a TTS-ready avatar. Prepare the following sentence for {language} speech by converting all numeric expressions. Identifiers like postal codes, phone numbers, OTP codes, and reference IDs should be read one digit at a time. Value expressions like prices, measurements, percentages, and dates should be converted to {language} words. Return the result.",
    "You are a multilingual TTS normalization assistant. For the {language}-speaking avatar, rewrite the sentence below so every number is in its correct spoken form. Rule 1 (identifiers): postal codes, phone numbers, PIN codes, OTPs, and IDs → expand digit by digit. Rule 2 (values): currency, decimal measurements, percentages, quantities, and dates → convert to {language} number words. Output only the rewritten sentence.",
]

# Prompts organized by number_type (for labeling expected_constraints)
NUMBER_VERBALIZATION_PROMPTS = [
    # ── Postal / ZIP codes (digit-by-digit) ───────────────────────────────────
    {"prompt": "Your delivery address postal code is 560066.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["560066"]},
    {"prompt": "Please confirm your PIN code for the area: 110001.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["110001"]},
    {"prompt": "The ZIP code for this service area is 90210.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["90210"]},
    {"prompt": "Enter your postal code 400001 to check delivery availability.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["400001"]},
    {"prompt": "Your registered postal code on file is 600020.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["600020"]},
    {"prompt": "We service the 30301 zip code area.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["30301"]},
    {"prompt": "Please update your postal code to 560100 in your profile.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["560100"]},
    {"prompt": "The postcode for your branch is 10001.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["10001"]},
    {"prompt": "Deliveries to postal code 700001 take 2 business days.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["700001"]},
    {"prompt": "Your billing address is linked to PIN 411001.", "number_type": "postal_code", "rule": "digit_by_digit", "numbers": ["411001"]},
    # ── Phone numbers (digit-by-digit) ────────────────────────────────────────
    {"prompt": "Call our support team at +91 98765 43210.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["+91 98765 43210"]},
    {"prompt": "The customer helpline number is 1800 123 4567.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["1800 123 4567"]},
    {"prompt": "Please call us back at +1 555 867 5309.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["+1 555 867 5309"]},
    {"prompt": "The emergency contact number is +44 7911 123456.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["+44 7911 123456"]},
    {"prompt": "You can reach the agent at 080 4567 8901.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["080 4567 8901"]},
    {"prompt": "Missed call from +91 70123 45678 — this is our verification line.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["+91 70123 45678"]},
    {"prompt": "Your registered mobile number ending in 4321 will receive the OTP.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["4321"]},
    {"prompt": "For billing queries, dial 1900 200 3000.", "number_type": "phone_number", "rule": "digit_by_digit", "numbers": ["1900 200 3000"]},
    # ── OTP / PIN / Reference IDs (digit-by-digit) ────────────────────────────
    {"prompt": "Your OTP is 483921. Please enter it within 5 minutes.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["483921"]},
    {"prompt": "The transaction reference ID is 7823456.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["7823456"]},
    {"prompt": "Your 4-digit PIN is 2847. Do not share it with anyone.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["2847"]},
    {"prompt": "Enter OTP 591034 to confirm your payment.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["591034"]},
    {"prompt": "Your ticket number is 90034712.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["90034712"]},
    {"prompt": "Reset code 3390 has been sent to your email.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["3390"]},
    {"prompt": "Complaint reference number 20240617 has been created.", "number_type": "otp_pin", "rule": "digit_by_digit", "numbers": ["20240617"]},
    # ── Currency / Money (word form) ──────────────────────────────────────────
    {"prompt": "Your account balance is ₹1,50,000.", "number_type": "currency", "rule": "word_form", "numbers": ["₹1,50,000"]},
    {"prompt": "The transaction amount is $2,499.99.", "number_type": "currency", "rule": "word_form", "numbers": ["$2,499.99"]},
    {"prompt": "You owe €350.75 in pending dues.", "number_type": "currency", "rule": "word_form", "numbers": ["€350.75"]},
    {"prompt": "A refund of ¥45,000 has been processed to your account.", "number_type": "currency", "rule": "word_form", "numbers": ["¥45,000"]},
    {"prompt": "The subscription fee is $9.99 per month.", "number_type": "currency", "rule": "word_form", "numbers": ["$9.99"]},
    {"prompt": "Minimum wallet balance required is ₹500.", "number_type": "currency", "rule": "word_form", "numbers": ["₹500"]},
    {"prompt": "Your cashback reward of £12.50 has been credited.", "number_type": "currency", "rule": "word_form", "numbers": ["£12.50"]},
    {"prompt": "The late payment penalty is $25.", "number_type": "currency", "rule": "word_form", "numbers": ["$25"]},
    {"prompt": "International transfer fee is €4.99.", "number_type": "currency", "rule": "word_form", "numbers": ["€4.99"]},
    {"prompt": "Your outstanding loan amount is ₹2,75,500.", "number_type": "currency", "rule": "word_form", "numbers": ["₹2,75,500"]},
    {"prompt": "The plan upgrade costs $49.99 per year.", "number_type": "currency", "rule": "word_form", "numbers": ["$49.99"]},
    {"prompt": "You received a bonus credit of ₹250 in your wallet.", "number_type": "currency", "rule": "word_form", "numbers": ["₹250"]},
    # ── Decimal / Measurements (word form) ────────────────────────────────────
    {"prompt": "The package weighs 2.5 kg.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["2.5"]},
    {"prompt": "Body temperature reading is 98.6°F.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["98.6"]},
    {"prompt": "The dosage prescribed is 0.5 mg per kg of body weight.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["0.5"]},
    {"prompt": "Your current upload speed is 12.3 Mbps.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["12.3"]},
    {"prompt": "The item dimensions are 30.5 cm × 22.0 cm.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["30.5", "22.0"]},
    {"prompt": "Blood sugar level recorded is 6.2 mmol/L.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["6.2"]},
    {"prompt": "The delivery weighs 0.75 kg and will arrive tomorrow.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["0.75"]},
    {"prompt": "Current air quality index is 153.4 in your area.", "number_type": "decimal_measurement", "rule": "word_form", "numbers": ["153.4"]},
    # ── Percentages (word form) ────────────────────────────────────────────────
    {"prompt": "You have used 87.5% of your monthly data.", "number_type": "percentage", "rule": "word_form", "numbers": ["87.5%"]},
    {"prompt": "A 15% service charge has been added to your bill.", "number_type": "percentage", "rule": "word_form", "numbers": ["15%"]},
    {"prompt": "The interest rate on your loan is 8.75% per annum.", "number_type": "percentage", "rule": "word_form", "numbers": ["8.75%"]},
    {"prompt": "Your savings account offers 6.5% annual interest.", "number_type": "percentage", "rule": "word_form", "numbers": ["6.5%"]},
    {"prompt": "Battery is at 23% — please charge your device.", "number_type": "percentage", "rule": "word_form", "numbers": ["23%"]},
    {"prompt": "This month's cashback rate is 2%.", "number_type": "percentage", "rule": "word_form", "numbers": ["2%"]},
    {"prompt": "GST applicable on this transaction is 18%.", "number_type": "percentage", "rule": "word_form", "numbers": ["18%"]},
    {"prompt": "Your plan renews with a 10% loyalty discount.", "number_type": "percentage", "rule": "word_form", "numbers": ["10%"]},
    # ── Large quantities (word form) ───────────────────────────────────────────
    {"prompt": "Over 1,500,000 customers trust our service.", "number_type": "quantity", "rule": "word_form", "numbers": ["1,500,000"]},
    {"prompt": "Your plan allows 10,000 API calls per month.", "number_type": "quantity", "rule": "word_form", "numbers": ["10,000"]},
    {"prompt": "The file size limit for uploads is 250 MB.", "number_type": "quantity", "rule": "word_form", "numbers": ["250"]},
    {"prompt": "We have processed 5,00,000 transactions this year.", "number_type": "quantity", "rule": "word_form", "numbers": ["5,00,000"]},
    {"prompt": "Your storage plan includes 100 GB of cloud space.", "number_type": "quantity", "rule": "word_form", "numbers": ["100"]},
    {"prompt": "This product has 4,200 five-star reviews.", "number_type": "quantity", "rule": "word_form", "numbers": ["4,200"]},
    {"prompt": "The promotion is valid for the first 500 customers.", "number_type": "quantity", "rule": "word_form", "numbers": ["500"]},
    # ── Dates (word form) ─────────────────────────────────────────────────────
    {"prompt": "Your subscription expires on 12/31/2026.", "number_type": "date", "rule": "word_form", "numbers": ["12/31/2026"]},
    {"prompt": "The appointment is scheduled for 06/17/2026.", "number_type": "date", "rule": "word_form", "numbers": ["06/17/2026"]},
    {"prompt": "Documents must be submitted before 01/15/2027.", "number_type": "date", "rule": "word_form", "numbers": ["01/15/2027"]},
    {"prompt": "Your account was created on 03/22/2024.", "number_type": "date", "rule": "word_form", "numbers": ["03/22/2024"]},
    {"prompt": "The offer is valid until 07/04/2026.", "number_type": "date", "rule": "word_form", "numbers": ["07/04/2026"]},
    # ── Mixed — sentence with multiple number types (both rules apply) ─────────
    {"prompt": "Your order #7823456 worth $149.99 will be delivered to postal code 560066.", "number_type": "mixed", "rule": "mixed", "numbers": ["7823456", "$149.99", "560066"]},
    {"prompt": "Call +91 98765 43210 to claim your 15% discount of $25.", "number_type": "mixed", "rule": "mixed", "numbers": ["+91 98765 43210", "15%", "$25"]},
    {"prompt": "Enter OTP 482910 to confirm the ₹2,500 payment.", "number_type": "mixed", "rule": "mixed", "numbers": ["482910", "₹2,500"]},
    {"prompt": "Shipment to ZIP 10001 weighs 1.5 kg and costs $12.99.", "number_type": "mixed", "rule": "mixed", "numbers": ["10001", "1.5", "$12.99"]},
    {"prompt": "Your PIN 3948 unlocks a ₹500 reward on 08/01/2026.", "number_type": "mixed", "rule": "mixed", "numbers": ["3948", "₹500", "08/01/2026"]},
    {"prompt": "Account #90034712 has a balance of €1,200.50 and earns 4.5% interest.", "number_type": "mixed", "rule": "mixed", "numbers": ["90034712", "€1,200.50", "4.5%"]},
    {"prompt": "The 10,000 MB plan at $9.99/month renews on 09/30/2026.", "number_type": "mixed", "rule": "mixed", "numbers": ["10,000", "$9.99", "09/30/2026"]},
    {"prompt": "Send your complaint to helpline 1800 123 4567 with reference 20240617.", "number_type": "mixed", "rule": "mixed", "numbers": ["1800 123 4567", "20240617"]},
    {"prompt": "Postal code 400001 delivery incurs a $4.99 fee for packages over 2.5 kg.", "number_type": "mixed", "rule": "mixed", "numbers": ["400001", "$4.99", "2.5"]},
    {"prompt": "Use OTP 519302 to transfer ₹75,000 — valid until 11:59 PM on 06/17/2026.", "number_type": "mixed", "rule": "mixed", "numbers": ["519302", "₹75,000", "06/17/2026"]},
]

NUMBER_VERBALIZATION_CONSTRAINTS = [
    {
        "number_type": p["number_type"],
        "rule": p["rule"],
        "numbers_in_prompt": p["numbers"],
        "digit_by_digit_types": ["postal_code", "phone_number", "otp_pin"],
        "word_form_types": ["currency", "decimal_measurement", "percentage", "quantity", "date"],
    }
    for p in NUMBER_VERBALIZATION_PROMPTS
]

CATEGORIES = [
    {
        "key": "tone_style",
        "instructions": TONE_STYLE_INSTRUCTIONS,
        "constraints": TONE_CONSTRAINTS,
        "count": 200,
    },
    {
        "key": "length_constraint",
        "instructions": LENGTH_INSTRUCTIONS,
        "constraints": LENGTH_CONSTRAINTS,
        "count": 200,
    },
    {
        "key": "language_compliance",
        "instructions": LANGUAGE_COMPLIANCE_INSTRUCTIONS,
        "constraints": LANGUAGE_COMPLIANCE_CONSTRAINTS,
        "count": 200,
    },
    {
        "key": "topic_boundary",
        "instructions": TOPIC_BOUNDARY_INSTRUCTIONS,
        "constraints": TOPIC_BOUNDARY_CONSTRAINTS,
        "count": 200,
    },
    {
        "key": "structured_output",
        "instructions": STRUCTURED_OUTPUT_INSTRUCTIONS,
        "constraints": STRUCTURED_OUTPUT_CONSTRAINTS,
        "count": 200,
    },
    {
        "key": "number_verbalization",
        "instructions": NUMBER_VERBALIZATION_INSTRUCTIONS,
        "constraints": NUMBER_VERBALIZATION_CONSTRAINTS,
        "prompts": NUMBER_VERBALIZATION_PROMPTS,  # uses its own prompts, not USER_PROMPTS
        "count": 200,
    },
]


def resolve(template, language):
    """Replace {language} placeholder in templates."""
    if isinstance(template, str):
        return template.replace("{language}", language)
    if isinstance(template, dict):
        return {k: resolve(v, language) for k, v in template.items()}
    return template


def build_samples(lang: dict) -> list:
    lang_name = lang["name"]
    slug = lang["slug"]
    samples = []
    sample_idx = 0

    for cat in CATEGORIES:
        n_instructions = len(cat["instructions"])
        count = cat["count"]

        # number_verbalization has its own prompt pool; others use USER_PROMPTS
        cat_prompts = cat.get("prompts", None)

        for i in range(count):
            instr_idx = i % n_instructions

            instruction = resolve(cat["instructions"][instr_idx], lang_name)

            if cat_prompts is not None:
                prompt_entry = cat_prompts[i % len(cat_prompts)]
                user_prompt = prompt_entry["prompt"]
                constraint = resolve(cat["constraints"][i % len(cat["constraints"])], lang_name)
            else:
                user_prompt = USER_PROMPTS[i % len(USER_PROMPTS)]
                constraint = resolve(cat["constraints"][instr_idx], lang_name)

            samples.append({
                "id": f"inst_{slug}_{cat['key']}_{i + 1:03d}",
                "language": lang_name,
                "region": lang["region"],
                "winner_model": lang["winner_model"],
                "category": cat["key"],
                "system_instruction": instruction,
                "user_prompt": user_prompt,
                "expected_constraints": constraint,
            })
            sample_idx += 1

    return samples


def process_language(lang: dict):
    print(f"  Building instruction dataset for {lang['name']}…", end=" ", flush=True)
    samples = build_samples(lang)

    out_dir = OUT_DIR / lang["slug"]
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "samples.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    category_counts = {}
    for cat in CATEGORIES:
        category_counts[cat["key"]] = len([s for s in samples if s["category"] == cat["key"]])

    meta = {
        "language": lang["name"],
        "slug": lang["slug"],
        "region": lang["region"],
        "winner_model": lang["winner_model"],
        "total_samples": len(samples),
        "categories": category_counts,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"{len(samples)} samples ({', '.join(f'{k}={v}' for k, v in category_counts.items())})")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", help="Process only this language (e.g. Tamil)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    langs = WINNERS
    if args.language:
        langs = [l for l in WINNERS if l["name"].lower() == args.language.lower()]
        if not langs:
            print(f"Unknown language '{args.language}'. Available: {[l['name'] for l in WINNERS]}")
            sys.exit(1)

    all_meta = []
    for lang in langs:
        meta = process_language(lang)
        all_meta.append(meta)

    summary = {
        "total_languages": len(all_meta),
        "total_samples": sum(m["total_samples"] for m in all_meta),
        "samples_per_language": 1000,
        "categories": {cat["key"]: cat["count"] for cat in CATEGORIES},
        "languages": all_meta,
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {summary['total_languages']} languages · {summary['total_samples']:,} total samples")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
