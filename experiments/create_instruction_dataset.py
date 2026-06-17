"""
Instruction Following Dataset Creator
======================================
Builds a template-based instruction following evaluation dataset for the 17
tokenizer-winner languages. Domain: Talking Avatar customer service.

Each language gets exactly 1000 samples across 5 categories (200 each):
  - tone_style        — respond with a specific tone (friendly, formal, empathetic…)
  - length_constraint — respond within a word/sentence limit
  - language_compliance — respond only in the target language, no mixing
  - topic_boundary    — stay on context, don't speculate
  - structured_output — numbered lists, greetings, Q&A format…

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
        n_prompts = len(USER_PROMPTS)
        count = cat["count"]

        for i in range(count):
            instr_idx = i % n_instructions
            prompt_idx = i % n_prompts

            instruction = resolve(cat["instructions"][instr_idx], lang_name)
            constraint = resolve(cat["constraints"][instr_idx], lang_name)
            user_prompt = USER_PROMPTS[prompt_idx]

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
