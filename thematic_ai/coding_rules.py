"""Error-derived coding rules, cues and contrastive examples.

Built from units the model missed or over-coded against the adjudicated gold.
The rules are generalisations, not copies of those units, so a unit being
scored never sees its own gold labels.
"""

from __future__ import annotations

import re

# Same-theme codes that humans treated as alternatives, not companions.
# Used by calibration so Jaccard-with-a-common-code does not hide a rare
# specific code (racial remark almost always sits next to Hate speech).
COMPETING_GROUPS: list[list[str]] = [
    [
        "Consistent offensive behavior",
        "One-off offensive",
        "one message is enough (offensive)",
        "Normal behavior",
        "Mix of offensive and non-offensive",
    ],
    [
        "Majority offesnive",
        "One-off offensive",
        "one message is enough (offensive)",
        "Mix of offensive and non-offensive",
        "Normal behavior",
    ],
    ["No offensive content", "Hate speech and offensive language"],
    ["No offensive content", "Threats of violence"],
    ["No offensive content", "Death wish"],
    ["No offensive content", "Harm other person"],
    ["No offensive content", "insult"],
    ["No offensive content", "Extreme offensive"],
    ["No offensive content", "racial remark"],
    ["No offensive content", "derogatory remarks"],
    ["Calm and thoughtful", "No offensive content"],
    ["Personal opinions and thoughts", "positive content"],
    ["Do not suspend", "Permanent ban", "Temporary suspension"],
    ["Obvious an clear suspension", "Permanent ban"],
]

# High-precision wording cues for codes the model under-recalled.
# Over-applied codes (Mix, one message is enough, Obvious, Do not suspend)
# are left out so we do not push the model to add them.
CODE_CUES: dict[str, tuple[str, ...]] = {
    "Death wish": (
        r"death threat",
        r"wish\w* death",
        r"wishing death",
        r"hope you die",
        r"telling someone to die",
        r"should die",
        r"deserve to die",
        r"bring(?:ing)? up death",
        r"hoped? .{0,20}die",
        r"wished death",
    ),
    "racial remark": (r"racis", r"racial"),
    "Extreme offensive": (
        r"extreme(?:ly)? offensive",
        r"severely offensive",
        r"brutally offensive",
        r"highly offensive",
        r"sickening",
    ),
    "Harm other person": (
        r"\bharm(?:ful|ing)?\b",
        r"hurtful",
        r"intent to harm",
        r"cause distress",
        r"bring people down",
        r"to harm",
    ),
    "insult": (r"insult", r"name.?call", r"really rude"),
    "Consistent offensive behavior": (
        r"every comment",
        r"all (?:of )?(?:the |their )?messages",
        r"both messages",
        r"\bmultiple\b",
        r"history of",
        r"\bnumerous\b",
        r"\brepeated\b",
        r"\bfrequent\b",
        r"standard behavior",
        r"kept going",
        r"all \d",
        r"more often than not",
        r"\d(?:/\d| out of \d)",
        r"a lot of .{0,20}offensive",
        r"nothing but .{0,30}(?:hurtful|offensive|hate|attack)",
    ),
    "Majority offesnive": (r"majority", r"most of", r"vast majority", r"more often than not"),
    "Calm and thoughtful": (
        r"\bcalm\b",
        r"thoughtful",
        r"eloquent",
        r"genuine",
        r"supportive",
        r"considerate",
        r"uplifting",
        r"nice and cool",
        r"appreciative",
    ),
    "Normal behavior": (r"\bnormal\b", r"\bordinary\b", r"typical", r"standard .{0,12}behavior"),
    "Personal opinions and thoughts": (
        r"\bopinions?\b",
        r"expressing .{0,20}thoughts",
        r"stating a fact",
        r"general opinion",
        r"their thoughts",
    ),
    "positive content": (
        r"\bpositive content\b",
        r"\bpositive message",
        r"complement",
        r"gratitude",
        r"good words",
        r"friendly",
    ),
    "Consequences beyond suspencion": (
        r"police",
        r"law enforcement",
        r"arrest",
        r"therap",
        r"watch list",
        r"mental health",
        r"investigat",
        r"more than suspension",
        r"beyond",
        r"authorities",
    ),
    "Permanent ban": (
        r"taken off the platform",
        r"account termination",
        r"permanent ban",
        r"permanently",
        r"should not be allowed to continue",
        r"banned from",
        r"account termination",
    ),
    "Need to know policy and TOS": (
        r"polic(?:y|ies)",
        r"terms of service",
        r"\bTOS\b",
        r"community guidelines",
    ),
    "Expressed uncertainty": (
        r"not .{0,12}confident",
        r"not sure",
        r"don't know",
        r"\biffy\b",
        r"enough info",
        r"enough information",
        r"lack of",
    ),
    "Contextual understanding": (r"\bcontext\b", r"banter", r"being bothered", r"misread"),
    "Appeal to intent": (r"intent", r"intention", r"humour", r"humor", r"venting", r"joke"),
    "Not too offensive": (
        r"not too offensive",
        r"wasn't enough",
        r"hardly offensive",
        r"not offensive enough",
        r"only verbal",
    ),
    "Poster is a bad person": (
        r"bad (?:person|poster)",
        r"horrible .{0,12}person",
        r"angry punk",
        r"sick(?:ening)? person",
        r"they are a bad",
    ),
    "One-off offensive": (r"only one .{0,40}offensive", r"one offensive", r"isolated"),
    "Needed to look at more messages": (r"should have .{0,20}(?:reviewed|looked|kept going)", r"more messages"),
    "Would revise the decision": (r"should have .{0,20}(?:suspended|clicked|chosen)", r"would now", r"looking back"),
    "need to contact police, or authorities.": (r"police", r"law enforcement", r"authorities"),
    "Ban if behavior continues": (r"if .{0,30}contin", r"if so, a definite"),
    "negative consequences": (r"hostile environment", r"negative consequences", r"potentially harmful"),
    "bullying": (r"bully", r"harass"),
    "sexual discrimination": (r"sexist", r"about women", r"gender"),
    "scary": (r"\bscary\b", r"\bscared\b"),
}

_COMPILED: dict[str, tuple[re.Pattern[str], ...]] = {
    code: tuple(re.compile(p, re.IGNORECASE) for p in patterns)
    for code, patterns in CODE_CUES.items()
}


def matched_codes(text: str) -> list[str]:
    """Codes whose wording cues fire in `text`, in codebook-priority order."""
    if not text:
        return []
    hit = [code for code, patterns in _COMPILED.items() if any(p.search(text) for p in patterns)]
    return hit


CODING_GUIDANCE = """\
## HOW TO CHOOSE CODES (from typical human/model disagreements)

Apply a code when the participant actually makes that point. Specific wording
beats a nearby broader code. Two different points in the same justification
get two codes.

### Type of offensiveness
- Death wish: wishing death, "death threat(s)", "hope you die", "telling someone \
to die", or bringing up death as the problem. "It's a death threat" is Death wish, \
not Threats of violence.
- Threats of violence: threatening to kill/harm, violence, murder, attacking. \
Apply alongside Death wish when BOTH wishing death AND a threat to commit violence \
are named.
- Hate speech and offensive language: slurs, hate speech, homophobia, targeted \
attacks, generic "offensive"/"hateful" language. Do NOT drop a more specific \
code because this one also fits.
- racial remark: they mention racism/racist/racial. Usually WITH Hate speech, \
not instead of it.
- Extreme offensive: they call the content extreme/severe/brutal/highly offensive. \
Usually WITH Hate speech.
- Harm other person: harm, harmful, hurtful, intent to harm, distress. Keep it \
even if Threats of violence or Hate speech is also applied.
- insult: they say insult/insulting/rude/name-calling and do not mention slurs \
or hate speech. If they only say "really rude", prefer insult over Hate speech.
- No offensive content: they say nothing was offensive / no reason based on \
content. Do not add it when the main characterisation is calm/thoughtful, \
ordinary/normal, or personal opinions — use those codes. Never with any \
offensiveness code.

### Patterns of behavior
- Consistent offensive behavior: a repeated pattern — every, all, both, \
multiple, history, numerous, repeated, frequent, "standard behavior", counts \
(3/4, four out of five), "nothing but". Apply even when Majority offesnive \
is also true.
- Majority offesnive: most / majority / vast majority. Can sit next to Consistent.
- one message is enough (offensive): ONE message was itself enough to suspend. \
Not when they reviewed several messages that were all bad (that is Consistent). \
Not a general "automatic suspension" policy statement.
- One-off offensive: isolated offensive content, the rest fine, often a reason \
NOT to treat it as a pattern.
- Mix of offensive and non-offensive: they explicitly note both kinds as a mix. \
Not when they mention a few calm/thoughtful posts then still suspend for the \
offensive ones (Calm and thoughtful + the offensiveness code). Not when most \
are offensive (Majority offesnive).
- Normal behavior: ordinary / normal / typical account. "Just a normal person's \
account" is Normal behavior, not No offensive content. Can accompany No \
offensive content when they also say nothing was offensive.

### Justifications
- Contextual understanding: need more context, banter, misread, "being bothered".
- Appeal to intent: humour, venting, motive, intention — not the same as context.
- Not too offensive: they admit offense but say it does not reach the threshold.
- Poster is a bad person: they judge the PERSON ("horrible person", "angry punk"), \
not merely the posts. Do not add this just because the content is hateful.
- negative consequences: harm outside the platform (hostile environment, etc.).

### Reflection on the decision
- Expressed uncertainty: not confident, not enough info, iffy.
- Would revise the decision: they would now choose differently. Do not add it \
only because they mention looking back.
- Need to know policy and TOS: they mention policies, TOS, or community guidelines \
as something they lack or are using as the standard.
- Needed to look at more messages: they say they should have reviewed more.
- Obvious an clear suspension: rare. Only if they call the suspension decision \
itself obvious/clear/certain. "Pretty clear the content is bad" is not this code.

### Decision outcomes
- Permanent ban: taken off the platform, account termination, banned, should not \
be allowed to continue posting. "Should be suspended" is not this.
- Consequences beyond suspencion: police, arrest, therapy, watch list, mental \
health, investigated, more than suspension.
- need to contact police, or authorities.: they name police/authorities.
- Do not suspend: extremely rare. Saying they did not suspend, or "no reason to \
suspend", is usually No offensive content, not this code.
- Ban if behavior continues: suspend only if it happens again.
- Temporary suspension: they ask for a temporary / time-limited ban.

### Positive content
- Calm and thoughtful: calm, thoughtful, eloquent, genuine, supportive, \
considerate, uplifting, "nice and cool". Humans used this INSTEAD of No \
offensive content, never with it.
- Personal opinions and thoughts: they read the posts as opinions/thoughts/facts, \
not misconduct. Do not also add positive content.
- positive content: they identify the content as positive, complementary, \
grateful, good words. Can accompany No offensive content when they also say \
nothing was offensive.

Never invent codes. Never use Not offensive at all (humans used No offensive \
content). Code the justification, not the account."""


CONTRASTIVE_EXAMPLES = """\
## CONTRASTIVE EXAMPLES (wrong extra/missing codes → the human coding)

These are paraphrases of recurring mistakes. Follow the RIGHT line.

1. "It's a death threat."
   WRONG: Threats of violence
   RIGHT: Death wish

2. "Wishing death or telling someone to die should mean automatic suspension."
   WRONG: Death wish + Threats of violence
   RIGHT: Death wish

3. "Homophobic slurs and racist language against others."
   WRONG: Hate speech and offensive language
   RIGHT: Hate speech and offensive language; racial remark

4. "All three messages are highly offensive, including death threats and slurs."
   WRONG: Hate speech and offensive language; Threats of violence
   RIGHT: Consistent offensive behavior; Death wish; Extreme offensive; Hate speech and offensive language

5. "Most posts were offensive; this looks like standard behavior for the account."
   WRONG: Majority offesnive
   RIGHT: Consistent offensive behavior; Majority offesnive

6. "Looks like ordinary messages. Nothing wrong here."
   WRONG: No offensive content
   RIGHT: No offensive content; Normal behavior

7. "Just seems like a normal person's account."
   WRONG: No offensive content
   RIGHT: Normal behavior

8. "Posts seem genuine and supportive; civil, not angry."
   WRONG: No offensive content
   RIGHT: Calm and thoughtful; Contextual understanding

9. "Almost all the comments were nice and cool, so I did not suspend."
   WRONG: Do not suspend; No offensive content
   RIGHT: Calm and thoughtful

10. "There's nothing but positive content, no reason to suspend."
    WRONG: Do not suspend; positive content
    RIGHT: No offensive content

11. "Looks like someone expressing their thoughts. No reason to suspend."
    WRONG: No offensive content; Personal opinions and thoughts
    RIGHT: Personal opinions and thoughts

12. "Even though some messages are calm and eloquent, others are homophobic."
    WRONG: Hate speech and offensive language; Mix of offensive and non-offensive
    RIGHT: Calm and thoughtful; Hate speech and offensive language

13. "They had to be taken off the platform as soon as possible."
    WRONG: Hate speech and offensive language; Obvious an clear suspension
    RIGHT: Hate speech and offensive language; Permanent ban

14. "This is a sign of violence and someone may be serious about this."
    WRONG: Threats of violence
    RIGHT: Consequences beyond suspencion; Threats of violence

15. "Really rude — comments like this should not be allowed."
    WRONG: Hate speech and offensive language
    RIGHT: insult

16. "I don't know the policies for this platform, so I go by what I think is no good."
    WRONG: Hate speech and offensive language; one message is enough (offensive)
    RIGHT: Hate speech and offensive language; Need to know policy and TOS

17. "They never said anything offensive and seemed normal."
    WRONG: No offensive content; Normal behavior
    RIGHT: No offensive content

18. "The first mention of killing someone results in automatic suspension."
    WRONG: Threats of violence; one message is enough (offensive)
    RIGHT: Threats of violence"""


def render_cue_hints(text: str) -> str:
    """Tell the model which specific codes the wording may support."""
    codes = matched_codes(text)
    if not codes:
        return ""
    lines = [
        "WORDING CUES in this justification (add a listed code only if the "
        "participant actually made that point; these are candidates, not answers):"
    ]
    for code in codes:
        lines.append(f"  - {code}")
    return "\n".join(lines)
