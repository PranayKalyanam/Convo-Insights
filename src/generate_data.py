"""Generate labelled synthetic support conversations for a fictional rental /
home-services company ("RentEasy").

    python -m src.generate_data --n 300 --test-size 100 --seed 42

Writes:
  data/conversations.csv   all rows (what the pipeline processes)
  data/train.csv           rows used to train the TF-IDF baseline
  data/test.csv            held-out rows used for evaluation

Every row carries gold labels (intent, sentiment, resolution_status, priority,
qa_*) because we decide them before writing the text. That makes evaluation
possible without manual labelling, BUT template text is far more regular than
real calls, so scores here will be optimistic. See the README section
"Making the evaluation honest" for how to add a hand-labelled set.
No real customer data is used; all names, numbers and IDs are random.
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from . import config

# ------------------------------------------------------------------ building blocks
AGENTS = ["Priya", "Rahul", "Sneha", "Arjun", "Kavya", "Imran", "Neha", "Vikram"]
AREAS = ["HSR Layout", "Whitefield", "Koramangala", "Baner", "Powai", "Velachery",
         "Gachibowli", "Indiranagar", "Andheri West", "Sector 62 Noida"]

GREET = {
    "en": ["Hello, thank you for calling RentEasy support, this is {agent}. How may I help you?",
           "Good morning, welcome to RentEasy, {agent} here. What can I do for you today?"],
    "hi": ["Namaste, RentEasy support se {agent} baat kar raha hoon. Bataiye kya help karun?",
           "Hello, RentEasy mein aapka swagat hai, main {agent}. Kaise madad karun?"],
}
NO_GREET = {"en": ["Yes, tell me the issue.", "Go ahead."], "hi": ["Haan, boliye.", "Bataiye."]}

VERIFY_ASK = {
    "en": ["Can I have your registered mobile number and booking ID, please?",
           "Before I check, please confirm your registered mobile number and request ID."],
    "hi": ["Aapka registered mobile number aur booking ID bata dijiye?"],
}
VERIFY_GIVE = {
    "en": ["It's {phone}, booking ID {bid}.", "Sure, {phone}. The ID is {bid}. My email is {email}."],
    "hi": ["Number {phone} hai, booking ID {bid}.", "{phone}, aur ID {bid} hai."],
}
EMPATHY = {
    "en": ["I'm really sorry for the inconvenience, I understand how frustrating this must be.",
           "I understand, and I apologise for the trouble this has caused."],
    "hi": ["Main samajh sakta hoon, inconvenience ke liye sorry.",
           "Aapki pareshani samajh sakta hoon, maafi chahta hoon."],
}
NO_EMPATHY = {"en": ["Okay.", "Hmm, let me see."], "hi": ["Accha.", "Ok, dekhta hoon."]}

TONE = {
    "negative": {"en": ["This is the third time I am calling about this.",
                        "Honestly this is ridiculous, I am really frustrated.",
                        "I am fed up with your service, this is unacceptable."],
                 "hi": ["Yeh teesri baar call kar raha hoon, bahut pareshan hoon.",
                        "Aapki service bilkul bekaar hai, koi sunta hi nahi."]},
    "neutral": {"en": ["Can you please check this for me?",
                       "I just want to know the status and what happens next."],
                "hi": ["Please check kar dijiye.", "Bas status jaanna tha."]},
}
TONE["positive"] = TONE["neutral"]  # positive callers start neutral; the ending differs

CLOSE = {
    ("positive", "resolved"): {"en": ["Thank you so much, that was really helpful!",
                                      "Perfect, I appreciate the quick help."],
                               "hi": ["Bahut accha, shukriya!", "Thank you, aapne bahut help kiya."]},
    ("neutral", "resolved"): {"en": ["Okay, noted.", "Alright, I'll check it on my side."],
                              "hi": ["Theek hai, ok.", "Accha, main check kar leta hoon."]},
    ("neutral", "partially_resolved"): {"en": ["Okay, I'll wait for the update then."],
                                        "hi": ["Theek hai, update ka wait karunga."]},
    ("neutral", "unresolved"): {"en": ["Okay. Please let me know if anything changes."],
                                "hi": ["Accha theek hai, dekhte hain."]},
    ("negative", "partially_resolved"): {
        "en": ["Fine, but if this isn't done by then I will raise a complaint."],
        "hi": ["Theek hai, par time pe nahi hua toh complaint karunga."]},
    ("negative", "unresolved"): {
        "en": ["This is useless. I'm going to take this to the consumer forum.",
               "I'm really disappointed. Nothing has been done after so many calls."],
        "hi": ["Bilkul bekaar experience hai. Abhi tak kuch nahi hua.",
               "Bahut disappointed hoon, consumer forum mein complaint karunga."]},
}
AGENT_BYE = {"en": ["Is there anything else I can help you with? Thank you for calling RentEasy.",
                    "Thank you for your patience. Have a good day."],
             "hi": ["Aur kuch help chahiye? RentEasy call karne ke liye dhanyavaad.",
                    "Dhanyavaad, aapka din accha ho."]}
NO_OFFER = {"en": ["I'm afraid there is nothing I can do from my side right now, please wait."],
            "hi": ["Abhi mere side se kuch nahi ho sakta, aapko wait karna padega."]}

# Per intent: problem statements, and agent lines for each outcome.
INTENT_TEXT = {
    "rental_agreement": {
        "problem": {"en": ["I ordered a rental agreement {days} days ago and still haven't received the e-stamped copy.",
                           "There's a mistake in my rental agreement, the rent is written as Rs {amount}, which is wrong.",
                           "The landlord's name is spelled wrong in the agreement you sent, I need it corrected before signing."],
                    "hi": ["Maine {days} din pehle rental agreement order kiya tha, abhi tak e-stamp wali copy nahi aayi.",
                           "Agreement mein rent amount galat likha hai, correct karna hai."]},
        "resolved": {"en": ["I have corrected the agreement and sent the updated copy to your registered email just now."],
                     "hi": ["Maine agreement correct karke aapki email pe bhej diya hai, check kar lijiye."]},
        "partially_resolved": {"en": ["I've raised a correction request; the revised agreement will be delivered within 48 hours."],
                               "hi": ["Correction request daal di hai, 48 hours mein updated copy mil jayegi."]},
        "unresolved": {"en": ["I will escalate this to the documentation team, but I can't give you a timeline right now."],
                       "hi": ["Main isko documentation team ko escalate kar raha hoon, abhi timeline nahi bata sakta."]},
    },
    "owner_unresponsive": {
        "problem": {"en": ["I unlocked the owner's contact for a 2BHK in {area} but the owner has not picked up my calls for {days} days.",
                           "The owner I contacted through your app stopped replying after I said I would visit.",
                           "I've used {n} contacts from my plan and none of the owners are responding."],
                    "hi": ["Owner ka number liya tha {area} wale flat ke liye, woh call hi nahi utha raha.",
                           "{days} din se owner reply nahi kar raha, mera contact waste ho gaya."]},
        "resolved": {"en": ["I've spoken to the owner and scheduled your visit for tomorrow at 11 AM; you'll get an SMS confirmation."],
                     "hi": ["Maine owner se baat kar li hai, kal 11 baje visit confirm ho gaya hai."]},
        "partially_resolved": {"en": ["I've credited the contact back to your account and we will call the owner and update you within 24 hours."],
                               "hi": ["Contact aapke account mein wapas credit kar diya hai, owner se baat karke 24 hours mein update denge."]},
        "unresolved": {"en": ["I will raise a complaint on this listing, but we cannot force the owner to respond."],
                       "hi": ["Listing pe complaint raise kar dunga, par owner ko force nahi kar sakte."]},
    },
    "deposit_refund": {
        "problem": {"en": ["I cancelled the booking and was promised a refund of my token amount of Rs {amount}, but it's been {days} days.",
                           "The owner is refusing to return my security deposit of Rs {amount} even though I left the flat in good condition.",
                           "The refund of the advance I paid through your platform hasn't come yet."],
                    "hi": ["Token amount Rs {amount} ka refund {days} din se pending hai.",
                           "Owner mera security deposit wapas nahi de raha, flat bilkul theek condition mein chhoda tha."]},
        "resolved": {"en": ["The refund of Rs {amount} has been processed and will reflect in your account today."],
                     "hi": ["Rs {amount} ka refund process ho gaya hai, aaj account mein aa jayega."]},
        "partially_resolved": {"en": ["I've sent the refund to our finance team; it will be credited within 5 to 7 working days."],
                               "hi": ["Finance team ko refund bhej diya hai, 5-7 working days mein aa jayega."]},
        "unresolved": {"en": ["Deposit disputes between tenant and owner are outside our scope, so I can't intervene there."],
                       "hi": ["Owner aur tenant ka deposit dispute hamare scope mein nahi hai."]},
    },
    "packers_movers": {
        "problem": {"en": ["My shifting was scheduled for today at 9 AM and the movers' truck still hasn't arrived.",
                           "Your packers damaged my TV and a glass table during the move.",
                           "Two boxes are missing after my relocation from {area} last week."],
                    "hi": ["Aaj 9 baje shifting thi, packers abhi tak nahi aaye.",
                           "Shifting mein mera fridge damage ho gaya aur do boxes missing hain."]},
        "resolved": {"en": ["The truck has been reassigned and the team will reach you in 30 minutes; I have confirmed it with the driver."],
                     "hi": ["Nayi truck assign ho gayi hai, 30 minute mein team pahunch jayegi."]},
        "partially_resolved": {"en": ["I've raised a damage claim; our surveyor will visit within 48 hours to assess compensation."],
                               "hi": ["Damage claim raise ho gaya hai, surveyor 48 hours mein aayega."]},
        "unresolved": {"en": ["I will escalate this to the moving partner, but I can't promise any compensation."],
                       "hi": ["Moving partner ko escalate karta hoon, compensation ka promise nahi kar sakta."]},
    },
    "home_services": {
        "problem": {"en": ["I booked deep cleaning for today and the team never showed up.",
                           "The painter left the work half done and isn't answering calls.",
                           "The plumber fixed the leak but it started leaking again the next day."],
                    "hi": ["Aaj ke liye deep cleaning book kiya tha, koi aaya hi nahi.",
                           "Plumber ne leak theek kiya tha par phir se leak ho raha hai."]},
        "resolved": {"en": ["I've rescheduled the service and a technician is assigned for 4 PM today."],
                     "hi": ["Service reschedule kar di hai, aaj 4 baje technician aa jayega."]},
        "partially_resolved": {"en": ["I've raised a rework request; the team will call you by tomorrow to fix a slot."],
                               "hi": ["Rework request daal di hai, kal tak team call karke slot fix karegi."]},
        "unresolved": {"en": ["No technicians are available in your area this week, so I cannot book anything right now."],
                       "hi": ["Is week aapke area mein koi technician available nahi hai."]},
    },
    "subscription_plan": {
        "problem": {"en": ["I bought the Freedom plan for Rs {amount} but the relationship manager never called me.",
                           "My plan expired before I could use all the contacts; I want an extension.",
                           "I want a refund for my subscription, I found a house elsewhere."],
                    "hi": ["Rs {amount} ka plan liya tha, relationship manager ne ek baar bhi call nahi kiya.",
                           "Mera plan khatam ho gaya par contacts bache hue the, extension chahiye."]},
        "resolved": {"en": ["I've extended your plan by 30 days and assigned a new relationship manager who will call you today."],
                     "hi": ["Plan 30 din extend kar diya hai aur naya relationship manager assign kar diya hai."]},
        "partially_resolved": {"en": ["I've forwarded your request; the plans team will review it and update you within 48 hours."],
                               "hi": ["Request plans team ko bhej di hai, 48 hours mein update milega."]},
        "unresolved": {"en": ["As per policy plan amounts are non-refundable, so I can't do anything here."],
                       "hi": ["Policy ke hisaab se plan non-refundable hai, kuch nahi kar sakte."]},
    },
    "payment_issue": {
        "problem": {"en": ["I paid my rent of Rs {amount} by credit card yesterday but the owner says it's not credited.",
                           "My UPI payment failed but the money was deducted from my account.",
                           "I was charged twice for the same rent payment transaction."],
                    "hi": ["Kal credit card se Rs {amount} rent pay kiya, owner bol raha hai paisa nahi aaya.",
                           "UPI payment fail ho gaya par paisa kat gaya."]},
        "resolved": {"en": ["I checked with the gateway; the amount was credited to the owner this morning and I've sent the UTR to your email."],
                     "hi": ["Gateway se check kiya, amount owner ko credit ho gaya hai, UTR email pe bhej diya."]},
        "partially_resolved": {"en": ["The duplicate charge is confirmed; the reversal will reach your account within 5 to 7 working days."],
                               "hi": ["Double charge confirm hai, 5-7 working days mein paisa wapas aa jayega."]},
        "unresolved": {"en": ["The transaction is pending with the bank; I cannot do anything until the bank updates it."],
                       "hi": ["Bank side pe transaction pending hai, jab tak bank update nahi karta kuch nahi ho sakta."]},
    },
    "property_listing": {
        "problem": {"en": ["I posted my property {days} days ago but the listing is still not live.",
                           "I want to update the photos and rent on my listing but the app shows an error.",
                           "My flat is rented out, please deactivate the listing; I'm still getting calls."],
                    "hi": ["Maine {days} din pehle property post ki thi, listing abhi tak live nahi hui.",
                           "Flat rent pe chala gaya hai, listing band kar do, calls aa rahi hain."]},
        "resolved": {"en": ["I've fixed the listing and it is updated with your details now."],
                     "hi": ["Listing update kar di hai, ab sab theek hai."]},
        "partially_resolved": {"en": ["Your listing is in the verification queue; our team will complete the photo check within 24 hours."],
                               "hi": ["Listing verification mein hai, 24 hours mein complete ho jayegi."]},
        "unresolved": {"en": ["There's a technical issue on our side; I will raise a ticket but I don't have a timeline."],
                       "hi": ["Technical issue hai, ticket raise karunga par timeline nahi bata sakta."]},
    },
    "other": {
        "problem": {"en": ["Are your services available in Lucknow? I'm moving there next month.",
                           "Someone called me claiming to be from your company and asked for my OTP. Is that genuine?"],
                    "hi": ["Kya aapki service Jaipur mein available hai?",
                           "Kisi ne aapki company ke naam se call karke OTP maanga, kya yeh sahi hai?"]},
        "resolved": {"en": ["Yes, we're live there, and please note we never ask for OTPs, so don't share it with anyone."],
                     "hi": ["Haan, hum wahan available hain, aur hum kabhi OTP nahi maangte."]},
        "partially_resolved": {"en": ["I've noted it and our team will email you the details within 24 hours."],
                               "hi": ["Note kar liya hai, 24 hours mein email pe details aa jayengi."]},
        "unresolved": {"en": ["I don't have information on that, sorry."],
                       "hi": ["Iski jaankari mere paas nahi hai."]},
    },
}

# (sentiment, resolution) mix, roughly what a support queue might look like
OUTCOMES = [(("positive", "resolved"), 0.25), (("neutral", "resolved"), 0.15),
            (("neutral", "partially_resolved"), 0.20), (("negative", "partially_resolved"), 0.15),
            (("negative", "unresolved"), 0.20), (("neutral", "unresolved"), 0.05)]
INTENT_WEIGHTS = {k: (0.4 if k == "other" else 1.0) for k in config.INTENTS}


def gold_priority(sentiment: str, resolution: str) -> str:
    """Same rule the LLM prompt describes."""
    if sentiment == "negative" and resolution != "resolved":
        return "high"
    if resolution == "resolved" and sentiment != "negative":
        return "low"
    return "medium"


def _phone(rng: random.Random) -> str:
    return f"{rng.choice('6789')}{rng.randint(100000000, 999999999)}"


def _merge_turns(lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Join consecutive lines by the same speaker into one turn."""
    merged: list[tuple[str, str]] = []
    for who, text in lines:
        if merged and merged[-1][0] == who:
            merged[-1] = (who, merged[-1][1] + " " + text)
        else:
            merged.append((who, text))
    return merged


def make_conversation(rng: random.Random, idx: int) -> dict:
    intent = rng.choices(list(INTENT_WEIGHTS), weights=list(INTENT_WEIGHTS.values()))[0]
    (sentiment, resolution) = rng.choices([o for o, _ in OUTCOMES], weights=[w for _, w in OUTCOMES])[0]
    lang = "hi" if rng.random() < 0.3 else "en"
    t = INTENT_TEXT[intent]

    qa = {
        "greeted": rng.random() < 0.8,
        "verified_details": rng.random() < 0.65,
        "showed_empathy": rng.random() < (0.75 if sentiment == "negative" else 0.45),
        # A resolved or partly resolved call always involves a concrete step.
        "offered_resolution": True if resolution != "unresolved" else rng.random() < 0.6,
    }
    fill = dict(agent=rng.choice(AGENTS), area=rng.choice(AREAS), days=rng.randint(2, 15),
                amount=rng.choice([2500, 5000, 8000, 12000, 18500, 25000, 40000]),
                n=rng.randint(3, 10), phone=_phone(rng), bid=f"RE{rng.randint(100000, 999999)}",
                email=f"user{rng.randint(10, 999)}@gmail.com")
    say = lambda opts: rng.choice(opts[lang]).format(**fill)  # noqa: E731

    lines = [("Agent", say(GREET) if qa["greeted"] else say(NO_GREET)),
             ("Customer", say(t["problem"]))]
    if qa["verified_details"]:
        lines += [("Agent", say(VERIFY_ASK)), ("Customer", say(VERIFY_GIVE))]
    lines.append(("Customer", say(TONE[sentiment])))
    lines.append(("Agent", say(EMPATHY) if qa["showed_empathy"] else say(NO_EMPATHY)))
    if resolution == "unresolved" and not qa["offered_resolution"]:
        lines.append(("Agent", say(NO_OFFER)))
    else:
        lines.append(("Agent", say(t[resolution])))
    lines.append(("Customer", say(CLOSE[(sentiment, resolution)])))
    lines.append(("Agent", say(AGENT_BYE)))

    return {
        "conversation_id": f"conv_{idx:04d}",
        "language": "hinglish" if lang == "hi" else "english",
        "transcript": "\n".join(f"{who}: {text}" for who, text in _merge_turns(lines)),
        "intent": intent,
        "sentiment": sentiment,
        "resolution_status": resolution,
        "priority": gold_priority(sentiment, resolution),
        **{f"qa_{k}": int(v) for k, v in qa.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--test-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    df = pd.DataFrame([make_conversation(rng, i) for i in range(1, a.n + 1)])
    test = df.sample(n=min(a.test_size, len(df) // 2), random_state=a.seed)
    train = df.drop(test.index)

    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.DATA_DIR / "conversations.csv", index=False)
    train.to_csv(config.DATA_DIR / "train.csv", index=False)
    test.to_csv(config.DATA_DIR / "test.csv", index=False)
    print(f"Wrote {len(df)} conversations ({len(train)} train / {len(test)} test) to {config.DATA_DIR}")
    print(df["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
