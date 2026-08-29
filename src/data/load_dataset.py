"""
src/data/load_dataset.py

Downloads code-mixed Hindi-English (Hinglish) text classification dataset
from Hugging Face datasets hub or curated conversational Hinglish voice-agent benchmarks,
maps labels to canonical voice-agent NLU intents, and saves the raw data to data/raw/raw_dataset.csv.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_curated_voice_agent_hinglish_data() -> pd.DataFrame:
    """
    Generates a rich, realistic conversational Hinglish dataset tailored for
    voice-agent NLU pipelines (sales, support, lead qualification, scheduling),
    covering diverse phonetic transliterations, conversational noise, and dialect variations.
    """
    data = [
        # ==========================================
        # 1. complaint
        # ==========================================
        ("Mera order abhi tak deliver nahi hua hai, it's been 5 days!", "complaint"),
        ("Aapki service bohot bekar hai, refund kab milega mera?", "complaint"),
        ("Item defective nikla hai, replace karwao jaldi please.", "complaint"),
        ("Executive ne bohot rude tarike se baat kiya mere se, complain register karo.", "complaint"),
        ("Paise kat gaye account se but order confirm nahi hua abhi tak.", "complaint"),
        ("Har baar delivery delay hoti hai aapki company se, very bad experience.", "complaint"),
        ("App bar bar crash ho raha hai aur checkout fail ho raha hai.", "complaint"),
        ("Galat product bhej diya aapne, mujhe ye wala nahi chahiye tha.", "complaint"),
        ("Customer support pe koi call receive nahi kar raha, fraud company hai kya?", "complaint"),
        ("Mera subscription bina permission ke auto-renew kaise ho gaya?", "complaint"),
        ("Bill me extra charges add kar diye bina bataye, resolve it immediately.", "complaint"),
        ("Delivery boy ne package door pe fek ke chala gaya, box damaged hai.", "complaint"),
        ("Food totally cold and stale tha, hygiene bilkul zero hai.", "complaint"),
        ("Refund initiate huye 10 din ho gaye par bank me credit nahi aaya.", "complaint"),
        ("Service agent time pe nahi aaya aur call bhi cancel kar diya.", "complaint"),
        ("Product quality ekdum third class hai, return request accept karo.", "complaint"),
        ("Network connectivity issue solve nahi hua 2 weeks se.", "complaint"),
        ("Mujhe manager se baat karni hai, agent problem resolve nahi kar paa raha.", "complaint"),
        ("Account hack ho gaya lagta hai, unauthorized transactions show kar raha hai.", "complaint"),
        ("Warranty claim karne me itna issue kyu aa raha hai?", "complaint"),
        ("Booking cancel kar di par cancellation charges kyu kaate mere?", "complaint"),
        ("Ye product description se match nahi karta bilkul fake item hai.", "complaint"),
        ("Internet speed promise se half bhi nahi aa rahi hai, complaint file karo.", "complaint"),
        ("Technical support team bilkul useless hai koi response nahi deta.", "complaint"),
        ("Package open mila mujhe seal already broken thi.", "complaint"),
        ("Delivery agent ne OTP bina liye status delivered mark kar diya.", "complaint"),
        ("Payment do baar deduct ho gayi single transaction ke liye.", "complaint"),
        ("Food quantity bohot kam thi price ke hisab se, fraud hai ye.", "complaint"),
        ("Mechanic ne repair theek se nahi kiya aur machine fir se band ho gayi.", "complaint"),
        ("Mujhe compensation chahiye jo inconvenience create hui hai.", "complaint"),

        # ==========================================
        # 2. purchase_inquiry
        # ==========================================
        ("Is product ke specifications aur pricing details share kar sakte ho?", "purchase_inquiry"),
        ("Kya ye model black color me available hai stock me?", "purchase_inquiry"),
        ("Bulk purchase pe koi corporate discount provide karte ho kya?", "purchase_inquiry"),
        ("Warranty period kitne saal ka hai is electrical appliance ka?", "purchase_inquiry"),
        ("Mujhe naya laptop lena hai coding ke liye, best option suggest karo.", "purchase_inquiry"),
        ("Is real estate property ka floor plan aur brochures WhatsApp pe bhej do.", "purchase_inquiry"),
        ("Kya is plan me international roaming included hai?", "purchase_inquiry"),
        ("EMI options kaun kaun se credit card pe valid hain?", "purchase_inquiry"),
        ("Course syllabus aur duration ke baare me detail chahiye.", "purchase_inquiry"),
        ("Delivery kitne din me ho jayegi Bangalore location ke liye?", "purchase_inquiry"),
        ("Is car ka mileage aur on-road price estimate kitna padega?", "purchase_inquiry"),
        ("Online payment ke alawa Cash on Delivery ka option hai kya?", "purchase_inquiry"),
        ("Annual maintenance contract ka renewal charge kitna hota hai?", "purchase_inquiry"),
        ("Kya is package me installation and demo free of cost hai?", "purchase_inquiry"),
        ("Size chart provide kar sakte ho kya standard measurements ke saath?", "purchase_inquiry"),
        ("Is smart TV me Netflix aur Prime Video pre-installed aata hai?", "purchase_inquiry"),
        ("Kya exchange offer chal raha hai puraane phone par?", "purchase_inquiry"),
        ("Is insurance policy ke coverage details aur terms & conditions share karo.", "purchase_inquiry"),
        ("Membership subscription me kya kya perks and benefits milte hain?", "purchase_inquiry"),
        ("Next batch kab se start ho raha hai training program ka?", "purchase_inquiry"),
        ("Free trial period kitne days ka milta hai software me?", "purchase_inquiry"),
        ("Is apartment me 3BHK flats vacant hain kya abhi?", "purchase_inquiry"),
        ("Kya certificate valid hoga international job applications ke liye?", "purchase_inquiry"),
        ("Compatibility issue to nahi aayegi Mac OS ke sath?", "purchase_inquiry"),
        ("Kya aap test drive arrange karwa sakte ho weekend par?", "purchase_inquiry"),
        ("Customization options available hain kya furniture design me?", "purchase_inquiry"),
        ("Is watch me heart rate aur SpO2 tracking sensor accurate hai?", "purchase_inquiry"),
        ("Consultation fee kitni lagegi specialist doctor se milne ki?", "purchase_inquiry"),
        ("Is banquet hall ki capacity kitne guests ki hai wedding ke liye?", "purchase_inquiry"),
        ("Kya enterprise tier me 24/7 dedicated support manager milega?", "purchase_inquiry"),

        # ==========================================
        # 3. price_negotiation
        # ==========================================
        ("Thoda discount de do na, price thoda zyada lag raha hai.", "price_negotiation"),
        ("Agar main full payment cash me karu to best price kya doge?", "price_negotiation"),
        ("Competitor to yahi same service 20% cheaper rate me de raha hai.", "price_negotiation"),
        ("Budget thoda tight hai, kuch festive discount ya promo code laga do.", "price_negotiation"),
        ("Final quote kitna doge agar main 5 units ek sath buy karu?", "price_negotiation"),
        ("Pichli baar to mujhe 15% off mila tha, is baar bhi wahi price kar do.", "price_negotiation"),
        ("Installation charges wave-off kar doge to main aaj hi deal close kar lunga.", "price_negotiation"),
        ("Price negotiation ke liye manager se baat karwa sakte ho kya?", "price_negotiation"),
        ("Bhai thoda reasonable lagao, itna high price afford nahi ho payega.", "price_negotiation"),
        ("Agar 2 years ka subscription lu to kuch extra discount doge?", "price_negotiation"),
        ("Market rate se kaafi high quote kar rahe ho, thoda adjust karo.", "price_negotiation"),
        ("Shipping fee free kar do to order place karta hu turant.", "price_negotiation"),
        ("First time customer ke liye koi introductory discount voucher hai?", "price_negotiation"),
        ("Mere budget me fit nahi baith raha, 5000 tak settle kar lo.", "price_negotiation"),
        ("Without GST price kitna padega agar invoice nahi chahiye?", "price_negotiation"),
        ("Agar renewal me discount nahi mila to switch karna padega doosre provider pe.", "price_negotiation"),
        ("Annual billing me extra months complimentary milenge kya?", "price_negotiation"),
        ("Last final amount batao kitna payment karna hoga final discount ke baad.", "price_negotiation"),
        ("Kuch combo deal bana do jisme overall cost thoda reduce ho jaye.", "price_negotiation"),
        ("Student discount ya referral discount apply ho sakta hai kya ispe?", "price_negotiation"),
        ("Agar advance me pay kar du pura amount to extra off milega?", "price_negotiation"),
        ("Online price to isse sasta dikha raha hai, match karoge rate?", "price_negotiation"),
        ("Bohot expensive lag raha hai product, koi lower variant saste me hai?", "price_negotiation"),
        ("Dussehra ya Diwali sale ka coupon code apply kar do please.", "price_negotiation"),
        ("Margin thoda kam rakh lo aur deal finalize karo.", "price_negotiation"),
        ("Delivery charges bohot high hain, discount de do shipping par.", "price_negotiation"),
        ("Thoda price kam karo toh main apne 2 friends ko bhi recommend karunga.", "price_negotiation"),
        ("Aapke competitor ka quote dekhoge to aap bhi price kam kar doge.", "price_negotiation"),
        ("Kuch discount do tabhi booking advance transfer karunga.", "price_negotiation"),
        ("Best bargain price do jisme dono side agree ho sake.", "price_negotiation"),

        # ==========================================
        # 4. callback_request
        # ==========================================
        ("Abhi main drive kar raha hu, can you please call me back around 6 PM?", "callback_request"),
        ("Meeting me busy hu abhi, kal subah 10 baje call karna.", "callback_request"),
        ("Aapki voice clearly nahi aa rahi, 15 minute baad fir se phone karo.", "callback_request"),
        ("Main hospital me hu urgent, please call after 2 hours.", "callback_request"),
        ("Aapke team lead ya manager ko bolo mujhe sham ko connect kare.", "callback_request"),
        ("Abhi office me hu baat nahi ho payegi, Sunday afternoon connect karna.", "callback_request"),
        ("Signal bohot drop ho raha hai yaha, WhatsApp pe text kar do ya baad me call karo.", "callback_request"),
        ("Family dinner pe hu, kal morning me reach out karna.", "callback_request"),
        ("Main travel kar raha hu train me, Monday ko 11 AM ring back karo.", "callback_request"),
        ("Please drop a callback request for tomorrow evening.", "callback_request"),
        ("Abhi class chal rahi hai student ki, dopehar 3 baje phone lagana.", "callback_request"),
        ("Currently driving on highway, safety issue hai call back later.", "callback_request"),
        ("Main client call pe hu, call me back in 45 minutes.", "callback_request"),
        ("Aapka executive kal sham ko call kare to comfortable rahega.", "callback_request"),
        ("Network coverage issue hai call cut ho rahi hai, fir se dial karo.", "callback_request"),
        ("Flight board kar raha hu, landing ke baad call back karo around 8 PM.", "callback_request"),
        ("Busy schedule hai aaj pura din, weekend pe time milega baat karne ka.", "callback_request"),
        ("Mujhe time dekar call back arrange kar do.", "callback_request"),
        ("Important presentation start ho rahi hai, please ring after 5 PM.", "callback_request"),
        ("Doctor appointment me wait kar raha hu, 1 hour baad call karo.", "callback_request"),
        ("Abhi commute kar raha hu metro me, reaching home by 7 PM tab call karna.", "callback_request"),
        ("Can your sales representative give me a callback tomorrow at 3 PM?", "callback_request"),
        ("Bank counter pe khada hu, 20 mins baad connect karo.", "callback_request"),
        ("Baby so raha hai abhi baat nahi kar sakta, evening time me call lagana.", "callback_request"),
        ("Aap note down kar lo, kal dopehar 1 baje callback fix kar lo.", "callback_request"),
        ("Ghar pe koi guest aaye hue hain, kal call kar sakte ho kya?", "callback_request"),
        ("Court hearing me hu, can not speak right now call later.", "callback_request"),
        ("Gym me hu workout kar raha hu, 1 ghante baad phone ghumana.", "callback_request"),
        ("Abhi shift chal rahi hai factory me, night 9 PM call back karna.", "callback_request"),
        ("Can we schedule this discussion for tomorrow morning at 10:30 AM?", "callback_request"),

        # ==========================================
        # 5. not_interested
        # ==========================================
        ("Mujhe nahi chahiye koi bhi loan ya credit card, don't call me again.", "not_interested"),
        ("Not interested at all, please remove my mobile number from your database.", "not_interested"),
        ("Aap log bar bar phone karke pareshan mat karo, DND activate kar do.", "not_interested"),
        ("Mere paas already ye service hai aur main bilkul satisfied hu usse.", "not_interested"),
        ("Mujhe kisi bhi course ya training me koi interest nahi hai.", "not_interested"),
        ("Please apna promotion band karo, mujhe koi requirement nahi hai.", "not_interested"),
        ("Spam call karna band karo otherwise main consumer forum me report karunga.", "not_interested"),
        ("Nahi chahiye bhai mat call karo.", "not_interested"),
        ("I have already purchased from another brand, close my lead.", "not_interested"),
        ("Filhal koi investment plan nahi hai mera, please stop calling.", "not_interested"),
        ("Mujhe property nahi kharidni hai, wrong number pe call kiya hai.", "not_interested"),
        ("Do not disturb list me daal do mera contact.", "not_interested"),
        ("Sir bilkul interested nahi hu, time waste mat kijiye.", "not_interested"),
        ("No thanks, I am not looking for any insurance policy.", "not_interested"),
        ("Kitni baar bolu nahi chahiye aapka product!", "not_interested"),
        ("Main already doosre company ka broadband use kar raha hu mujhe switch nahi karna.", "not_interested"),
        ("Meri job change ho gayi hai to abhi mujhe iski zarurat nahi.", "not_interested"),
        ("Kindly delete my data from your telemarketing system.", "not_interested"),
        ("Nahi lena kuch bhi, disconnect the call.", "not_interested"),
        ("I am happy with my existing vendor, no interest in new proposals.", "not_interested"),
        ("Daily 10 call aate hain aapke, block list me daal raha hu number.", "not_interested"),
        ("Budget bhi nahi hai aur interest bhi nahi, thank you.", "not_interested"),
        ("Mujhe shares ya trading me koi interest nahi hai please call mat karo.", "not_interested"),
        ("Main retired person hu mere kaam ka nahi hai ye.", "not_interested"),
        ("Already bola tha last week ki not interested, fir kyu call kiya?", "not_interested"),
        ("Stop harassing with promotional offers, unsubscribe me.", "not_interested"),
        ("Mujhe koi membership nahi chahiye, thanks for calling.", "not_interested"),
        ("No need of personal loan, I have enough funds.", "not_interested"),
        ("Bilkul requirement nahi hai, goodbye.", "not_interested"),
        ("Please don't disturb during working hours, strictly not interested.", "not_interested"),

        # ==========================================
        # 6. positive_confirmation
        # ==========================================
        ("Haan bilkul theek hai, aap booking proceed kar dijiye.", "positive_confirmation"),
        ("Yes I am ready to purchase, payment link WhatsApp kar do.", "positive_confirmation"),
        ("Deal confirm hai, contract papers email kar dijiye immediately.", "positive_confirmation"),
        ("Sahi hai offer, main interested hu account activate kar do.", "positive_confirmation"),
        ("Haan delivery address correct hai, dispatch karwa do order.", "positive_confirmation"),
        ("Sounds great, please schedule my slot for tomorrow morning.", "positive_confirmation"),
        ("Sure, main online advance token amount transfer kar raha hu abhi.", "positive_confirmation"),
        ("Bilkul agree hu terms se, KYC verification complete kar lete hain.", "positive_confirmation"),
        ("Yes proceed with the subscription plan, credit card details shared.", "positive_confirmation"),
        ("Haan order pack kar do aur invoice copy bhej dena.", "positive_confirmation"),
        ("Perfect plan hai ye, sign up form ka link send karo.", "positive_confirmation"),
        ("Yes confirmed, main demo session attend karne ke liye ready hu.", "positive_confirmation"),
        ("Theek hai mujhe ye deal pasand aayi, lock kar dijiye.", "positive_confirmation"),
        ("Haan membership renew kar do same card se.", "positive_confirmation"),
        ("Deal final hai, kal subah technician ko installation ke liye bhej do.", "positive_confirmation"),
        ("Yes go ahead and book the flight tickets for me.", "positive_confirmation"),
        ("Sab details approve kar di hain maine, process aage badhao.", "positive_confirmation"),
        ("Haan bilkul main registration fee pay karne ke liye willing hu.", "positive_confirmation"),
        ("Confirmed hai sir, delivery time slot 2 PM to 4 PM rakhna.", "positive_confirmation"),
        ("Yes please send the executive for document collection at my home.", "positive_confirmation"),
        ("Haan plan finalize ho gaya hai, invoice raise kar do.", "positive_confirmation"),
        ("Deal done, payment QR code scan karke transaction successful ho gaya.", "positive_confirmation"),
        ("Agreed, main onboarding session join karne ke liye ready hu.", "positive_confirmation"),
        ("Yes everything looks good to me, proceed immediately.", "positive_confirmation"),
        ("Theek hai pack karke dispatch karwa dijiye.", "positive_confirmation"),
        ("Haanji bilkul, main offer accept karta hu.", "positive_confirmation"),
        ("Sure go ahead with the policy generation.", "positive_confirmation"),
        ("Yes that works for me, lock this pricing.", "positive_confirmation"),
        ("Aap registration submit kar do, sab documents valid hain.", "positive_confirmation"),
        ("Approved from my side, initiate the service today itself.", "positive_confirmation"),
    ]

    # Expand data synthetically with linguistic variations to make a robust benchmark
    variations = [
        # Noise additions, prefixes, polite markers, punctuation
        ("", ""),
        ("Arre ", " please"),
        ("Hey, ", "!"),
        ("Bhai ", " jaldi batao"),
        ("Sir ", " kindly confirm"),
        ("Sunna ", "..."),
        ("Dekho ", " actually"),
        ("Hello team, ", ""),
    ]

    expanded_records = []
    for text, label in data:
        for prefix, suffix in variations:
            mod_text = f"{prefix}{text}{suffix}".strip()
            expanded_records.append({"text": mod_text, "intent": label})

    df = pd.DataFrame(expanded_records)
    return df


def download_or_generate_dataset() -> pd.DataFrame:
    """
    Tries to download datasets from Hugging Face if available,
    and combines with domain-curated Hinglish intent utterances.
    """
    logger.info("Attempting to acquire Hindi-English code-mixed datasets...")
    
    # Try fetching public datasets if accessible
    hf_records = []
    try:
        from datasets import load_dataset
        logger.info("Checking Hugging Face datasets hub for Hinglish corpora...")
        # Check l3cube-pune/hinglish-sentiment or similar if available
        ds = load_dataset("l3cube-pune/hinglish-sentiment", split="train", trust_remote_code=True)
        logger.info("Successfully fetched %d rows from Hugging Face dataset", len(ds))
        
        # Map sentiment into intent proxy categories to supplement corpus
        # 0: negative -> complaint
        # 1: neutral -> purchase_inquiry
        # 2: positive -> positive_confirmation
        label_map = {0: "complaint", 1: "purchase_inquiry", 2: "positive_confirmation"}
        for item in ds:
            text = item.get("text") or item.get("tweet")
            label_id = item.get("label")
            if text and label_id in label_map:
                hf_records.append({"text": str(text), "intent": label_map[label_id]})
    except Exception as e:
        logger.warning("Could not download HF dataset directly (or network restricted): %s", e)
        logger.info("Using rich curated conversational voice-agent Hinglish corpus.")

    curated_df = generate_curated_voice_agent_hinglish_data()
    
    if hf_records:
        hf_df = pd.DataFrame(hf_records).sample(n=min(len(hf_records), 600), random_state=config.SEED)
        combined_df = pd.concat([curated_df, hf_df], ignore_index=True)
    else:
        combined_df = curated_df

    return combined_df


def main():
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_output_path = config.RAW_DATA_DIR / "raw_dataset.csv"

    df = download_or_generate_dataset()
    logger.info("Total acquired raw records: %d", len(df))
    logger.info("Class distribution in raw data:\n%s", df["intent"].value_counts())

    df.to_csv(raw_output_path, index=False, encoding="utf-8")
    logger.info("Saved raw dataset to %s", raw_output_path)


if __name__ == "__main__":
    main()
