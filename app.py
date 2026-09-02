from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

app = Flask(__name__, template_folder='templates')

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "plant_disease_model_mobilenet.h5"
CLASS_JSON_PATH = "class_indices.json"
IMG_SIZE = (128, 128)

# -----------------------------
# Load model (FIXED 🔥)
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# Load class mapping
# -----------------------------
with open(CLASS_JSON_PATH, "r") as f:
    class_map = json.load(f)

index_to_class = {v: k for k, v in class_map.items()}

# -----------------------------
# Minimal disease info (you can keep your full dict)
# -----------------------------
def default_info_for(class_key):
    return {
        "name": {
            "en": class_key.replace("_", " "),
            "ta": class_key.replace("_", " "),
            "hi": class_key.replace("_", " ")
        },
        "description": {
            "en": "No additional info available.",
            "ta": "மேலும் தகவல் இல்லை.",
            "hi": "अतिरिक्त जानकारी उपलब्ध नहीं है।"
        },
        "treatment": {
            "en": "Consult local agricultural expert.",
            "ta": "உங்கள் விவசாய நிபுணரை அணுகவும்.",
            "hi": "स्थानीय कृषि विशेषज्ञ से संपर्क करें।"
        },
        "spot_location": {
            "en": "Not available",
            "ta": "கிடைக்கவில்லை",
            "hi": "उपलब्ध नहीं"
        }
    }


# -----------------------------
# Multilingual disease information (EN / TA / HI)
# Keys must match your class names in class_indices.json
# -----------------------------
disease_info = {
    "Apple___Apple_scab": {
        "name": {"en": "Apple Scab", "ta": "ஆப்பிள் ஸ்காப்", "hi": "एप्पल स्कैब"},
        "description": {
            "en": "Fungal disease causing olive-black scabby spots on leaves and fruit.",
            "ta": "இலைகளிலும் பழங்களிலும் கரும்புள்ளிகளாகத் தோன்றும் பூஞ்சைப் பாதிப்பு.",
            "hi": "पत्तियों और फलों पर काले, खुरदुरे धब्बे पैदा करने वाला कवकजन्य रोग।"
        },
        "treatment": {
            "en": "Prune infected parts and use fungicides such as Captan or Mancozeb.",
            "ta": "பாதிக்கப்பட்ட பகுதிகளை வெட்டவும்; Captan அல்லது Mancozeb போன்ற பூஞ்சைக் கொல்லிகளை பயன்படுத்தவும்.",
            "hi": "संक्रमित हिस्सों को हटा कर Captan या Mancozeb जैसे फफूंदनाशकों का प्रयोग करें।"
        },
        "spot_location": {
            "en": "Dark, round spots typically on the upper leaf surface and fruit skin.",
            "ta": "இலை மேற்பரப்பிலும் பழ பலகையும் பொதுவாக கருப்புப் புள்ளிகள் காணப்படுகின்றன.",
            "hi": "सामान्यतः पत्तियों की ऊपरी सतह और फल की त्वचा पर काले गोल धब्बे दिखते हैं।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy all infected leaves and fruit from the tree and ground",
                "Apply fungicide sprays in early spring before bud break",
                "Use resistant apple varieties like 'Liberty' or 'Freedom'",
                "Prune trees to improve air circulation and sunlight penetration",
                "Clean up fallen leaves and debris in autumn to reduce overwintering spores",
                "Avoid overhead irrigation to minimize leaf wetness duration",
                "Apply dormant sprays like copper or lime sulfur before bud swell",
                "Monitor trees weekly during growing season for early detection",
                "Use organic options like sulfur or potassium bicarbonate sprays",
                "Consult local extension for specific fungicide timing in your area"
            ],
            "ta": [
                "மரத்திலும் தரையிலும் உள்ள அனைத்து பாதிக்கப்பட்ட இலைகளையும் பழங்களையும் அகற்றி அழிக்கவும்",
                "மொக்கு உடைப்பதற்கு முன் வசந்த காலத்தின் தொடக்கத்தில் பூஞ்சைக் கொல்லி தெளிப்புகளைப் பயன்படுத்தவும்",
                "'லிபர்டி' அல்லது 'ஃப்ரீடம்' போன்ற எதிர்ப்பு ஆப்பிள் வகைகளைப் பயன்படுத்தவும்",
                "காற்றோட்டம் மற்றும் சூரிய ஒளி ஊடுருவலை மேம்படுத்த மரங்களை கத்தரிக்கவும்",
                "குளிர்காலத்தில் தங்கும் வித்துக்களைக் குறைக்க இலையுதிர் காலத்தில் விழுந்த இலைகள் மற்றும் குப்பைகளை சுத்தம் செய்யவும்",
                "இலை ஈரப்பதத்தின் காலத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "மொக்கு வீக்கத்திற்கு முன் தாமிரம் அல்லது சுண்ணாம்பு சல்பர் போன்ற உறக்க தெளிப்புகளைப் பயன்படுத்தவும்",
                "ஆரம்பகால கண்டறிதலுக்கு வளரும் பருவத்தில் வாராந்திர மரங்களை கண்காணிக்கவும்",
                "கந்தகம் அல்லது பொட்டாசியம் பைகார்பனேட் தெளிப்புகள் போன்ற கரிம விருப்பங்களைப் பயன்படுத்தவும்",
                "உங்கள் பகுதிக்கு குறிப்பிட்ட பூஞ்சைக் கொல்லி நேரத்திற்கு உள்ளூர் நீட்டிப்பை அணுகவும்"
            ],
            "hi": [
                "पेड़ और जमीन से सभी संक्रमित पत्तियों और फलों को हटाकर नष्ट करें",
                "कली फूटने से पहले शुरुआती वसंत में फफूंदनाशक स्प्रे लगाएं",
                "'लिबर्टी' या 'फ्रीडम' जैसी प्रतिरोधी सेब किस्मों का उपयोग करें",
                "हवा के संचार और सूर्य के प्रकाश को बेहतर बनाने के लिए पेड़ों की छंटाई करें",
                "सर्दियों में बीजाणुओं को कम करने के लिए पतझड़ में गिरी पत्तियों और मलबे की सफाई करें",
                "पत्तियों की नमी की अवधि कम करने के लिए ऊपर से सिंचाई से बचें",
                "कली फूलने से पहले कॉपर या लाइम सल्फर जैसे निष्क्रिय स्प्रे लगाएं",
                "शुरुआती पहचान के लिए बढ़ते मौसम के दौरान साप्ताहिक पेड़ों की निगरानी करें",
                "सल्फर या पोटेशियम बाइकार्बोनेट स्प्रे जैसे जैविक विकल्पों का उपयोग करें",
                "अपने क्षेत्र में विशिष्ट फफूंदनाशक समय के लिए स्थानीय विस्तार सेवा से परामर्श लें"
            ]
        }
    },

    "Apple___Black_rot": {
        "name": {"en": "Apple Black Rot", "ta": "ஆப்பிள் பிளாக் ராட்", "hi": "एप्पल ब्लैक रॉट"},
        "description": {
            "en": "Fungal disease causing brown-to-black circular lesions on fruit and leaves.",
            "ta": "பழங்கள் மற்றும் இலைகளில் கருப்பு வட்டத் திடல்கள் ஏற்படுத்தும் பூஞ்சை நோய்.",
            "hi": "फल और पत्तियों पर भूरा-से-काला वृताकार घाव पैदा करने वाला फफूंदी रोग।"
        },
        "treatment": {
            "en": "Remove infected fruit; apply copper-based fungicides; maintain sanitation.",
            "ta": "பாதிக்கப்பட்ட பழங்களை அகற்று; காப்பர் அடிப்படையிலான பூஞ்சிக் கொல்லிகளைப் பாவிக்கவும்.",
            "hi": "संक्रमित फल हटाएँ; कॉपर-आधारित फफूंदनाशक लगाएँ; स्वच्छ रखें।"
        },
        "spot_location": {
            "en": "Circular lesions on fruit and leaf centers; often with concentric rings.",
            "ta": "பழ மற்றும் இலை மத்திய பகுதிகளில் வட்ட சிதலைகள், சில நேரங்களில் வளியுரு வளையங்கள் காணப்படும்.",
            "hi": "फल और पत्तियों के केन्द्र पर गोल घाव, अक्सर समकेंद्र वर्तुल दिखते हैं।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy all mummified fruit from trees and ground",
                "Prune out dead or diseased branches and cankers during dormancy",
                "Apply copper-based fungicides during dormancy and early season",
                "Ensure proper tree spacing for good air circulation",
                "Remove wild apple and crabapple trees near orchard",
                "Apply fungicides at petal fall and first cover spray timing",
                "Avoid mechanical injuries to fruit that provide entry points",
                "Use resistant varieties when establishing new plantings",
                "Maintain balanced nutrition without excess nitrogen",
                "Monitor regularly during fruit development stage"
            ],
            "ta": [
                "மரங்கள் மற்றும் தரையில் இருந்து அனைத்து மம்மிஃபைடு பழங்களையும் அகற்றி அழிக்கவும்",
                "உறக்க நிலையில் இறந்த அல்லது நோய்வாய்ப்பட்ட கிளைகள் மற்றும் புண்களை கத்தரிக்கவும்",
                "உறக்க நிலை மற்றும் ஆரம்ப பருவத்தில் தாமிரம் அடிப்படையிலான பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "நல்ல காற்றோட்டத்திற்கு சரியான மர இடைவெளியை உறுதிப்படுத்தவும்",
                "பழத் தோட்டத்திற்கு அருகில் உள்ள காட்டு ஆப்பிள் மற்றும் கிராப்ஆப்பிள் மரங்களை அகற்றவும்",
                "இதழ் விழுதல் மற்றும் முதல் கவர் தெளிப்பு நேரத்தில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "நுழைவு புள்ளிகளை வழங்கும் பழங்களின் இயந்திர காயங்களைத் தவிர்க்கவும்",
                "புதிய நடவு செய்யும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "அதிக நைட்ரஜன் இல்லாமல் சீரான ஊட்டச்சத்தை பராமரிக்கவும்",
                "பழ வளர்ச்சி நிலையில் தவறாமல் கண்காணிக்கவும்"
            ],
            "hi": [
                "पेड़ों और जमीन से सभी ममीकृत फलों को हटाकर नष्ट करें",
                "निष्क्रिय अवस्था के दौरान मृत या रोगग्रस्त शाखाओं और कैंकर को काटें",
                "निष्क्रिय अवस्था और शुरुआती मौसम के दौरान कॉपर-आधारित फफूंदनाशक लगाएं",
                "अच्छे वायु संचार के लिए उचित पेड़ रिक्ति सुनिश्चित करें",
                "बाग के पास जंगली सेब और क्रैबएप्पल पेड़ों को हटाएं",
                "पंखुड़ी गिरने और पहले कवर स्प्रे के समय फफूंदनाशक लगाएं",
                "फलों को यांत्रिक चोटों से बचाएं जो प्रवेश बिंदु प्रदान करते हैं",
                "नए पौधे लगाते समय प्रतिरोधी किस्मों का उपयोग करें",
                "अतिरिक्त नाइट्रोजन के बिना संतुलित पोषण बनाए रखें",
                "फल विकास अवस्था के दौरान नियमित रूप से निगरानी करें"
            ]
        }
    },

    "Apple___Cedar_apple_rust": {
        "name": {"en": "Cedar-Apple Rust", "ta": "சீடார்-ஆப்பிள் ரஸ்ட்", "hi": "सीडर-एप्पल रस्ट"},
        "description": {
            "en": "A fungal rust disease causing orange gelatinous spots on leaves.",
            "ta": "இலைகளில் ஆரஞ்சு கல்சியம் போன்ற புண்ணகங்கள் ஏற்படுத்தும் பூஞ்சை நோய்.",
            "hi": "पत्तियों पर नारंगी जेलीनस धब्बे पैदा करने वाला कवकजन्य रोग।"
        },
        "treatment": {
            "en": "Remove nearby junipers/cedars if possible; apply fungicide sprays early in season.",
            "ta": "இருக்கக் கூடிய சீடர் மரங்களை நீக்கவும்; பருவத்தின் துவக்கத்தில் பூஞ்சிக் கொல்லிகளைக் குறைச்சல்.",
            "hi": "यदि संभव हो तो पास के सीडर/जूनिपर हटाएँ; मौसम की शुरुआत में फफूंदनाशक छिड़काव करें।"
        },
        "spot_location": {
            "en": "Orange/yellow spots on upper leaf surface; gelatinous spore horns on underside in wet weather.",
            "ta": "மேற்பகுதியில் ஆரஞ்சு/மஞ்சள் நுணுக்கங்கள்; ஈரமான காலங்களில் கீழ்புறத்தில் ஜெலிலைப் போன்று தோன்றும்.",
            "hi": "ऊपरी सतह पर नारंगी/पीले धब्बे; गीले मौसम में नीचे दूधिया स्पोर होर्न दिखाई देते हैं।"
        },
        "treatment_steps": {
            "en": [
                "Remove eastern red cedar trees within 2 miles of apple orchard if possible",
                "Apply protective fungicides before infection periods in spring",
                "Use fungicides with myclobutanil or triadimefon for best control",
                "Time sprays based on cedar galls becoming orange and gelatinous",
                "Apply first spray at pink bud stage and repeat every 10-14 days",
                "Plant resistant apple varieties like 'William's Pride' or 'Freedom'",
                "Remove galls from cedar trees in winter to reduce spore source",
                "Improve air circulation through proper pruning",
                "Avoid planting apples near natural cedar areas",
                "Monitor weather conditions for infection risk periods"
            ],
            "ta": [
                "முடிந்தால் ஆப்பிள் தோட்டத்திற்கு 2 மைல் தொலைவில் கிழக்கு சிவப்பு சீடர் மரங்களை அகற்றவும்",
                "வசந்த காலத்தில் தொற்று காலங்களுக்கு முன் பாதுகாப்பு பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "சிறந்த கட்டுப்பாட்டிற்கு மைக்ளோபுடானில் அல்லது டிரையாடிமெஃபான் கொண்ட பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "சீடர் பித்தங்கள் ஆரஞ்சு மற்றும் ஜெலட்டினாக மாறுவதை அடிப்படையாகக் கொண்டு தெளிப்பு நேரம்",
                "பிங்க் பட் நிலையில் முதல் தெளிப்பைப் பயன்படுத்தவும் மற்றும் ஒவ்வொரு 10-14 நாட்களுக்கும் மீண்டும் செய்யவும்",
                "'வில்லியம் பிரைட்' அல்லது 'ஃப்ரீடம்' போன்ற எதிர்ப்பு ஆப்பிள் வகைகளை நடவு செய்யவும்",
                "வித்து மூலத்தைக் குறைக்க குளிர்காலத்தில் சீடர் மரங்களில் இருந்து பித்தங்களை அகற்றவும்",
                "சரியான கத்தரித்தல் மூலம் காற்றோட்டத்தை மேம்படுத்தவும்",
                "இயற்கை சீடர் பகுதிகளுக்கு அருகில் ஆப்பிள்களை நடவு செய்வதைத் தவிர்க்கவும்",
                "தொற்று ஆபத்து காலங்களுக்கான வானிலை நிலைமைகளை கண்காணிக்கவும்"
            ],
            "hi": [
                "यदि संभव हो तो सेब के बाग के 2 मील के भीतर पूर्वी लाल देवदार के पेड़ हटाएं",
                "वसंत ऋतु में संक्रमण की अवधि से पहले सुरक्षात्मक फफूंदनाशक लगाएं",
                "सर्वोत्तम नियंत्रण के लिए माइक्लोब्यूटानिल या ट्रायडिमेफोन वाले फफूंदनाशकों का उपयोग करें",
                "देवदार के गॉल के नारंगी और जेलाटिनस होने के आधार पर स्प्रे का समय निर्धारित करें",
                "गुलाबी कली अवस्था में पहला स्प्रे लगाएं और हर 10-14 दिनों में दोहराएं",
                "'विलियम्स प्राइड' या 'फ्रीडम' जैसी प्रतिरोधी सेब किस्में लगाएं",
                "बीजाणु स्रोत को कम करने के लिए सर्दियों में देवदार के पेड़ों से गॉल हटाएं",
                "उचित छंटाई के माध्यम से वायु संचार में सुधार करें",
                "प्राकृतिक देवदार क्षेत्रों के पास सेब न लगाएं",
                "संक्रमण जोखिम अवधि के लिए मौसम की स्थिति की निगरानी करें"
            ]
        }
    },

    "Apple___healthy": {
        "name": {"en": "Healthy Apple", "ta": "ஆப்பிள் - ஆரோக்கியம்", "hi": "स्वस्थ सेब"},
        "description": {
            "en": "No visible disease; foliage and fruit appear healthy.",
            "ta": "காணக்கூடிய நோய் இல்லை; இலைகளும் பழமும் ஆரோக்கியமாய் தோன்றுகிறது.",
            "hi": "कोई दिखाई देने योग्य रोग नहीं; पत्तियाँ और फल स्वस्थ दिखते हैं।"
        },
        "treatment": {
            "en": "Maintain good cultural practices: proper irrigation, pruning, and monitoring.",
            "ta": "சரி கனவு செயற்பாடுகள்: நீர்த்தேக்க சரி, கிளைக் குறைப்பு, கண்காணிப்பு ஆகியவற்றை தொடரவும்.",
            "hi": "सही सिंचाई, छंटाई और निगरानी जैसी अच्छा कृषि अभ्यास जारी रखें।"
        },
        "spot_location": {
            "en": "No spots.",
            "ta": "புள்ளிகள் இல்லை.",
            "hi": "कोई धब्बे नहीं।"
        },
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for early disease detection",
                "Maintain proper irrigation schedule based on weather conditions",
                "Prune annually to maintain good air circulation and light penetration",
                "Apply balanced fertilizer according to soil test recommendations",
                "Monitor for pests and take action only when necessary",
                "Keep orchard floor clean of debris and fallen fruit",
                "Use mulch to conserve moisture and suppress weeds",
                "Practice crop rotation if growing other crops in orchard",
                "Maintain proper tree spacing for optimal growth",
                "Keep records of tree health and any treatments applied"
            ],
            "ta": [
                "ஆரம்பகால நோய் கண்டறிதலுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "வானிலை நிலைமைகளின் அடிப்படையில் சரியான நீர்ப்பாசன அட்டவணையை பராமரிக்கவும்",
                "நல்ல காற்றோட்டம் மற்றும் ஒளி ஊடுருவலை பராமரிக்க வருடாந்திர கத்தரித்தல்",
                "மண் சோதனை பரிந்துரைகளின்படி சீரான உரத்தைப் பயன்படுத்தவும்",
                "பூச்சிகளை கண்காணித்து, தேவையான时候 மட்டுமே நடவடிக்கை எடுக்கவும்",
                "குப்பை மற்றும் விழுந்த பழங்களிலிருந்து தோட்டத்தின் தரையை சுத்தமாக வைத்திருங்கள்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் களைகளை அடக்க மல்ச் பயன்படுத்தவும்",
                "தோட்டத்தில் பிற பயிர்களை வளர்த்தால் பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "உகந்த வளர்ச்சிக்கு சரியான மர இடைவெளியை பராமரிக்கவும்",
                "மர ஆரோக்கியம் மற்றும் பயன்படுத்தப்படும் எந்த சிகிச்சைகளின் பதிவுகளை வைத்திருங்கள்"
            ],
            "hi": [
                "शुरुआती रोग का पता लगाने के लिए नियमित निगरानी जारी रखें",
                "मौसम की स्थिति के आधार पर उचित सिंचाई कार्यक्रम बनाए रखें",
                "अच्छे वायु संचार और प्रकाश प्रवेश को बनाए रखने के लिए सालाना छंटाई करें",
                "मृदा परीक्षण की सिफारिशों के अनुसार संतुलित उर्वरक लगाएं",
                "कीटों की निगरानी करें और केवल आवश्यकता पड़ने पर कार्रवाई करें",
                "बाग के फर्श को मलबे और गिरे हुए फलों से साफ रखें",
                "नमी संरक्षण और खरपतवार दमन के लिए मल्च का उपयोग करें",
                "यदि बाग में अन्य फसलें उगा रहे हैं तो फसल चक्र अपनाएं",
                "इष्टतम विकास के लिए उचित पेड़ रिक्ति बनाए रखें",
                "पेड़ के स्वास्थ्य और किए गए किसी भी उपचार का रिकॉर्ड रखें"
            ]
        }
    },

    "Blueberry___healthy": {
        "name": {"en": "Healthy Blueberry", "ta": "நலம் - புளூபெர்ரி", "hi": "स्वस्थ ब्लूबेरी"},
        "description": {
            "en": "No signs of disease on leaves or fruit.",
            "ta": "இலையிலும் பழத்திலும் நோய் அறிகுறிகள் இல்லை.",
            "hi": "पत्तियों या फलों पर रोग के कोई लक्षण नहीं।"
        },
        "treatment": {
            "en": "Continue good nutrition and irrigation practices.",
            "ta": "நல்ல ஊட்டச்சத்து மற்றும் நீர்த்தேக்கத்தை பராமரிக்கவும்.",
            "hi": "अच्छी पोषण और सिंचाई बनाए रखें।"
        },
        "spot_location": {"en": "No spots.", "ta": "புள்ளிகள் இல்லை.", "hi": "कोई धब्बे नहीं।"},
        "treatment_steps": {
            "en": [
                "Maintain soil pH between 4.5-5.5 for optimal blueberry growth",
                "Apply acid-loving plant fertilizer in early spring",
                "Use organic mulch like pine needles or wood chips",
                "Ensure consistent moisture, especially during fruit development",
                "Prune annually to remove dead wood and encourage new growth",
                "Monitor for common blueberry pests like spotted wing drosophila",
                "Test soil every 2-3 years and amend as needed",
                "Provide adequate spacing between plants for air circulation",
                "Use bird netting if birds are damaging fruit",
                "Remove any diseased plant material immediately if noticed"
            ],
            "ta": [
                "உகந்த புளூபெர்ரி வளர்ச்சிக்கு மண் pH 4.5-5.5 க்கு இடையில் பராமரிக்கவும்",
                "வசந்த காலத்தின் தொடக்கத்தில் அமில-விரும்பும் தாவர உரத்தைப் பயன்படுத்தவும்",
                "பைன் ஊசிகள் அல்லது மரத் துண்டுகள் போன்ற கரிம மல்ச் பயன்படுத்தவும்",
                "பழ வளர்ச்சிய期间 குறிப்பாக சீரான ஈரப்பதத்தை உறுதிப்படுத்தவும்",
                "இறந்த மரத்தை அகற்றவும் மற்றும் புதிய வளர்ச்சியை ஊக்குவிக்க வருடாந்திர கத்தரிக்கவும்",
                "ஸ்பாட்டட் விங் டிரோசோபிலா போன்ற பொதுவான புளூபெர்ரி பூச்சிகளை கண்காணிக்கவும்",
                "ஒவ்வொரு 2-3 ஆண்டுகளுக்கும் மண்ணை சோதித்து தேவைக்கேற்ப திருத்தவும்",
                "காற்றோட்டத்திற்கு தாவரங்களுக்கு இடையே போதுமான இடைவெளியை வழங்கவும்",
                "பறவைகள் பழங்களை சேதப்படுத்தினால் பறவை வலை பயன்படுத்தவும்",
                "கவனிக்கப்பட்டால் எந்த நோய்வாய்ப்பட்ட தாவர பொருட்களையும் உடனடியாக அகற்றவும்"
            ],
            "hi": [
                "इष्टतम ब्लूबेरी विकास के लिए मिट्टी का pH 4.5-5.5 के बीच बनाए रखें",
                "शुरुआती वसंत में अम्ल-प्रेमी पौधों की खाद डालें",
                "पाइन सुइयों या लकड़ी के चिप्स जैसी जैविक मल्च का उपयोग करें",
                "फल विकास के दौरान विशेष रूप से लगातार नमी सुनिश्चित करें",
                "मृत लकड़ी को हटाने और नई वृद्धि को प्रोत्साहित करने के लिए सालाना छंटाई करें",
                "स्पॉटेड विंग ड्रोसोफिला जैसे सामान्य ब्लूबेरी कीटों की निगरानी करें",
                "हर 2-3 साल में मिट्टी का परीक्षण करें और आवश्यकतानुसार संशोधन करें",
                "वायु संचार के लिए पौधों के बीच पर्याप्त रिक्ति प्रदान करें",
                "यदि पक्षी फलों को नुकसान पहुंचा रहे हैं तो बर्ड नेटिंग का उपयोग करें",
                "यदि देखा जाए तो किसी भी रोगग्रस्त पौधे की सामग्री को तुरंत हटा दें"
            ]
        }
    },

    "Cherry_(including_sour)___Powdery_mildew": {
        "name": {"en": "Cherry Powdery Mildew", "ta": "செரி தூசி காளான்", "hi": "चेरी पाउडरी मिल्ड्यू"},
        "description": {
            "en": "White powdery fungal growth on leaf surfaces and shoots.",
            "ta": "இலையை மற்றும் கொம்புகளை வெள்ளை தூசி போன்று மூடிய பூஞ்சை வளர்ச்சி.",
            "hi": "पत्तियों और शाखाओं की सतह पर सफेद पाउडरी कवक विकास।"
        },
        "treatment": {
            "en": "Apply sulfur or potassium bicarbonate sprays; remove heavily infected shoots.",
            "ta": "சல்பர் அல்லது பொட்டாசியம் பை கார்பொனேட் தெளிக்கவும்; அதிகமாக பாதிக்கப்பட்ட கொம்புகளை அகற்று.",
            "hi": "सल्फर या पोटैशियम बाइकार्बोनेट छिड़काव करें; गंभीर रूप से संक्रमित क्लिप हटाएँ।"
        },
        "spot_location": {
            "en": "White powder on upper leaf surface and young shoots.",
            "ta": "மேல்தர இலை மற்றும் இளம் கொம்புகளில் வெள்ளை தூசி.",
            "hi": "ऊपरी पत्ती सतह और युवा शाखाओं पर सफेद पाउडरी।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy severely infected shoots and leaves",
                "Apply sulfur-based fungicides early in the season",
                "Use horticultural oils or potassium bicarbonate sprays",
                "Improve air circulation by proper pruning and spacing",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Apply fungicides before symptoms appear in high-risk areas",
                "Use resistant cherry varieties when planting new trees",
                "Remove suckers and water sprouts that shade the canopy",
                "Apply fungicides at 7-14 day intervals during favorable conditions",
                "Monitor new growth carefully as it's most susceptible"
            ],
            "ta": [
                "கடுமையாக பாதிக்கப்பட்ட கிளைகள் மற்றும் இலைகளை அகற்றி அழிக்கவும்",
                "பருவத்தின் ஆரம்பத்தில் கந்தக அடிப்படையிலான பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "தோட்டக்கலை எண்ணெய்கள் அல்லது பொட்டாசியம் பைகார்பனேட் தெளிப்புகளைப் பயன்படுத்தவும்",
                "சரியான கத்தரித்தல் மற்றும் இடைவெளி மூலம் காற்றோட்டத்தை மேம்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "அதிக ஆபத்துள்ள பகுதிகளில் அறிகுறிகள் தோன்றுவதற்கு முன் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "புதிய மரங்களை நடும் போது எதிர்ப்பு செர்ரி வகைகளைப் பயன்படுத்தவும்",
                "கனோபியை நிழலாக்கும் சக்கர்கள் மற்றும் நீர் முளைகளை அகற்றவும்",
                "சாதகமான நிலைமைகளில் 7-14 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "மிகவும் பாதிக்கப்படக்கூடியதாக இருப்பதால் புதிய வளர்ச்சியை கவனமாக கண்காணிக்கவும்"
            ],
            "hi": [
                "गंभीर रूप से संक्रमित शाखाओं और पत्तियों को हटाकर नष्ट करें",
                "मौसम की शुरुआत में सल्फर-आधारित फफूंदनाशक लगाएं",
                "बागवानी तेल या पोटेशियम बाइकार्बोनेट स्प्रे का उपयोग करें",
                "उचित छंटाई और रिक्ति द्वारा वायु संचार में सुधार करें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "उच्च जोखिम वाले क्षेत्रों में लक्षण दिखाई देने से पहले फफूंदनाशक लगाएं",
                "नए पेड़ लगाते समय प्रतिरोधी चेरी किस्मों का उपयोग करें",
                "कैनोपी को छाया देने वाले सकर्स और वाटर स्प्राउट्स को हटाएं",
                "अनुकूल परिस्थितियों के दौरान 7-14 दिनों के अंतराल पर फफूंदनाशक लगाएं",
                "नई वृद्धि की सावधानीपूर्वक निगरानी करें क्योंकि यह सबसे अधिक संवेदनशील है"
            ]
        }
    },

    "Cherry_(including_sour)___healthy": {
        "name": {"en": "Healthy Cherry", "ta": "செரி - ஆரோக்கியம்", "hi": "स्वस्थ चेरी"},
        "description": {"en": "No disease", "ta": "நோய் இல்லை", "hi": "रोग नहीं"},
        "treatment": {"en": "Standard care", "ta": "பொதுவான பராமரிப்பு", "hi": "मानक देखभाल"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Prune annually to maintain tree shape and productivity",
                "Apply balanced fertilizer in early spring",
                "Ensure consistent watering during fruit development",
                "Protect trees from bird damage with netting if needed",
                "Monitor for cherry fruit fly and other common pests",
                "Test soil pH and maintain between 6.0-6.8",
                "Remove any water sprouts or suckers promptly",
                "Apply mulch to conserve moisture and suppress weeds",
                "Harvest fruit at proper maturity for best quality"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "மர வடிவம் மற்றும் உற்பத்தித்திறனை பராமரிக்க வருடாந்திர கத்தரித்தல்",
                "வசந்த காலத்தின் தொடக்கத்தில் சீரான உரத்தைப் பயன்படுத்தவும்",
                "பழ வளர்ச்சிய期间 சீரான நீர்ப்பாசனத்தை உறுதிப்படுத்தவும்",
                "தேவைப்பட்டால் பறவை சேதத்திலிருந்து மரங்களை வலை மூலம் பாதுகாக்கவும்",
                "செர்ரி பழ ஈ மற்றும் பிற பொதுவான பூச்சிகளை கண்காணிக்கவும்",
                "மண் pH ஐ சோதித்து 6.0-6.8 க்கு இடையில் பராமரிக்கவும்",
                "எந்தவொரு நீர் முளைகள் அல்லது சக்கர்களையும் உடனடியாக அகற்றவும்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் களைகளை அடக்க மல்ச் பயன்படுத்தவும்",
                "சிறந்த தரத்திற்கு சரியான முதிர்ச்சியில் பழங்களை அறுவடை செய்யுங்கள்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "पेड़ के आकार और उत्पादकता बनाए रखने के लिए सालाना छंटाई करें",
                "शुरुआती वसंत में संतुलित उर्वरक लगाएं",
                "फल विकास के दौरान लगातार पानी सुनिश्चित करें",
                "यदि आवश्यक हो तो नेटिंग से पक्षियों की क्षति से पेड़ों की रक्षा करें",
                "चेरी फ्रूट फ्लाई और अन्य सामान्य कीटों की निगरानी करें",
                "मिट्टी का pH परीक्षण करें और 6.0-6.8 के बीच बनाए रखें",
                "किसी भी वाटर स्प्राउट या सकर को तुरंत हटा दें",
                "नमी संरक्षण और खरपतवार दमन के लिए मल्च लगाएं",
                "सर्वोत्तम गुणवत्ता के लिए उचित परिपक्वता पर फल की कटाई करें"
            ]
        }
    },

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": {"en": "Cercospora / Gray Leaf Spot", "ta": "செர்கோஸ்போரா / சாம்பு இலை நோய்", "hi": "सर्कोस्पोरा / ग्रे लीफ़ स्पॉट"},
        "description": {
            "en": "Small tan-to-gray rectangular lesions between veins on leaves.",
            "ta": "இலைச் சதைகளின் இடைவெளிகளில் கொண்டு சாம்பு வடிவத் திசைகள்.",
            "hi": "पत्तियों की नसों के बीच छोटे भूरे-से-धूसर आयताकार घाव।"
        },
        "treatment": {
            "en": "Rotate crops, remove debris, and apply fungicide when necessary.",
            "ta": "பயிர் மாறுதல், சிதைந்த பகுதிகளை அகற்று; தேவையான போது பூஞ்சிக் கொல்லி பாவி.",
            "hi": "फसल चक्रीकरण, अवशेष निकालें और आवश्यक होने पर फफूंदनाशक लगाएँ।"
        },
        "spot_location": {
            "en": "Rectangular grayish lesions between veins, often on lower leaves first.",
            "ta": "நரம்புகளுக்கு இடையில் சதுரமான சாம்பு ம்ட்டுகள்; பெரும்பாலும் கீழ் இலையில் முதலில்.",
            "hi": "नसों के बीच आयताकार धूसर घाव; अक्सर नीचे की पत्तियों पर पहले दिखते हैं।"
        },
        "treatment_steps": {
            "en": [
                "Practice crop rotation with non-host crops for 2-3 years",
                "Plow under or remove crop debris after harvest",
                "Use resistant hybrid varieties when available",
                "Apply fungicides at first sign of disease in field",
                "Space plants properly to improve air circulation",
                "Avoid working in fields when plants are wet",
                "Use balanced fertility without excess nitrogen",
                "Apply fungicides at tasseling to early silking stages",
                "Monitor lower leaves regularly for early detection",
                "Consider reduced tillage to maintain residue cover"
            ],
            "ta": [
                "2-3 ஆண்டுகளுக்கு ஹோஸ்ட் அல்லாத பயிர்களுடன் பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை உழுதல் அல்லது அகற்றுதல்",
                "கிடைக்கும் போது எதிர்ப்பு கலப்பின வகைகளைப் பயன்படுத்தவும்",
                "வயலில் நோயின் முதல் அறிகுறியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "காற்றோட்டத்தை மேம்படுத்த தாவரங்களை சரியாக இடைவெளி விடவும்",
                "தாவரங்கள் ஈரமாக இருக்கும் போது வயல்களில் வேலை செய்வதைத் தவிர்க்கவும்",
                "அதிக நைட்ரஜன் இல்லாமல் சீரான உரமிடுதலைப் பயன்படுத்தவும்",
                "டாசலிங் முதல் ஆரம்ப சில்கிங் நிலைகள் வரை பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "ஆரம்பகால கண்டறிதலுக்கு கீழ் இலைகளை தவறாமல் கண்காணிக்கவும்",
                "மீதி கவரை பராமரிக்க குறைக்கப்பட்ட உழவைக் கவனியுங்கள்"
            ],
            "hi": [
                "2-3 वर्षों के लिए गैर-होस्ट फसलों के साथ फसल चक्र अपनाएं",
                "कटाई के बाद फसल के अवशेषों को जोतें या हटाएं",
                "उपलब्ध होने पर प्रतिरोधी संकर किस्मों का उपयोग करें",
                "खेत में रोग के पहले लक्षण पर फफूंदनाशक लगाएं",
                "वायु संचार में सुधार के लिए पौधों को ठीक से रिक्ति दें",
                "पौधों के गीले होने पर खेतों में काम करने से बचें",
                "अतिरिक्त नाइट्रोजन के बिना संतुलित उर्वरक का उपयोग करें",
                "टसलिंग से लेकर शुरुआती सिल्किंग अवस्था तक फफूंदनाशक लगाएं",
                "शुरुआती पहचान के लिए निचली पत्तियों की नियमित निगरानी करें",
                "अवशेष आवरण बनाए रखने के लिए कम जुताई पर विचार करें"
            ]
        }
    },

    "Corn_(maize)___Common_rust_": {
        "name": {"en": "Common Rust (Corn)", "ta": "பரபரப்பான மர்மம் (மக்கா)", "hi": "कॉमन रस्ट (मक्का)"},
        "description": {
            "en": "Reddish-brown pustules (uredinia) on both leaf surfaces.",
            "ta": "இரட்டை இலைப் பாகங்களிலும் சிவப்பு பழுப்பு புண்டிகள்.",
            "hi": "दोनों पत्ती सतहों पर लाल-भूरे पेस्टुल्स (यूरेडिनिया)।"
        },
        "treatment": {
            "en": "Use resistant hybrids and apply recommended fungicides if needed.",
            "ta": "உறுதியான வகைகளை பாவிக்கவும்; தேவையாயின் பரிந்துரைக் பூஞ்சிக் கொல்லிகளை பாவிக்கவும்.",
            "hi": "प्रतिरोधी हाइब्रिड का उपयोग और आवश्यक होने पर सुझाए गए फफूंदनाशक लगाएँ।"
        },
        "spot_location": {
            "en": "Round reddish-brown pustules often scattered on upper and lower leaf surfaces.",
            "ta": "சிவப்பு-பழுப்பு சதுர புண்டிகள் ஆகியவை மேல்நிலை மற்றும் கீழ்நிலையில் பரவலாக காணப்படக் கூடும்.",
            "hi": "ऊपरी व निचली दोनों पत्ती सतहों पर गोल लाल-भूरे पेस्टुल्स।"
        },
        "treatment_steps": {
            "en": [
                "Plant rust-resistant hybrid varieties",
                "Apply fungicides when pustules first appear on lower leaves",
                "Time planting to avoid peak rust development periods",
                "Use balanced fertility without excess nitrogen",
                "Monitor fields regularly from knee-high stage",
                "Apply fungicides at first sign of disease in susceptible varieties",
                "Consider early planting to avoid late-season rust",
                "Remove volunteer corn plants that may harbor rust",
                "Use strobilurin or triazole fungicides for best control",
                "Rotate crops to reduce inoculum carryover"
            ],
            "ta": [
                "துரு-எதிர்ப்பு கலப்பின வகைகளை நடவு செய்யுங்கள்",
                "கீழ் இலைகளில் முதலில் புண்டிகள் தோன்றும் போது பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "உச்ச துரு வளர்ச்சி காலங்களைத் தவிர்க்க நடவு நேரம்",
                "அதிக நைட்ரஜன் இல்லாமல் சீரான உரமிடுதலைப் பயன்படுத்தவும்",
                "முழங்கால்-உயர நிலையிலிருந்து வயல்களை தவறாமல் கண்காணிக்கவும்",
                "பாதிக்கப்படக்கூடிய வகைகளில் நோயின் முதல் அறிகுறியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "பருவத்தின் பிற்பகுதியில் துரு தவிர்க்க ஆரம்ப நடவு கருதுங்கள்",
                "துருவை வைத்திருக்கக்கூடிய தன்னார்வ மக்கா தாவரங்களை அகற்றவும்",
                "சிறந்த கட்டுப்பாட்டிற்கு ஸ்ட்ரோபிலுரின் அல்லது ட்ரையசோல் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "இனோகுலம் கேரியோவரைக் குறைக்க பயிர்களை சுழற்றவும்"
            ],
            "hi": [
                "जंग-प्रतिरोधी संकर किस्में लगाएं",
                "निचली पत्तियों पर पहली बार पुस्ट्यूल दिखाई देने पर फफूंदनाशक लगाएं",
                "चरम जंग विकास अवधि से बचने के लिए रोपण का समय निर्धारित करें",
                "अतिरिक्त नाइट्रोजन के बिना संतुलित उर्वरक का उपयोग करें",
                "घुटने की ऊंचाई की अवस्था से नियमित रूप से खेतों की निगरानी करें",
                "संवेदनशील किस्मों में रोग के पहले लक्षण पर फफूंदनाशक लगाएं",
                "मौसम के अंत में जंग से बचने के लिए शीघ्र रोपण पर विचार करें",
                "स्वैच्छिक मक्का के पौधों को हटाएं जो जंग को आश्रय दे सकते हैं",
                "सर्वोत्तम नियंत्रण के लिए स्ट्रोबिलुरिन या ट्राइजोल फफूंदनाशकों का उपयोग करें",
                "इनोकुलम कैरीओवर को कम करने के लिए फसलों को घुमाएं"
            ]
        }
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": {"en": "Northern Leaf Blight", "ta": "வடக்கு இலை அழுக்கு", "hi": "नॉर्दन लीफ ब्लाइट"},
        "description": {
            "en": "Long cigar-shaped gray-green lesions that may coalesce into large necrotic areas.",
            "ta": "நீண்டு சிகர் வடிவத்தில் விழுங்கும் போன்ற சாம்பு-பச்சை தழல்கள், பெரிய பகுதியை ஒத்தாக்கலாம்.",
            "hi": "लंबे सिगार जैसे धूसर-हरे घाव जो मिलकर बड़े मृत हिस्से बना सकते हैं।"
        },
        "treatment": {
            "en": "Use resistant varieties, rotate crops, and apply fungicide at early disease stages.",
            "ta": "பாதுகாப்பான வகைகளை பாவிக்கவும்; பயிர் மாறுதல்; ஆரம்ப கட்டத்தில் பூஞ்சிக் கொல்லி பயன்படுத்தவும்.",
            "hi": "प्रतिरोधी किस्में, फसल चक्रीकरण और रोग के शुरुआती चरण में फफूंदनाशक।"
        },
        "spot_location": {
            "en": "Long, narrow lesions along the leaf, usually starting on lower leaves.",
            "ta": "இலையின் மேலும் நீளமான வாரிசை, பொதுவாக கீழ் இலையிலிருந்து தொடங்கும்.",
            "hi": "पत्ती के साथ लंबी, पतली चोटें, आमतौर पर निचली पत्तियों पर शुरू होती हैं।"
        },
        "treatment_steps": {
            "en": [
                "Plant resistant hybrids with good northern leaf blight resistance",
                "Practice crop rotation with small grains or legumes",
                "Plow under corn debris to reduce overwintering inoculum",
                "Apply fungicides when lesions first appear on lower leaves",
                "Use proper plant spacing to improve air circulation",
                "Avoid continuous corn planting in same field",
                "Apply fungicides at tasseling to early silking stages",
                "Monitor fields regularly after canopy closure",
                "Use balanced fertility without excess nitrogen",
                "Consider reduced tillage to maintain residue cover"
            ],
            "ta": [
                "நல்ல வடக்கு இலை ப்ளைட் எதிர்ப்புடன் எதிர்ப்பு கலப்பினங்களை நடவு செய்யுங்கள்",
                "சிறிய தானியங்கள் அல்லது பருப்பு வகைகளுடன் பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "குளிர்காலத்தில் தங்கும் இனோகுலத்தைக் குறைக்க மக்கா குப்பைகளை உழுதல்",
                "கீழ் இலைகளில் முதலில் புண்கள் தோன்றும் போது பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "காற்றோட்டத்தை மேம்படுத்த சரியான தாவர இடைவெளியைப் பயன்படுத்தவும்",
                "அதே வயலில் தொடர்ச்சியான மக்கா நடவு செய்வதைத் தவிர்க்கவும்",
                "டாசலிங் முதல் ஆரம்ப சில்கிங் நிலைகள் வரை பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "கனோபி மூடிய பிறகு வயல்களை தவறாமல் கண்காணிக்கவும்",
                "அதிக நைட்ரஜன் இல்லாமல் சீரான உரமிடுதலைப் பயன்படுத்தவும்",
                "மீதி கவரை பராமரிக்க குறைக்கப்பட்ட உழவைக் கவனியுங்கள்"
            ],
            "hi": [
                "अच्छी उत्तरी पत्ती झुलसा प्रतिरोध के साथ प्रतिरोधी संकर लगाएं",
                "छोटे अनाज या फलियों के साथ फसल चक्र अपनाएं",
                "सर्दियों में इनोकुलम को कम करने के लिए मक्का के अवशेषों को जोतें",
                "निचली पत्तियों पर घाव के पहले दिखाई देने पर फफूंदनाशक लगाएं",
                "वायु संचार में सुधार के लिए उचित पौध रिक्ति का उपयोग करें",
                "एक ही खेत में लगातार मक्का रोपण से बचें",
                "टसलिंग से लेकर शुरुआती सिल्किंग अवस्था तक फफूंदनाशक लगाएं",
                "कैनोपी बंद होने के बाद नियमित रूप से खेतों की निगरानी करें",
                "अतिरिक्त नाइट्रोजन के बिना संतुलित उर्वरक का उपयोग करें",
                "अवशेष आवरण बनाए रखने के लिए कम जुताई पर विचार करें"
            ]
        }
    },

    "Corn_(maize)___healthy": {
        "name": {"en": "Healthy Maize", "ta": "மக்கா - ஆரோக்கியம்", "hi": "स्वस्थ मक्का"},
        "description": {"en": "No visible disease signs.", "ta": "காணக்கூடிய நोய் இல்லை.", "hi": "कोई दिखाई देने वाला रोग नहीं।"},
        "treatment": {"en": "Maintain good farm practices.", "ta": "நல்ல விவசாய பயிற்சிகளை கடைபிடிக்கவும்.", "hi": "अच्छा कृषीय अभ्यास बनाए रखें।"},
        "spot_location": {"en": "No spots.", "ta": "புள்ளிகள் இல்லை.", "hi": "कोई धब्बे नहीं।"},
        "treatment_steps": {
            "en": [
                "Continue regular field monitoring for pests and diseases",
                "Maintain proper plant spacing for optimal growth",
                "Apply balanced fertilizer based on soil test results",
                "Ensure adequate irrigation during critical growth stages",
                "Practice crop rotation to maintain soil health",
                "Monitor for common corn pests like corn borers and earworms",
                "Test soil regularly and amend pH if necessary",
                "Use integrated pest management strategies",
                "Control weeds that compete with corn for nutrients",
                "Harvest at proper moisture content for best storage"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான வயல் கண்காணிப்பைத் தொடரவும்",
                "உகந்த வளர்ச்சிக்கு சரியான தாவர இடைவெளியை பராமரிக்கவும்",
                "மண் சோதனை முடிவுகளின் அடிப்படையில் சீரான உரத்தைப் பயன்படுத்தவும்",
                "முக்கியமான வளர்ச்சி நிலைகளில் போதுமான நீர்ப்பாசனத்தை உறுதிப்படுத்தவும்",
                "மண் ஆரோக்கியத்தை பராமரிக்க பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "மக்கா போரர்கள் மற்றும் காது புழுக்கள் போன்ற பொதுவான மக்கா பூச்சிகளை கண்காணிக்கவும்",
                "தவறாமல் மண்ணை சோதித்து தேவைப்பட்டால் pH ஐ திருத்தவும்",
                "ஒருங்கிணைந்த பூச்சி மேலாண்மை உத்திகளைப் பயன்படுத்தவும்",
                "மக்காவுடன் ஊட்டச்சத்துக்காக போட்டியிடும் களைகளை கட்டுப்படுத்தவும்",
                "சிறந்த சேமிப்பிற்கு சரியான ஈரப்பத உள்ளடக்கத்தில் அறுவடை செய்யுங்கள்"
            ],
            "hi": [
                "कीटों और रोगों के लिए नियमित खेत निगरानी जारी रखें",
                "इष्टतम विकास के लिए उचित पौध रिक्ति बनाए रखें",
                "मृदा परीक्षण परिणामों के आधार पर संतुलित उर्वरक लगाएं",
                "महत्वपूर्ण विकास अवस्थाओं के दौरान पर्याप्त सिंचाई सुनिश्चित करें",
                "मिट्टी के स्वास्थ्य को बनाए रखने के लिए फसल चक्र अपनाएं",
                "कॉर्न बोरर और अर्कवर्म जैसे सामान्य मक्का कीटों की निगरानी करें",
                "नियमित रूप से मिट्टी का परीक्षण करें और यदि आवश्यक हो तो pH में संशोधन करें",
                "एकीकृत कीट प्रबंधन रणनीतियों का उपयोग करें",
                "उन खरपतवारों को नियंत्रित करें जो मक्का के साथ पोषक तत्वों के लिए प्रतिस्पर्धा करते हैं",
                "सर्वोत्तम भंडारण के लिए उचित नमी सामग्री पर कटाई करें"
            ]
        }
    },

    "Grape___Black_rot": {
        "name": {"en": "Grape Black Rot", "ta": "திராட்சை பிளாக் ராட்", "hi": "अंगूर ब्लैक रॉट"},
        "description": {
            "en": "Fungal disease causing brown-black spots on leaves and rotting of berries.",
            "ta": "இலைகளிலும் பழங்களிலும் கருப்பு-பழுப்பு புள்ளிகள் மற்றும் பழங்கள் உமிழ்தல் ஏற்படும்.",
            "hi": "पत्तियों और बेरीज पर भूरे-काले धब्बे और फल सड़ना।"
        },
        "treatment": {
            "en": "Prune diseased shoots, remove mummified berries, and apply fungicides.",
            "ta": "பாதிக்கப்பட்ட கொம்புகளை வெட்டவும்; சிதைந்த பழங்களை அகற்று; பூஞ்சிக் கொல்லி பராமரிக்கவும்.",
            "hi": "संक्रमित शाखाओं को काटें, सड़े फल हटाएँ, और फफूंदनाशक लगाएँ।"
        },
        "spot_location": {
            "en": "Round dark spots on leaves and shriveled black berries.",
            "ta": "இலையில் வட்டமான கருப்பு புள்ளிகள் மற்றும் சுருங்கிய கருப்பு பழங்கள்.",
            "hi": "पत्तियों पर गोल काले धब्बे और सिकुड़े काले फल।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy all mummified berries from vines and ground",
                "Prune out infected canes and spurs during dormancy",
                "Apply fungicides starting at bud break and continue through fruit set",
                "Use captan, mancozeb, or myclobutanil fungicides",
                "Improve air circulation through proper pruning and training",
                "Remove wild grapes near vineyard that may harbor disease",
                "Apply fungicides before and after rainfall during critical periods",
                "Time sprays carefully from pre-bloom through 3-4 weeks after bloom",
                "Use resistant varieties like 'Catawba' or 'Norton' in high-risk areas",
                "Sanitize pruning tools between vines to prevent spread"
            ],
            "ta": [
                "வைன்கள் மற்றும் தரையில் இருந்து அனைத்து மம்மிஃபைடு பெர்ரிகளையும் அகற்றி அழிக்கவும்",
                "உறக்க நிலையில் பாதிக்கப்பட்ட கேன்கள் மற்றும் ஸ்பர்களை கத்தரிக்கவும்",
                "மொக்கு உடைப்பில் தொடங்கி பழம் அமைக்கும் வரை பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "காப்டான், மான்கோசெப் அல்லது மைக்ளோபுடானில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "சரியான கத்தரித்தல் மற்றும் பயிற்சி மூலம் காற்றோட்டத்தை மேம்படுத்தவும்",
                "நோயை வைத்திருக்கக்கூடிய திராட்சைத் தோட்டத்திற்கு அருகில் உள்ள காட்டு திராட்சைகளை அகற்றவும்",
                "முக்கியமான காலங்களில் மழைக்கு முன்னும் பின்னும் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "பூக்கும் முன் முதல் பூக்கும் 3-4 வாரங்கள் வரை தெளிப்புகளை கவனமாக நேரம் கணக்கிடுங்கள்",
                "அதிக ஆபத்துள்ள பகுதிகளில் 'காடாவ்பா' அல்லது 'நார்டன்' போன்ற எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "பரவலைத் தடுக்க வைன்களுக்கு இடையில் கத்தரிக்கும் கருவிகளை சுத்தம் செய்யுங்கள்"
            ],
            "hi": [
                "बेलों और जमीन से सभी ममीकृत बेरीज को हटाकर नष्ट करें",
                "निष्क्रिय अवस्था के दौरान संक्रमित डंठल और स्पर्स को काटें",
                "कली फूटने से शुरू करके फल लगने तक फफूंदनाशक लगाएं",
                "कैप्टन, मैन्कोजेब, या माइक्लोब्यूटानिल फफूंदनाशकों का उपयोग करें",
                "उचित छंटाई और प्रशिक्षण के माध्यम से वायु संचार में सुधार करें",
                "वाइनयार्ड के पास जंगली अंगूरों को हटाएं जो रोग को आश्रय दे सकते हैं",
                "महत्वपूर्ण अवधि के दौरान बारिश से पहले और बाद में फफूंदनाशक लगाएं",
                "फूल आने से पहले से लेकर फूल आने के 3-4 सप्ताह बाद तक स्प्रे का सावधानीपूर्वक समय निर्धारित करें",
                "उच्च जोखिम वाले क्षेत्रों में 'काटाव्बा' या 'नॉर्टन' जैसी प्रतिरोधी किस्मों का उपयोग करें",
                "प्रसार को रोकने के लिए बेलों के बीच छंटाई के उपकरणों को सैनिटाइज करें"
            ]
        }
    },

    "Grape___Esca_(Black_Measles)": {
        "name": {"en": "Esca (Black Measles)", "ta": "எஸ்கா (கருப்பு பண்டங்கள்)", "hi": "एस्का (ब्लैक मीज़ल्स)"},
        "description": {
            "en": "Complex trunk and wood disease causing spotted fruit and vine decline.",
            "ta": "மரபு மற்றும் மர நோயின் சிக்கல்; பழத்தில் புள்ளிகள் மற்றும் வெயில் குறைப்பு.",
            "hi": "लत और लकड़ी को प्रभावित करने वाला जटिल रोग जो फल पर धब्बे और बेल में गिरावट करता है।"
        },
        "treatment": {
            "en": "Remove affected wood, improve drainage, and use certified planting material.",
            "ta": "பாதிக்கப்பட்ட மரங்களை அகற்றி; நீர் வடிகட்டி; சான்றளிக்கப்பட்ட விதைகளை பயன்படுத்து.",
            "hi": "प्रभावित लकड़ी हटाएँ, जल निकासी सुधरें, प्रमाणित रोपण सामग्री का उपयोग करें।"
        },
        "spot_location": {
            "en": "Irregular dark spots on berries and leaf yellowing near veins.",
            "ta": "முந்திய கறுப்பு புள்ளிகள் பழங்களில்; நரம்புகளுக்கு அருகே இலை மஞ்சள் ஆகும்.",
            "hi": "बेरीज पर अनियमित काले धब्बे और नसों के पास पत्तियों का पीला होना।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy severely infected vines showing decline symptoms",
                "Make proper pruning cuts to promote rapid wound healing",
                "Avoid mechanical injuries to trunks and cordons",
                "Use certified disease-free planting material",
                "Improve soil drainage in poorly drained areas",
                "Apply wound protectants after pruning large cuts",
                "Remove and burn infected wood from vineyard",
                "Avoid excessive nitrogen fertilization",
                "Maintain balanced vine vigor through proper canopy management",
                "Consider trunk renewal on affected vines when possible"
            ],
            "ta": [
                "சரிவு அறிகுறிகள் காட்டும் கடுமையாக பாதிக்கப்பட்ட வைன்களை அகற்றி அழிக்கவும்",
                "விரைவான காயம் குணமாக ஊக்குவிக்க சரியான கத்தரிக்கும் வெட்டுகளைச் செய்யுங்கள்",
                "தண்டுகள் மற்றும் கார்டன்களுக்கு இயந்திர காயங்களைத் தவிர்க்கவும்",
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத நடவு பொருட்களைப் பயன்படுத்தவும்",
                "மோசமாக வடிகட்டிய பகுதிகளில் மண் வடிகால் மேம்படுத்தவும்",
                "பெரிய வெட்டுக்களை கத்தரித்த பிறகு காயம் பாதுகாப்புகளைப் பயன்படுத்தவும்",
                "திராட்சைத் தோட்டத்திலிருந்து பாதிக்கப்பட்ட மரத்தை அகற்றி எரிக்கவும்",
                "அதிக நைட்ரஜன் உரமிடுவதைத் தவிர்க்கவும்",
                "சரியான கனோபி மேலாண்மை மூலம் சீரான வைன் வீரியத்தை பராமரிக்கவும்",
                "முடிந்த時に பாதிக்கப்பட்ட வைன்களில் தண்டு புதுப்பிப்பைக் கவனியுங்கள்"
            ],
            "hi": [
                "गिरावट के लक्षण दिखाने वाली गंभीर रूप से संक्रमित बेलों को हटाकर नष्ट करें",
                "तेजी से घाव भरने को बढ़ावा देने के लिए उचित छंटाई कटौती करें",
                "तनों और कॉर्डन को यांत्रिक चोटों से बचाएं",
                "प्रमाणित रोग-मुक्त रोपण सामग्री का उपयोग करें",
                "खराब जल निकासी वाले क्षेत्रों में मिट्टी की जल निकासी में सुधार करें",
                "बड़े कटौती की छंटाई के बाद घाव के संरक्षक लगाएं",
                "वाइनयार्ड से संक्रमित लकड़ी को हटाकर जलाएं",
                "अत्यधिक नाइट्रोजन उर्वरक से बचें",
                "उचित कैनोपी प्रबंधन के माध्यम से संतुलित बेल जोर बनाए रखें",
                "संभव होने पर प्रभावित बेलों पर तने के नवीनीकरण पर विचार करें"
            ]
        }
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": {"en": "Grape Leaf Blight (Isariopsis)", "ta": "திராட்சை இலை பிளைட் (Isariopsis)", "hi": "अंगूर लीफ ब्लाइट (Isariopsis)"},
        "description": {
            "en": "Leaf spots that can merge into large dead areas, reducing photosynthesis.",
            "ta": "இலையில் புள்ளிகள் பெரிய சாணை பகுதியாக இணைந்து படுக்கலாம்; ஒளிச்சேர்க்கை குறைக்கும்.",
            "hi": "पत्ती पर धब्बे जो मिलकर बड़े मृत क्षेत्र बना सकते हैं और प्रकाश संश्लेषण कम कर देते हैं।"
        },
        "treatment": {
            "en": "Apply recommended fungicides and remove infected leaves; maintain canopy airflow.",
            "ta": "பரிந்துரைத்த பூஞ்சிக் கொல்லிகளை பயன்படுத்தவும்; பாதிக்கப்பட்ட இலைகளை அகற்று; கனோபியை காற்றோட்டம் வைத்திரு.",
            "hi": "सुझाए गए फफूंदनाशक लगाएँ, संक्रमित पत्तियाँ हटाएँ और पर्णसमूह में वायु प्रवाह बनाए रखें।"
        },
        "spot_location": {
            "en": "Small to large brown spots on leaf surface, often forming patches.",
            "ta": "இலையின் மேற்பரப்பில் சிறிய முதல் பெரிய புள்ளிகள், பலமுறை தொகுதிகள் உருவாக்கும்.",
            "hi": "पत्ती सतह पर छोटे से बड़े भूरे धब्बे, अक्सर पैच बना लेते हैं।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy severely infected leaves when possible",
                "Apply copper-based fungicides early in the season",
                "Use mancozeb or chlorothalonil for preventive control",
                "Improve air circulation through proper canopy management",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Apply fungicides at 10-14 day intervals during favorable conditions",
                "Remove weeds that may harbor the disease",
                "Time sprays to protect new growth as it emerges",
                "Use resistant grape varieties when planting new vines",
                "Monitor vineyards regularly from bud break through harvest"
            ],
            "ta": [
                "முடிந்த時に கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்",
                "பருவத்தின் ஆரம்பத்தில் தாமிரம் அடிப்படையிலான பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "தடுப்பு கட்டுப்பாட்டிற்கு மான்கோசெப் அல்லது குளோரோதலோனில் பயன்படுத்தவும்",
                "சரியான கனோபி மேலாண்மை மூலம் காற்றோட்டத்தை மேம்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "சாதகமான நிலைமைகளில் 10-14 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "நோயை வைத்திருக்கக்கூடிய களைகளை அகற்றவும்",
                "புதிய வளர்ச்சி எழும்போது அதைப் பாதிக்க தெளிப்புகளை நேரம் கணக்கிடுங்கள்",
                "புதிய வைன்களை நடும் போது எதிர்ப்பு திராட்சை வகைகளைப் பயன்படுத்தவும்",
                "மொக்கு உடைப்பிலிருந்து அறுவடை வரை திராட்சைத் தோட்டங்களை தவறாமல் கண்காணிக்கவும்"
            ],
            "hi": [
                "यदि संभव हो तो गंभीर रूप से संक्रमित पत्तियों को हटाकर नष्ट करें",
                "मौसम की शुरुआत में कॉपर-आधारित फफूंदनाशक लगाएं",
                "निवारक नियंत्रण के लिए मैन्कोजेब या क्लोरोथैलोनिल का उपयोग करें",
                "उचित कैनोपी प्रबंधन के माध्यम से वायु संचार में सुधार करें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "अनुकूल परिस्थितियों के दौरान 10-14 दिनों के अंतराल पर फफूंदनाशक लगाएं",
                "उन खरपतवारों को हटाएं जो रोग को आश्रय दे सकते हैं",
                "नई वृद्धि के उभरने पर उसे सुरक्षित रखने के लिए स्प्रे का समय निर्धारित करें",
                "नई बेलें लगाते समय प्रतिरोधी अंगूर किस्मों का उपयोग करें",
                "कली फूटने से लेकर कटाई तक नियमित रूप से अंगूर के बागों की निगरानी करें"
            ]
        }
    },

    "Grape___healthy": {
        "name": {"en": "Healthy Grape", "ta": "திராட்சை - ஆரோக்கியம்", "hi": "स्वस्थ अंगूर"},
        "description": {"en": "No visible disease signs.", "ta": "காணக்கூடிய நோய் இல்லை.", "hi": "कोई दिखाई देने वाला रोग नहीं।"},
        "treatment": {"en": "Maintain best practices.", "ta": "சிறந்த பழக்க வழக்கங்களை தொடரவும்.", "hi": "सर्वोत्तम अभ्यास बनाए रखें।"},
        "spot_location": {"en": "No spots.", "ta": "புள்ளிகள் இல்லை.", "hi": "कोई धब्बे नहीं।"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Prune annually during dormancy to maintain vine structure",
                "Apply balanced fertilizer based on soil test results",
                "Manage canopy for optimal sunlight exposure and air circulation",
                "Monitor for common grape pests like grape berry moth",
                "Test soil pH and maintain between 5.5-6.5 for grapes",
                "Use drip irrigation to conserve water and reduce leaf wetness",
                "Control weeds that compete with vines for nutrients",
                "Harvest grapes at proper sugar content for intended use",
                "Keep records of vineyard operations and observations"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "வைன் கட்டமைப்பை பராமரிக்க உறக்க நிலையில் வருடாந்திர கத்தரித்தல்",
                "மண் சோதனை முடிவுகளின் அடிப்படையில் சீரான உரத்தைப் பயன்படுத்தவும்",
                "உகந்த சூரிய ஒளி வெளிப்பாடு மற்றும் காற்றோட்டத்திற்கு கனோபியை நிர்வகிக்கவும்",
                "திராட்சை பெர்ரி அந்துப்பூச்சி போன்ற பொதுவான திராட்சை பூச்சிகளை கண்காணிக்கவும்",
                "மண் pH ஐ சோதித்து திராட்சைக்கு 5.5-6.5 க்கு இடையில் பராமரிக்கவும்",
                "நீரைச் சேமிக்க மற்றும் இலை ஈரப்பதத்தைக் குறைக்க டிரிப் நீர்ப்பாசனத்தைப் பயன்படுத்தவும்",
                "வைன்களுடன் ஊட்டச்சத்துக்காக போட்டியிடும் களைகளை கட்டுப்படுத்தவும்",
                "நோக்கம் கொண்ட பயன்பாட்டிற்கு சரியான சர்க்கரை உள்ளடக்கத்தில் திராட்சையை அறுவடை செய்யுங்கள்",
                "திராட்சைத் தோட்ட செயல்பாடுகள் மற்றும் கண்காணிப்புகளின் பதிவுகளை வைத்திருங்கள்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "बेल संरचना बनाए रखने के लिए निष्क्रिय अवस्था के दौरान सालाना छंटाई करें",
                "मृदा परीक्षण परिणामों के आधार पर संतुलित उर्वरक लगाएं",
                "इष्टतम सूर्य के प्रकाश एक्सपोजर और वायु संचार के लिए कैनोपी प्रबंधित करें",
                "ग्रेप बेरी मॉथ जैसे सामान्य अंगूर कीटों की निगरानी करें",
                "मिट्टी का pH परीक्षण करें और अंगूरों के लिए 5.5-6.5 के बीच बनाए रखें",
                "पानी बचाने और पत्तियों की नमी कम करने के लिए ड्रिप सिंचाई का उपयोग करें",
                "उन खरपतवारों को नियंत्रित करें जो बेलों के साथ पोषक तत्वों के लिए प्रतिस्पर्धा करते हैं",
                "इच्छित उपयोग के लिए उचित चीनी सामग्री पर अंगूर की कटाई करें",
                "वाइनयार्ड संचालन और अवलोकनों का रिकॉर्ड रखें"
            ]
        }
    },

    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": {"en": "Huanglongbing (Citrus Greening)", "ta": "சிட்ரஸ் கிரீனிங்", "hi": "हुआन्गलोंगबिंग (साइट्रस ग्रिनिंग)"},
        "description": {
            "en": "Bacterial disease causing yellowing, mottled leaves, and poor fruit quality.",
            "ta": "பாக்டீரியா நோய்; இலை மஞ்சள், சிதறல் மற்றும் பழத்தின் தர குறைவு ஏற்படுகிறது.",
            "hi": "बैक्टीरियल रोग जो पत्तियों का पीला पड़ना, धब्बे और फल की खराब गुणवत्ता पैदा करता है।"
        },
        "treatment": {
            "en": "No cure; remove infected trees, control psyllid vectors and use clean nursery stock.",
            "ta": "மோதமிக்க மரங்களை அகற்று; உயிர்வழி கட்டுப்பாடு மற்றும் சுத்தமான செடிகள் பயன்படுத்தவும்.",
            "hi": "उपचार नहीं; संक्रमित पेड़ों को हटाएँ, वेक्टर नियंत्रित करें और साफ नर्सरी सामग्री का प्रयोग करें।"
        },
        "spot_location": {
            "en": "Asymmetric yellow mottling on leaves, often with vein clearing.",
            "ta": "இலையில் மாறுபட்ட மஞ்சள் நிற மேல் கோடுகள்; நரம்பு பகுதிகள் தெளிவாக தோன்றும்.",
            "hi": "पत्तियों पर असममित पीला धब्बा, अक्सर नसों के पास स्पष्टता।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy infected trees immediately upon detection",
                "Control Asian citrus psyllid vectors with systemic insecticides",
                "Use only certified disease-free nursery stock for new plantings",
                "Monitor orchards regularly for early symptoms of HLB",
                "Remove alternative host plants that may harbor psyllids",
                "Implement area-wide psyllid management programs",
                "Use yellow sticky traps to monitor psyllid populations",
                "Apply soil-applied systemic insecticides for psyllid control",
                "Coordinate with neighbors for community-wide management",
                "Consider planting HLB-tolerant varieties where available"
            ],
            "ta": [
                "கண்டறியப்பட்டதும் உடனடியாக பாதிக்கப்பட்ட மரங்களை அகற்றி அழிக்கவும்",
                "சிஸ்டமிக் பூச்சிக்கொல்லிகளுடன் ஆசிய சிட்ரஸ் சில்லிட் வெக்டர்களை கட்டுப்படுத்தவும்",
                "புதிய நடவு செய்வதற்கு சான்றளிக்கப்பட்ட நோய்-இல்லாத நர்சரி பங்குகளை மட்டுமே பயன்படுத்தவும்",
                "HLB இன் ஆரம்ப அறிகுறிகளுக்கு தோட்டங்களை தவறாமல் கண்காணிக்கவும்",
                "சில்லிட்களை வைத்திருக்கக்கூடிய மாற்று ஹோஸ்ட் தாவரங்களை அகற்றவும்",
                "பரவலான சில்லிட் மேலாண்மை திட்டங்களை செயல்படுத்தவும்",
                "சில்லிட் மக்கள்தொகையை கண்காணிக்க மஞ்சள் ஒட்டும் பொறிகளைப் பயன்படுத்தவும்",
                "சில்லிட் கட்டுப்பாட்டிற்கு மண்ணில் பயன்படுத்தப்படும் சிஸ்டமிக் பூச்சிக்கொல்லிகளைப் பயன்படுத்தவும்",
                "சமூகம் தழுவிய மேலாண்மைக்கு அண்டை வீட்டாருடன் ஒருங்கிணைக்கவும்",
                "கிடைக்கும் இடங்களில் HLB-சகிப்புத்தன்மை கொண்ட வகைகளை நடவு செய்வதைக் கவனியுங்கள்"
            ],
            "hi": [
                "पता चलने पर तुरंत संक्रमित पेड़ों को हटाकर नष्ट करें",
                "सिस्टमिक कीटनाशकों के साथ एशियाई साइट्रस साइलिड वैक्टरों को नियंत्रित करें",
                "नए पौधे लगाने के लिए केवल प्रमाणित रोग-मुक्त नर्सरी स्टॉक का उपयोग करें",
                "HLB के शुरुआती लक्षणों के लिए नियमित रूप से बागों की निगरानी करें",
                "वैकल्पिक होस्ट पौधों को हटाएं जो साइलिड को आश्रय दे सकते हैं",
                "क्षेत्र-व्यापी साइलिड प्रबंधन कार्यक्रम लागू करें",
                "साइलिड आबादी की निगरानी के लिए पीले चिपचिपे जाल का उपयोग करें",
                "साइलिड नियंत्रण के लिए मिट्टी में लगाए जाने वाले सिस्टमिक कीटनाशक लगाएं",
                "समुदाय-व्यापी प्रबंधन के लिए पड़ोसियों के साथ समन्वय करें",
                "उपलब्ध होने पर HLB-सहनशील किस्में लगाने पर विचार करें"
            ]
        }
    },

    "Peach___Bacterial_spot": {
        "name": {"en": "Peach Bacterial Spot", "ta": "பீச் பாக்டீரியா புள்ளி", "hi": "पीच बैक्टीरियल स्पॉट"},
        "description": {
            "en": "Bacterial lesions on leaves and fruit leading to scabby spots.",
            "ta": "இலை மற்றும் பழங்களில் பாக்டீரிய புள்ளிகள்; கருநிலையாக தோன்றும்.",
            "hi": "पत्तियों और फलों पर बैक्टीरियल घाव जो खुरदुरे धब्बे बनाते हैं।"
        },
        "treatment": {
            "en": "Use copper sprays, avoid overhead irrigation and remove affected fruit.",
            "ta": "காப்பர் தெழுகலைப் பயன்படுத்தவும்; மேலே நீர் ஊற்றுவதை தவிர்க்கவும்; பாதிக்கப்பட்ட பழங்களை அகற்று.",
            "hi": "कॉपर छिड़काव करें, ऊपर से सिंचाई से बचें और संक्रमित फल हटाएँ।"
        },
        "spot_location": {
            "en": "Small dark spots on leaves and fruit skin, sometimes surrounded by yellow halos.",
            "ta": "இலை மற்றும் பழம் மேற்பரப்பில் சிறிய கருப்பு புள்ளிகள்; சில நேரங்களில் மஞ்சள் சுற்றவட்டம்.",
            "hi": "पत्तियों और फल की त्वचा पर छोटे काले धब्बे, कभी-कभी पीले घेरे।"
        },
        "treatment_steps": {
            "en": [
                "Apply copper bactericides during dormancy and before bud break",
                "Use streptomycin or oxytetracycline during bloom if permitted",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Remove and destroy severely infected fruit and leaves",
                "Plant resistant varieties like 'Candor' or 'Harrow Diamond'",
                "Apply sprays during dry weather for better coverage",
                "Time copper applications to cover susceptible growth stages",
                "Improve air circulation through proper pruning",
                "Monitor orchards regularly during warm, wet weather",
                "Use windbreaks to reduce bacterial spread by wind-driven rain"
            ],
            "ta": [
                "உறக்க நிலை மற்றும் மொக்கு உடைப்பதற்கு முன் தாமிர பாக்டீரிசைடுகளைப் பயன்படுத்தவும்",
                "அனுமதிக்கப்பட்டால் பூக்கும் போது ஸ்ட்ரெப்டோமைசின் அல்லது ஆக்சிடெட்ராசைக்ளின் பயன்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "கடுமையாக பாதிக்கப்பட்ட பழங்கள் மற்றும் இலைகளை அகற்றி அழிக்கவும்",
                "'கேண்டர்' அல்லது 'ஹாரோ டயமண்ட்' போன்ற எதிர்ப்பு வகைகளை நடவு செய்யுங்கள்",
                "சிறந்த கவரேஜிற்கு வறண்ட வானிலையில் தெளிப்புகளைப் பயன்படுத்தவும்",
                "பாதிக்கப்படக்கூடிய வளர்ச்சி நிலைகளை உள்ளடக்கிய தாமிர பயன்பாடுகளை நேரம் கணக்கிடுங்கள்",
                "சரியான கத்தரித்தல் மூலம் காற்றோட்டத்தை மேம்படுத்தவும்",
                "சூடான, ஈரமான வானிலையில் தோட்டங்களை தவறாமல் கண்காணிக்கவும்",
                "காற்றால் இயக்கப்படும் மழையால் பாக்டீரியா பரவுவதைக் குறைக்க காற்று தடுப்புகளைப் பயன்படுத்தவும்"
            ],
            "hi": [
                "निष्क्रिय अवस्था और कली फूटने से पहले कॉपर जीवाणुनाशक लगाएं",
                "यदि अनुमति हो तो फूल आने के दौरान स्ट्रेप्टोमाइसिन या ऑक्सीटेट्रासाइक्लिन का उपयोग करें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "गंभीर रूप से संक्रमित फलों और पत्तियों को हटाकर नष्ट करें",
                "'कैंडर' या 'हैरो डायमंड' जैसी प्रतिरोधी किस्में लगाएं",
                "बेहतर कवरेज के लिए शुष्क मौसम के दौरान स्प्रे लगाएं",
                "संवेदनशील विकास अवस्थाओं को कवर करने के लिए कॉपर अनुप्रयोगों का समय निर्धारित करें",
                "उचित छंटाई के माध्यम से वायु संचार में सुधार करें",
                "गर्म, गीले मौसम के दौरान नियमित रूप से बागों की निगरानी करें",
                "हवा से चलने वाली बारिश से जीवाणु फैलने को कम करने के लिए विंडब्रेक का उपयोग करें"
            ]
        }
    },

    "Peach___healthy": {
        "name": {"en": "Healthy Peach", "ta": "பீச் - ஆரோக்கியம்", "hi": "स्वस्थ पीच"},
        "description": {"en": "No disease visible", "ta": "நோய் தெரியாது", "hi": "रोग नहीं दिखाई देता"},
        "treatment": {"en": "General care", "ta": "பொதுவான பராமரிப்பு", "hi": "सामान्य देखभाल"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Prune annually to maintain open center tree structure",
                "Apply balanced fertilizer based on soil test results",
                "Thin fruit to improve size and quality",
                "Monitor for common peach pests like peach tree borers",
                "Ensure adequate irrigation during fruit development",
                "Test soil pH and maintain between 6.0-6.5",
                "Apply dormant oil sprays to control overwintering pests",
                "Use mulch to conserve moisture and suppress weeds",
                "Harvest fruit at proper maturity for best flavor"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "திறந்த மைய மர கட்டமைப்பை பராமரிக்க வருடாந்திர கத்தரித்தல்",
                "மண் சோதனை முடிவுகளின் அடிப்படையில் சீரான உரத்தைப் பயன்படுத்தவும்",
                "அளவு மற்றும் தரத்தை மேம்படுத்த பழங்களை மெல்லியதாக்குங்கள்",
                "பீச் ட்ரீ போரர்கள் போன்ற பொதுவான பீச் பூச்சிகளை கண்காணிக்கவும்",
                "பழ வளர்ச்சிய期间 போதுமான நீர்ப்பாசனத்தை உறுதிப்படுத்தவும்",
                "மண் pH ஐ சோதித்து 6.0-6.5 க்கு இடையில் பராமரிக்கவும்",
                "குளிர்காலத்தில் தங்கும் பூச்சிகளை கட்டுப்படுத்த உறக்க எண்ணெய் தெளிப்புகளைப் பயன்படுத்தவும்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் களைகளை அடக்க மல்ச் பயன்படுத்தவும்",
                "சிறந்த சுவைக்கு சரியான முதிர்ச்சியில் பழங்களை அறுவடை செய்யுங்கள்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "खुले केंद्र वाली पेड़ संरचना बनाए रखने के लिए सालाना छंटाई करें",
                "मृदा परीक्षण परिणामों के आधार पर संतुलित उर्वरक लगाएं",
                "आकार और गुणवत्ता में सुधार के लिए फलों को पतला करें",
                "पीच ट्री बोरर जैसे सामान्य पीच कीटों की निगरानी करें",
                "फल विकास के दौरान पर्याप्त सिंचाई सुनिश्चित करें",
                "मिट्टी का pH परीक्षण करें और 6.0-6.5 के बीच बनाए रखें",
                "सर्दियों में कीटों को नियंत्रित करने के लिए निष्क्रिय तेल स्प्रे लगाएं",
                "नमी संरक्षण और खरपतवार दमन के लिए मल्च का उपयोग करें",
                "सर्वोत्तम स्वाद के लिए उचित परिपक्वता पर फल की कटाई करें"
            ]
        }
    },

    "Pepper,_bell___Bacterial_spot": {
        "name": {"en": "Bell Pepper Bacterial Spot", "ta": "மிளகாய் பாக்டீரியா புள்ளி", "hi": "शिमला मिर्च बैक्टीरियल स्पॉट"},
        "description": {
            "en": "Bacterial lesions causing dark sunken spots on fruit and leaves.",
            "ta": "பழங்கள் மற்றும் இலைகளில் கருப்பு அழுகிய புள்ளிகள் உருவாகும்.",
            "hi": "फल और पत्तियों पर डूबे हुए काले धब्बे पैदा करने वाले बैक्टीरियल घाव।"
        },
        "treatment": {
            "en": "Remove infected material, avoid wetting foliage, apply copper-based sprays.",
            "ta": "பாதிக்கப்பட்டவை அகற்று; இலைகளை ஈரப்படுத்தாதீர்; காப்பர் தெளுகலை பாவி.",
            "hi": "संक्रमित सामग्री हटाएँ, पत्तियों को गीला न करें, कॉपर स्प्रे लगाएँ।"
        },
        "spot_location": {
            "en": "Dark, water-soaked lesions on fruit surface and leaf margins.",
            "ta": "பழ மேற்பரப்பில் மற்றும் இலை விளிம்புகளில் கரும் நீர் நுகர்ந்த புள்ளிகள்.",
            "hi": "फल सतह और पत्ती किनारों पर काले, पानी से भरे घाव।"
        },
        "treatment_steps": {
            "en": [
                "Use certified disease-free seed and transplants",
                "Apply copper-based bactericides preventively",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Remove and destroy infected plants promptly",
                "Practice 2-3 year crop rotation with non-host crops",
                "Disinfect tools and equipment between uses",
                "Use resistant varieties when available",
                "Apply streptomycin sprays if copper resistance develops",
                "Space plants properly for good air circulation",
                "Avoid working in fields when plants are wet"
            ],
            "ta": [
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத விதைகள் மற்றும் மாற்றுகளைப் பயன்படுத்தவும்",
                "தடுப்பு முறையில் தாமிரம் அடிப்படையிலான பாக்டீரிசைடுகளைப் பயன்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "பாதிக்கப்பட்ட தாவரங்களை உடனடியாக அகற்றி அழிக்கவும்",
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் 2-3 ஆண்டு பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "பயன்பாடுகளுக்கு இடையில் கருவிகள் மற்றும் உபகரணங்களை கிருமி நீக்கம் செய்யுங்கள்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "தாமிர எதிர்ப்பு வளர்ந்தால் ஸ்ட்ரெப்டோமைசின் தெளிப்புகளைப் பயன்படுத்தவும்",
                "நல்ல காற்றோட்டத்திற்கு தாவரங்களை சரியாக இடைவெளி விடவும்",
                "தாவரங்கள் ஈரமாக இருக்கும் போது வயல்களில் வேலை செய்வதைத் தவிர்க்கவும்"
            ],
            "hi": [
                "प्रमाणित रोग-मुक्त बीज और पौध का उपयोग करें",
                "निवारक रूप से कॉपर-आधारित जीवाणुनाशक लगाएं",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "संक्रमित पौधों को तुरंत हटाकर नष्ट करें",
                "गैर-होस्ट फसलों के साथ 2-3 वर्ष का फसल चक्र अपनाएं",
                "उपयोगों के बीच उपकरणों और उपकरणों को कीटाणुरहित करें",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "यदि कॉपर प्रतिरोध विकसित हो तो स्ट्रेप्टोमाइसिन स्प्रे लगाएं",
                "अच्छे वायु संचार के लिए पौधों को ठीक से रिक्ति दें",
                "पौधों के गीले होने पर खेतों में काम करने से बचें"
            ]
        }
    },

    "Pepper,_bell___healthy": {
        "name": {"en": "Healthy Bell Pepper", "ta": "மிளகாய் - ஆரோக்கியம்", "hi": "स्वस्थ शिमला मिर्च"},
        "description": {"en": "No visible disease", "ta": "காணக்கூடிய நோய் இல்லை", "hi": "कोई रोग नहीं"},
        "treatment": {"en": "Standard care", "ta": "மாதிரி பராமரிப்பு", "hi": "मानक देखभाल"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Provide consistent moisture especially during fruit set",
                "Apply balanced fertilizer according to soil test",
                "Use mulch to conserve moisture and suppress weeds",
                "Monitor for common pepper pests like aphids and mites",
                "Ensure proper spacing for good air circulation",
                "Test soil pH and maintain between 6.0-6.8",
                "Use row covers for early season protection if needed",
                "Harvest peppers regularly to encourage continued production",
                "Rotate peppers with unrelated crops each year"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "பழம் அமைக்கும்期间 குறிப்பாக சீரான ஈரப்பதத்தை வழங்கவும்",
                "மண் சோதனைக்கு ஏற்ப சீரான உரத்தைப் பயன்படுத்தவும்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் களைகளை அடக்க மல்ச் பயன்படுத்தவும்",
                "அப்பிட்கள் மற்றும் மைட்டுகள் போன்ற பொதுவான மிளகாய் பூச்சிகளை கண்காணிக்கவும்",
                "நல்ல காற்றோட்டத்திற்கு சரியான இடைவெளியை உறுதிப்படுத்தவும்",
                "மண் pH ஐ சோதித்து 6.0-6.8 க்கு இடையில் பராமரிக்கவும்",
                "தேவைப்பட்டால் ஆரம்ப பருவ பாதுகாப்பிற்கு வரிசை உறைகளைப் பயன்படுத்தவும்",
                "தொடர்ந்து உற்பத்தியை ஊக்குவிக்க வழக்கமாக மிளகாயை அறுவடை செய்யுங்கள்",
                "ஒவ்வொரு ஆண்டும் தொடர்பில்லாத பயிர்களுடன் மிளகாயை சுழற்றவும்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "फल लगने के दौरान विशेष रूप से लगातार नमी प्रदान करें",
                "मृदा परीक्षण के अनुसार संतुलित उर्वरक लगाएं",
                "नमी संरक्षण और खरपतवार दमन के लिए मल्च का उपयोग करें",
                "एफिड्स और माइट्स जैसे सामान्य मिर्च के कीटों की निगरानी करें",
                "अच्छे वायु संचार के लिए उचित रिक्ति सुनिश्चित करें",
                "मिट्टी का pH परीक्षण करें और 6.0-6.8 के बीच बनाए रखें",
                "यदि आवश्यक हो तो शुरुआती मौसम की सुरक्षा के लिए पंक्ति कवर का उपयोग करें",
                "निरंतर उत्पादन को प्रोत्साहित करने के लिए नियमित रूप से मिर्च की कटाई करें",
                "प्रत्येक वर्ष असंबंधित फसलों के साथ मिर्च को घुमाएं"
            ]
        }
    },

    "Potato___Early_blight": {
        "name": {"en": "Potato Early Blight", "ta": "ஆலு முதற்கட்ட அழற்சி", "hi": "आलू अर्ली ब्लाइट"},
        "description": {
            "en": "Targets older leaves with brown concentric rings; can defoliate plants.",
            "ta": "பழைய இலைகளை உடைத்துக் கொள்வதற்கான வெட்டை வளையேற்றம்; தாவரங்கள் இலை இழக்கலாம்.",
            "hi": "पुरानी पत्तियों को निशाना बनाकर भूरे समकेंद्र रिंग बनाता है; पौधे पत्तियाँ खो सकते हैं।"
        },
        "treatment": {
            "en": "Crop rotation, remove residues, and apply recommended fungicides.",
            "ta": "பயிர் மாறுதல், துகள்களை நீக்கு; பரிந்துரைக்கப்பட்ட பூஞ்சிக் கொல்லிகளை பாவிக்கவும்.",
            "hi": "फसल चक्रीकरण, अवशेष हटाना और सुझाए गए फफूंदनाशक लगाना।"
        },
        "spot_location": {
            "en": "Concentric brown rings on older leaves starting at the margin or center.",
            "ta": "பழைய இலைக்களில் வளிமண்டலம் மாமிசமான கருப்பு வளைகள்.",
            "hi": "पुरानी पत्तियों पर समकेंद्र भूरे छल्ले, किनारे या केन्द्र से शुरू।"
        },
        "treatment_steps": {
            "en": [
                "Practice 3-4 year crop rotation with non-host crops",
                "Remove and destroy crop debris after harvest",
                "Use certified disease-free seed potatoes",
                "Apply fungicides when plants are 6-8 inches tall",
                "Use chlorothalonil or mancozeb for preventive control",
                "Maintain adequate potassium levels in soil",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Space plants properly for good air circulation",
                "Apply fungicides at 7-10 day intervals during favorable conditions",
                "Monitor lower leaves regularly for early symptoms"
            ],
            "ta": [
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் 3-4 ஆண்டு பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை அகற்றி அழிக்கவும்",
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத விதை உருளைக்கிழங்குகளைப் பயன்படுத்தவும்",
                "தாவரங்கள் 6-8 அங்குல உயரமாக இருக்கும் போது பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "தடுப்பு கட்டுப்பாட்டிற்கு குளோரோதலோனில் அல்லது மான்கோசெப் பயன்படுத்தவும்",
                "மண்ணில் போதுமான பொட்டாசியம் அளவை பராமரிக்கவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "நல்ல காற்றோட்டத்திற்கு தாவரங்களை சரியாக இடைவெளி விடவும்",
                "சாதகமான நிலைமைகளில் 7-10 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "ஆரம்ப அறிகுறிகளுக்கு கீழ் இலைகளை தவறாமல் கண்காணிக்கவும்"
            ],
            "hi": [
                "गैर-होस्ट फसलों के साथ 3-4 वर्ष का फसल चक्र अपनाएं",
                "कटाई के बाद फसल के अवशेषों को हटाकर नष्ट करें",
                "प्रमाणित रोग-मुक्त आलू के बीज का उपयोग करें",
                "जब पौधे 6-8 इंच लंबे हों तो फफूंदनाशक लगाएं",
                "निवारक नियंत्रण के लिए क्लोरोथैलोनिल या मैन्कोजेब का उपयोग करें",
                "मिट्टी में पर्याप्त पोटेशियम स्तर बनाए रखें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "अच्छे वायु संचार के लिए पौधों को ठीक से रिक्ति दें",
                "अनुकूल परिस्थितियों के दौरान 7-10 दिनों के अंतराल पर फफूंदनाशक लगाएं",
                "शुरुआती लक्षणों के लिए निचली पत्तियों की नियमित निगरानी करें"
            ]
        }
    },

    "Potato___Late_blight": {
        "name": {"en": "Potato Late Blight", "ta": "ஆலு பின்னர் அழற்சி", "hi": "आलू लेट ब्लाइट"},
        "description": {
            "en": "Rapidly spreading water-soaked lesions leading to plant collapse and tuber rot.",
            "ta": "தொழில் விரைவில் பரவுமாறு நீர் நன்கு புகும் பாதிப்பு; தாவரம் அழுகும்.",
            "hi": "तेजी से फैलने वाले पानी-भीगे घाव जो पौधे के ढहने और ट्यूबर सड़ने का कारण बनते हैं।"
        },
        "treatment": {
            "en": "Use certified seed, apply fungicides promptly, and remove infected plants.",
            "ta": "சான்றளிக்கப்பட்ட விதைகளைப் பயன்படுத்தவும்; விரைவில் பூஞ்சிக் கொல்லி பயன்படுத்தவும்; பாதிக்கப்பட்ட தாவரங்களை அகற்று.",
            "hi": "प्रमाणित बीज का उपयोग करें, तुरंत फफूंदनाशक लगाएं और संक्रमित पौधों को हटाएं।"
        },
        "spot_location": {
            "en": "Large irregular water-soaked lesions on leaf and stem; white sporulation under humidity.",
            "ta": "இலை மற்றும் தண்டுகளில் பெரிய ஒழுங்கற்ற நீர் நன்கு புண்ணாக்கள்; ஈரப்பதத்தில் வெள்ளை பூஞ்சை உற்பத்தி.",
            "hi": "पत्ती और तने पर बड़े अनियमित पानी-भीगे घाव; नमी में सफेद स्पोरेशन।"
        },
        "treatment_steps": {
            "en": [
                "Use certified disease-free seed potatoes",
                "Destroy volunteer potatoes and nightshade weeds",
                "Apply fungicides preventively before disease appears",
                "Use metalaxyl or mefenoxam for systemic protection",
                "Apply protectant fungicides like chlorothalonil regularly",
                "Destroy infected plants immediately upon detection",
                "Hill potatoes properly to protect tubers from spores",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Harvest potatoes only after vines are completely dead",
                "Store potatoes in cool, dry conditions with good ventilation"
            ],
            "ta": [
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத விதை உருளைக்கிழங்குகளைப் பயன்படுத்தவும்",
                "தன்னார்வ உருளைக்கிழங்குகள் மற்றும் நைட்ஷேட் களைகளை அழிக்கவும்",
                "நோய் தோன்றுவதற்கு முன் தடுப்பு முறையில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "சிஸ்டமிக் பாதுகாப்பிற்கு மெட்டலாக்சில் அல்லது மெஃபினாக்சம் பயன்படுத்தவும்",
                "குளோரோதலோனில் போன்ற பாதுகாப்பு பூஞ்சைக் கொல்லிகளை தவறாமல் பயன்படுத்தவும்",
                "கண்டறியப்பட்டதும் உடனடியாக பாதிக்கப்பட்ட தாவரங்களை அழிக்கவும்",
                "வித்துக்களிலிருந்து கிழங்குகளைப் பாதுகாக்க உருளைக்கிழங்குகளை சரியாக மலைக்குச் செல்லவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "வைன்கள் முற்றிலும் இறந்த பிறகே உருளைக்கிழங்கை அறுவடை செய்யுங்கள்",
                "நல்ல காற்றோட்டத்துடன் குளிர்ந்த, வறண்ட நிலைமைகளில் உருளைக்கிழங்கை சேமிக்கவும்"
            ],
            "hi": [
                "प्रमाणित रोग-मुक्त आलू के बीज का उपयोग करें",
                "स्वैच्छिक आलू और नाइटशेड खरपतवारों को नष्ट करें",
                "रोग के प्रकट होने से पहले निवारक रूप से फफूंदनाशक लगाएं",
                "सिस्टमिक सुरक्षा के लिए मेटालैक्सिल या मेफेनोक्सम का उपयोग करें",
                "क्लोरोथैलोनिल जैसे सुरक्षात्मक फफूंदनाशकों को नियमित रूप से लगाएं",
                "पता चलने पर तुरंत संक्रमित पौधों को नष्ट करें",
                "बीजाणुओं से कंदों की रक्षा के लिए आलू को ठीक से ढेर करें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "बेलें पूरी तरह से मरने के बाद ही आलू की कटाई करें",
                "अच्छे वेंटिलेशन के साथ ठंडी, सूखी परिस्थितियों में आलू संग्रहीत करें"
            ]
        }
    },

    "Potato___healthy": {
        "name": {"en": "Healthy Potato", "ta": "ஆலு - ஆரோக்கியம்", "hi": "स्वस्थ आलू"},
        "description": {"en": "No visible disease", "ta": "பூஞ்சை இல்லை", "hi": "रोग नहीं दिखाई देता"},
        "treatment": {"en": "Standard agronomy", "ta": "பொதுவான விவசாயம்", "hi": "सामान्य कृषि अभ्यास"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Practice crop rotation with non-host crops",
                "Use certified disease-free seed potatoes",
                "Hill potatoes properly to protect developing tubers",
                "Maintain consistent soil moisture during tuber development",
                "Monitor for common potato pests like Colorado potato beetle",
                "Test soil and maintain proper pH and fertility",
                "Control weeds that compete with potatoes",
                "Harvest when vines have died back naturally",
                "Cure potatoes properly before long-term storage"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத விதை உருளைக்கிழங்குகளைப் பயன்படுத்தவும்",
                "வளரும் கிழங்குகளைப் பாதுகாக்க உருளைக்கிழங்குகளை சரியாக மலைக்குச் செல்லவும்",
                "கிழங்கு வளர்ச்சிய期间 சீரான மண் ஈரப்பதத்தை பராமரிக்கவும்",
                "கொலராடோ உருளைக்கிழங்கு வண்டு போன்ற பொதுவான உருளைக்கிழங்கு பூச்சிகளை கண்காணிக்கவும்",
                "மண்ணை சோதித்து சரியான pH மற்றும் வளத்தை பராமரிக்கவும்",
                "உருளைக்கிழங்குடன் போட்டியிடும் களைகளை கட்டுப்படுத்தவும்",
                "வைன்கள் இயற்கையாக இறந்த பிறகு அறுவடை செய்யுங்கள்",
                "நீண்ட கால சேமிப்புக்கு முன் உருளைக்கிழங்கை சரியாக குணப்படுத்தவும்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "गैर-होस्ट फसलों के साथ फसल चक्र अपनाएं",
                "प्रमाणित रोग-मुक्त आलू के बीज का उपयोग करें",
                "विकसित हो रहे कंदों की रक्षा के लिए आलू को ठीक से ढेर करें",
                "कंद विकास के दौरान लगातार मिट्टी की नमी बनाए रखें",
                "कोलोराडो आलू बीटल जैसे सामान्य आलू कीटों की निगरानी करें",
                "मिट्टी का परीक्षण करें और उचित pH और उर्वरता बनाए रखें",
                "आलू के साथ प्रतिस्पर्धा करने वाले खरपतवारों को नियंत्रित करें",
                "जब बेलें प्राकृतिक रूप से मर जाएं तो कटाई करें",
                "दीर्घकालिक भंडारण से पहले आलू को ठीक से सुखाएं"
            ]
        }
    },

    "Raspberry___healthy": {
        "name": {"en": "Healthy Raspberry", "ta": "ராச்பெர்ரி - ஆரோக்கியம்", "hi": "स्वस्थ रसभेरी"},
        "description": {"en": "No disease", "ta": "நோய் இல்லை", "hi": "रोग नहीं"},
        "treatment": {"en": "Maintain good practices", "ta": "நல்ல பழக்க வழக்கங்களை பின்பற்று", "hi": "अच्छा पालन करें"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Prune canes annually to maintain productivity",
                "Apply balanced fertilizer in early spring",
                "Use mulch to conserve moisture and suppress weeds",
                "Monitor for common raspberry pests like raspberry cane borer",
                "Ensure adequate irrigation during fruit development",
                "Test soil pH and maintain between 5.5-6.5",
                "Remove old fruiting canes after harvest",
                "Provide support with trellising for better growth",
                "Harvest berries regularly when fully ripe"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "உற்பத்தித்திறனை பராமரிக்க வருடாந்திர கேன்களை கத்தரிக்கவும்",
                "வசந்த காலத்தின் தொடக்கத்தில் சீரான உரத்தைப் பயன்படுத்தவும்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் களைகளை அடக்க மல்ச் பயன்படுத்தவும்",
                "ராஸ்பெர்ரி கேன் போரர் போன்ற பொதுவான ராஸ்பெர்ரி பூச்சிகளை கண்காணிக்கவும்",
                "பழ வளர்ச்சிய期间 போதுமான நீர்ப்பாசனத்தை உறுதிப்படுத்தவும்",
                "மண் pH ஐ சோதித்து 5.5-6.5 க்கு இடையில் பராமரிக்கவும்",
                "அறுவடைக்குப் பிறகு பழைய பழம் கேன்களை அகற்றவும்",
                "சிறந்த வளர்ச்சிக்கு ட்ரெல்லிஸிங்குடன் ஆதரவை வழங்கவும்",
                "முழுமையாக பழுத்த போது வழக்கமாக பெர்ரிகளை அறுவடை செய்யுங்கள்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "उत्पादकता बनाए रखने के लिए सालाना डंठल की छंटाई करें",
                "शुरुआती वसंत में संतुलित उर्वरक लगाएं",
                "नमी संरक्षण और खरपतवार दमन के लिए मल्च का उपयोग करें",
                "रास्पबेरी केन बोरर जैसे सामान्य रास्पबेरी कीटों की निगरानी करें",
                "फल विकास के दौरान पर्याप्त सिंचाई सुनिश्चित करें",
                "मिट्टी का pH परीक्षण करें और 5.5-6.5 के बीच बनाए रखें",
                "कटाई के बाद पुराने फलने वाले डंठलों को हटा दें",
                "बेहतर विकास के लिए ट्रेलिस के साथ सहायता प्रदान करें",
                "पूरी तरह से पकने पर नियमित रूप से बेरीज की कटाई करें"
            ]
        }
    },

    "Soybean___healthy": {
        "name": {"en": "Healthy Soybean", "ta": "சோயாபீன் - ஆரோக்கியம்", "hi": "स्वस्थ सोयाबीन"},
        "description": {"en": "No disease", "ta": "நோய் இல்லை", "hi": "रोग नहीं"},
        "treatment": {"en": "Standard care", "ta": "பொதுவான பராமரிப்பு", "hi": "मानक देखभाल"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे इल्लैं"},
        "treatment_steps": {
            "en": [
                "Continue regular field monitoring for pests and diseases",
                "Practice crop rotation with non-legume crops",
                "Use certified disease-free seed",
                "Inoculate seeds with proper rhizobium strains",
                "Monitor for common soybean pests like soybean aphid",
                "Maintain proper soil pH between 6.0-6.8",
                "Use balanced fertility based on soil test results",
                "Control weeds that compete with soybeans",
                "Harvest at proper moisture content to avoid seed damage",
                "Keep records of field observations and yields"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான வயல் கண்காணிப்பைத் தொடரவும்",
                "லெக்யூம் அல்லாத பயிர்களுடன் பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத விதைகளைப் பயன்படுத்தவும்",
                "சரியான ரைசோபியம் திரங்களுடன் விதைகளை தடவவும்",
                "சோயாபீன் அப்பிட் போன்ற பொதுவான சோயாபீன் பூச்சிகளை கண்காணிக்கவும்",
                "மண் pH ஐ 6.0-6.8 க்கு இடையில் பராமரிக்கவும்",
                "மண் சோதனை முடிவுகளின் அடிப்படையில் சீரான வளத்தைப் பயன்படுத்தவும்",
                "சோயாபீனுடன் போட்டியிடும் களைகளை கட்டுப்படுத்தவும்",
                "விதை சேதத்தைத் தவிர்க்க சரியான ஈரப்பத உள்ளடக்கத்தில் அறுவடை செய்யுங்கள்",
                "வயல் கண்காணிப்புகள் மற்றும் மகசூல்களின் பதிவுகளை வைத்திருங்கள்"
            ],
            "hi": [
                "कीटों और रोगों के लिए नियमित खेत निगरानी जारी रखें",
                "गैर-फलियों वाली फसलों के साथ फसल चक्र अपनाएं",
                "प्रमाणित रोग-मुक्त बीज का उपयोग करें",
                "उचित राइजोबियम उपभेदों के साथ बीजों को इनोकुलेट करें",
                "सोयाबीन एफिड जैसे सामान्य सोयाबीन कीटों की निगरानी करें",
                "मिट्टी का pH 6.0-6.8 के बीच बनाए रखें",
                "मृदा परीक्षण परिणामों के आधार पर संतुलित उर्वरक का उपयोग करें",
                "सोयाबीन के साथ प्रतिस्पर्धा करने वाले खरपतवारों को नियंत्रित करें",
                "बीज क्षति से बचने के लिए उचित नमी सामग्री पर कटाई करें",
                "खेत अवलोकन और उपज का रिकॉर्ड रखें"
            ]
        }
    },

    "Squash___Powdery_mildew": {
        "name": {"en": "Squash Powdery Mildew", "ta": "ஸ்குவாஷ் தூசி பூஞ்சை", "hi": "स्क्वैश पाउडरी मिल्ड्यू"},
        "description": {
            "en": "White powdery patches on leaves and stems reducing photosynthesis.",
            "ta": "இலைகளிலும் தண்டு பகுதிகளிலும் வெள்ளை தூசி படலங்கள்.",
            "hi": "पत्तियों और तनों पर सफेद पाउडरी पैच जो प्रकाश संश्लेषण कम करते हैं।"
        },
        "treatment": {
            "en": "Improve air circulation, remove infected leaves, apply sulfur sprays if needed.",
            "ta": "காற்றோட்டத்தை மேம்படுத்து; பாதிக்கப்பட்ட இலைகளை அகற்று; தேவையானால் சல்பர் தெளிக்கவும்.",
            "hi": "वायु परिसंचरण बढ़ाएँ, संक्रमित पत्तियाँ हटाएँ, आवश्यक पर सल्फर छिड़काव करें।"
        },
        "spot_location": {
            "en": "Powdery white growth on upper leaf surface and stems.",
            "ta": "மேல்தர இலை மற்றும் தண்டு மேல் வெள்ளை தூசி வளர்ச்சி.",
            "hi": "ऊपरी पत्ती सतह और तनों पर सफेद पाउडरी वृद्धि।"
        },
        "treatment_steps": {
            "en": [
                "Remove severely infected leaves when first noticed",
                "Apply sulfur or potassium bicarbonate sprays",
                "Use horticultural oils like neem oil for control",
                "Improve air circulation through proper spacing",
                "Avoid overhead irrigation to reduce humidity",
                "Apply fungicides before disease becomes severe",
                "Use resistant varieties when available",
                "Remove crop debris after harvest",
                "Apply baking soda sprays (1 tablespoon per gallon of water)",
                "Monitor plants regularly during warm, dry weather"
            ],
            "ta": [
                "முதலில் கவனிக்கப்பட்ட時に கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றவும்",
                "கந்தகம் அல்லது பொட்டாசியம் பைகார்பனேட் தெளிப்புகளைப் பயன்படுத்தவும்",
                "கட்டுப்பாட்டிற்கு நீம் ஆயில் போன்ற தோட்டக்கலை எண்ணெய்களைப் பயன்படுத்தவும்",
                "சரியான இடைவெளி மூலம் காற்றோட்டத்தை மேம்படுத்தவும்",
                "ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "நோய் கடுமையாகும் முன் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை அகற்றவும்",
                "பேக்கிங் சோடா தெளிப்புகளைப் பயன்படுத்தவும் (1 கேலன் தண்ணீருக்கு 1 தேக்கரண்டி)",
                "சூடான, வறண்ட வானிலையில் தாவரங்களை தவறாமல் கண்காணிக்கவும்"
            ],
            "hi": [
                "पहली बार देखे जाने पर गंभीर रूप से संक्रमित पत्तियों को हटा दें",
                "सल्फर या पोटेशियम बाइकार्बोनेट स्प्रे लगाएं",
                "नियंत्रण के लिए नीम ऑयल जैसे बागवानी तेलों का उपयोग करें",
                "उचित रिक्ति के माध्यम से वायु संचार में सुधार करें",
                "आर्द्रता कम करने के लिए ऊपर से सिंचाई से बचें",
                "रोग गंभीर होने से पहले फफूंदनाशक लगाएं",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "कटाई के बाद फसल के अवशेषों को हटा दें",
                "बेकिंग सोडा स्प्रे लगाएं (1 गैलन पानी में 1 बड़ा चम्मच)",
                "गर्म, शुष्क मौसम के दौरान नियमित रूप से पौधों की निगरानी करें"
            ]
        }
    },

    "Strawberry___Leaf_scorch": {
        "name": {"en": "Strawberry Leaf Scorch", "ta": "ஸ்ட்ராபெர்ரி இலை எரிப்பு", "hi": "स्ट्रॉबेरी लीफ स्कॉर्च"},
        "description": {
            "en": "Leaf edges turn brown and dry; can be due to pathogen or environmental stress.",
            "ta": "இலைவிளிம்பு கருங்கலையாய் உலர்ச்சியாக மாறும்; காரணம் பூஞ்சை அல்லது சூழலியல்.",
            "hi": "पत्ती की धारें भूरी और सूखी हो जाती हैं; कारण रोग या पर्यावरणीय तनाव हो सकता है।"
        },
        "treatment": {
            "en": "Improve watering, avoid leaf wetting, remove badly affected leaves, use fungicide if confirmed fungal.",
            "ta": "நீர்வழியை சரிசெய்; இலை ஈரப்படுத்துவதை தவிர்க்கவும்; பாதிக்கப்பட்ட இலைகளை அகற்று.",
            "hi": "सिंचाई सुधारें, पत्तियों को गीला न करें, गंभीर पत्तियाँ हटाएँ, फंगल पुष्टि पर फफूंदनाशक।"
        },
        "spot_location": {
            "en": "Brown, dry edges or scorched patches usually starting at leaf margins.",
            "ta": "கரும் உலரும் விளிம்பு பொதுவாக இலை விளிம்பில் ஆரம்பிக்கும்.",
            "hi": "भूरी, सूखी धारें या जली हुई पैच, अक्सर पत्ती के किनारों पर शुरू।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy severely affected leaves",
                "Improve irrigation practices to avoid drought stress",
                "Apply fungicides if fungal cause is confirmed",
                "Use resistant strawberry varieties when planting",
                "Ensure proper plant spacing for good air circulation",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Apply mulch to conserve soil moisture",
                "Test soil and correct any nutrient deficiencies",
                "Remove old leaves after harvest to reduce inoculum",
                "Monitor plants regularly during dry periods"
            ],
            "ta": [
                "கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்",
                "வறட்சி அழுத்தத்தைத் தவிர்க்க நீர்ப்பாசன பழக்கங்களை மேம்படுத்தவும்",
                "பூஞ்சை காரணம் உறுதிப்படுத்தப்பட்டால் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "நடும் போது எதிர்ப்பு ஸ்ட்ராபெர்ரி வகைகளைப் பயன்படுத்தவும்",
                "நல்ல காற்றோட்டத்திற்கு சரியான தாவர இடைவெளியை உறுதிப்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "மண் ஈரப்பதத்தைப் பாதுகாக்க மல்ச் பயன்படுத்தவும்",
                "மண்ணை சோதித்து ஏதேனும் ஊட்டச்சத்து குறைபாடுகளை சரிசெய்யவும்",
                "இனோகுலத்தைக் குறைக்க அறுவடைக்குப் பிறகு பழைய இலைகளை அகற்றவும்",
                "வறண்ட காலங்களில் தாவரங்களை தவறாமல் கண்காணிக்கவும்"
            ],
            "hi": [
                "गंभीर रूप से प्रभावित पत्तियों को हटाकर नष्ट करें",
                "सूखे के तनाव से बचने के लिए सिंचाई प्रथाओं में सुधार करें",
                "यदि कवक कारण की पुष्टि हो जाए तो फफूंदनाशक लगाएं",
                "रोपण करते समय प्रतिरोधी स्ट्रॉबेरी किस्मों का उपयोग करें",
                "अच्छे वायु संचार के लिए उचित पौध रिक्ति सुनिश्चित करें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "मिट्टी की नमी संरक्षण के लिए मल्च लगाएं",
                "मिट्टी का परीक्षण करें और किसी भी पोषक तत्व की कमी को ठीक करें",
                "इनोकुलम को कम करने के लिए कटाई के बाद पुरानी पत्तियों को हटा दें",
                "शुष्क अवधि के दौरान नियमित रूप से पौधों की निगरानी करें"
            ]
        }
    },

    "Strawberry___healthy": {
        "name": {"en": "Healthy Strawberry", "ta": "ஸ்ட்ராபெர்ரி - ஆரோக்கியம்", "hi": "स्वस्थ स्ट्रॉबेरी"},
        "description": {"en": "No disease", "ta": "நோய் இல்லை", "hi": "रोग नहीं"},
        "treatment": {"en": "Standard care", "ta": "பொதுவான பராமரிப்பு", "hi": "मानक देखभाल"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Renovate strawberry beds annually after harvest",
                "Apply balanced fertilizer in early spring",
                "Use mulch to conserve moisture and keep berries clean",
                "Monitor for common strawberry pests like tarnished plant bug",
                "Ensure adequate irrigation during fruit development",
                "Test soil pH and maintain between 5.5-6.5",
                "Remove old leaves and debris from beds",
                "Provide winter protection in cold climates",
                "Harvest berries regularly when fully ripe"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "அறுவடைக்குப் பிறகு வருடாந்திர ஸ்ட்ராபெர்ரி படுக்கைகளை புதுப்பிக்கவும்",
                "வசந்த காலத்தின் தொடக்கத்தில் சீரான உரத்தைப் பயன்படுத்தவும்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் பெர்ரிகளை சுத்தமாக வைத்திருக்க மல்ச் பயன்படுத்தவும்",
                "கறைபடிந்த தாவர பிழை போன்ற பொதுவான ஸ்ட்ராபெர்ரி பூச்சிகளை கண்காணிக்கவும்",
                "பழ வளர்ச்சிய期间 போதுமான நீர்ப்பாசனத்தை உறுதிப்படுத்தவும்",
                "மண் pH ஐ சோதித்து 5.5-6.5 க்கு இடையில் பராமரிக்கவும்",
                "படுக்கைகளில் இருந்து பழைய இலைகள் மற்றும் குப்பைகளை அகற்றவும்",
                "குளிர் காலநிலையில் குளிர்கால பாதுகாப்பை வழங்கவும்",
                "முழுமையாக பழுத்த போது வழக்கமாக பெர்ரிகளை அறுவடை செய்யுங்கள்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "कटाई के बाद सालाना स्ट्रॉबेरी की क्यारियों का नवीनीकरण करें",
                "शुरुआती वसंत में संतुलित उर्वरक लगाएं",
                "नमी संरक्षण और बेरीज को साफ रखने के लिए मल्च का उपयोग करें",
                "टार्निश्ड प्लांट बग जैसे सामान्य स्ट्रॉबेरी कीटों की निगरानी करें",
                "फल विकास के दौरान पर्याप्त सिंचाई सुनिश्चित करें",
                "मिट्टी का pH परीक्षण करें और 5.5-6.5 के बीच बनाए रखें",
                "क्यारियों से पुरानी पत्तियों और मलबे को हटा दें",
                "ठंडे जलवायु में सर्दियों की सुरक्षा प्रदान करें",
                "पूरी तरह से पकने पर नियमित रूप से बेरीज की कटाई करें"
            ]
        }
    },

    "Tomato___Bacterial_spot": {
        "name": {"en": "Tomato Bacterial Spot", "ta": "தக்காளி பாக்டீரியா புள்ளி", "hi": "टमाटर बैक्टीरियल स्पॉट"},
        "description": {
            "en": "Bacterial lesions on leaves and fruit that can reduce yield and quality.",
            "ta": "இலைகளிலும் பழங்களில் ஏற்படும் பாக்டீரியா பாதிப்பு; விளைவு குறைவு.",
            "hi": "पत्तियों और फलों पर बैक्टीरियल घाव, उत्पादन और गुणवत्ता घटा सकते हैं।"
        },
        "treatment": {
            "en": "Use copper sprays, avoid overhead irrigation, remove infected debris.",
            "ta": "காப்பர் தெளுகலைப் பயன்படுத்தவும்; மேல்நீரிழிவு தவிர்க்கவும்; பாதிக்கப்பட்ட பகுதிகளை அகற்று.",
            "hi": "कॉपर स्प्रे का प्रयोग करें, ऊपर से सिंचाई से बचें, संक्रमित मलबा हटाएँ।"
        },
        "spot_location": {
            "en": "Small dark water-soaked spots on leaves and fruits, may enlarge and coalesce.",
            "ta": "இலைகளிலும் பழங்களிலும் நீர் நன்கு கருப்பு புள்ளிகள்; பரவலாம்.",
            "hi": "पत्तियों और फलों पर छोटे काले पानी-संचित धब्बे, जो बढ़ सकते हैं।"
        },
        "treatment_steps": {
            "en": [
                "Use certified disease-free seed and transplants",
                "Apply copper-based bactericides preventively",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Remove and destroy infected plants promptly",
                "Practice 2-3 year crop rotation with non-host crops",
                "Disinfect tools and equipment between uses",
                "Use resistant varieties when available",
                "Apply streptomycin sprays if copper resistance develops",
                "Space plants properly for good air circulation",
                "Avoid working in fields when plants are wet"
            ],
            "ta": [
                "சான்றளிக்கப்பட்ட நோய்-இல்லாத விதைகள் மற்றும் மாற்றுகளைப் பயன்படுத்தவும்",
                "தடுப்பு முறையில் தாமிரம் அடிப்படையிலான பாக்டீரிசைடுகளைப் பயன்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "பாதிக்கப்பட்ட தாவரங்களை உடனடியாக அகற்றி அழிக்கவும்",
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் 2-3 ஆண்டு பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "பயன்பாடுகளுக்கு இடையில் கருவிகள் மற்றும் உபகரணங்களை கிருமி நீக்கம் செய்யுங்கள்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "தாமிர எதிர்ப்பு வளர்ந்தால் ஸ்ட்ரெப்டோமைசின் தெளிப்புகளைப் பயன்படுத்தவும்",
                "நல்ல காற்றோட்டத்திற்கு தாவரங்களை சரியாக இடைவெளி விடவும்",
                "தாவரங்கள் ஈரமாக இருக்கும் போது வயல்களில் வேலை செய்வதைத் தவிர்க்கவும்"
            ],
            "hi": [
                "प्रमाणित रोग-मुक्त बीज और पौध का उपयोग करें",
                "निवारक रूप से कॉपर-आधारित जीवाणुनाशक लगाएं",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "संक्रमित पौधों को तुरंत हटाकर नष्ट करें",
                "गैर-होस्ट फसलों के साथ 2-3 वर्ष का फसल चक्र अपनाएं",
                "उपयोगों के बीच उपकरणों और उपकरणों को कीटाणुरहित करें",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "यदि कॉपर प्रतिरोध विकसित हो तो स्ट्रेप्टोमाइसिन स्प्रे लगाएं",
                "अच्छे वायु संचार के लिए पौधों को ठीक से रिक्ति दें",
                "पौधों के गीले होने पर खेतों में काम करने से बचें"
            ]
        }
    },

    "Tomato___Early_blight": {
        "name": {"en": "Tomato Early Blight", "ta": "தக்காளி ஆரம்ப அழற்சி", "hi": "टमाटर अर्ली ब्लाइट"},
        "description": {
            "en": "Target-like brown spots with concentric rings on older leaves.",
            "ta": "வளையுரு வட்ட வடிவில் பழைய இலைகளில் கருப்பு வட்டங்கள்.",
            "hi": "पुरानी पत्तियों पर समकेंद्रित रिंग वाले भूरे धब्बे।"
        },
        "treatment": {
            "en": "Remove lower infected leaves, rotate crops, apply fungicide if severe.",
            "ta": "கீழ் பாதிக்கப்பட்ட இலைகளை அகற்று; பயிர் மாறுதல்; பாதிப்பு அதிகமாவிட்டால் பூஞ்சைக் கொல்லி.",
            "hi": "निचली संक्रमित पत्तियाँ हटाएँ, फसल चक्रीकरण और गंभीर होने पर फफूंदनाशक।"
        },
        "spot_location": {
            "en": "Circular 'target' lesions often on older lower leaves.",
            "ta": "வட்டமான 'நோக்கு' புள்ளிகள் பொதுவாக பழைய கீழ் இலைகளில்.",
            "hi": "गोल 'टार्गेट' की तरह घाव अक्सर पुरानी निचली पत्तियों पर।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy lower infected leaves",
                "Practice 2-3 year crop rotation with non-host crops",
                "Apply chlorothalonil or copper-based fungicides",
                "Use resistant varieties when available",
                "Stake plants to improve air circulation",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Apply fungicides preventively before symptoms appear",
                "Remove crop debris after harvest",
                "Maintain adequate potassium levels in soil",
                "Monitor plants regularly, especially lower leaves"
            ],
            "ta": [
                "கீழ் பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்",
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் 2-3 ஆண்டு பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "குளோரோதலோனில் அல்லது தாமிரம் அடிப்படையிலான பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "காற்றோட்டத்தை மேம்படுத்த தாவரங்களை கட்டவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "அறிகுறிகள் தோன்றுவதற்கு முன் தடுப்பு முறையில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை அகற்றவும்",
                "மண்ணில் போதுமான பொட்டாசியம் அளவை பராமரிக்கவும்",
                "தாவரங்களை தவறாமல் கண்காணிக்கவும், குறிப்பாக கீழ் இலைகள்"
            ],
            "hi": [
                "निचली संक्रमित पत्तियों को हटाकर नष्ट करें",
                "गैर-होस्ट फसलों के साथ 2-3 वर्ष का फसल चक्र अपनाएं",
                "क्लोरोथैलोनिल या कॉपर-आधारित फफूंदनाशक लगाएं",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "वायु संचार में सुधार के लिए पौधों को सहारा दें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "लक्षण दिखाई देने से पहले निवारक रूप से फफूंदनाशक लगाएं",
                "कटाई के बाद फसल के अवशेषों को हटा दें",
                "मिट्टी में पर्याप्त पोटेशियम स्तर बनाए रखें",
                "पौधों की नियमित निगरानी करें, विशेष रूप से निचली पत्तियों की"
            ]
        }
    },

    "Tomato___Late_blight": {
        "name": {"en": "Tomato Late Blight", "ta": "தக்காளி பின்னர் அழற்சி", "hi": "टमाटर लेट ब्लाइट"},
        "description": {
            "en": "Rapidly spreading water-soaked lesions often with white sporulation underneath.",
            "ta": "விரைவாக பரவுகிறது; நீர் நன்கு புகும் சிதைவு; கீழ்ப்புற வெள்ளை பூஞ்சை ஏற்படும்.",
            "hi": "तेजी से फैलने वाले पानी-भीगे घाव और नीचे सफ़ेद स्पोरेशन।"
        },
        "treatment": {
            "en": "Remove infected plants, apply appropriate fungicides immediately.",
            "ta": "பாதிக்கப்பட்ட தாவரங்களை அகற்று; உடனே பூஞ்சிக் கொல்லி பாவிக்கவும்.",
            "hi": "संक्रमित पौध हटाएँ और तुरंत उपयुक्त फफूंदनाशक लगाएँ।"
        },
        "spot_location": {
            "en": "Irregular, large, water-soaked lesions on leaves and fruits.",
            "ta": "இலை மற்றும் பழத்தில் பெரிய ஒழுங்கற்ற நீர் நன்கு புண்கள்.",
            "hi": "पत्तियाँ और फल पर बड़े अनियमित पानी-भीगे घाव।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy infected plants immediately",
                "Apply fungicides preventively before disease appears",
                "Use chlorothalonil or mancozeb for protective control",
                "Use systemic fungicides like mefenoxam for curative action",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Stake plants to improve air circulation",
                "Remove potato and tomato volunteers and debris",
                "Apply fungicides at 7-10 day intervals during favorable weather",
                "Use resistant varieties when available",
                "Monitor weather forecasts for disease-favorable conditions"
            ],
            "ta": [
                "பாதிக்கப்பட்ட தாவரங்களை உடனடியாக அகற்றி அழிக்கவும்",
                "நோய் தோன்றுவதற்கு முன் தடுப்பு முறையில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "பாதுகாப்பு கட்டுப்பாட்டிற்கு குளோரோதலோனில் அல்லது மான்கோசெப் பயன்படுத்தவும்",
                "சிகிச்சை நடவடிக்கைக்கு மெஃபினாக்சம் போன்ற சிஸ்டமிக் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "காற்றோட்டத்தை மேம்படுத்த தாவரங்களை கட்டவும்",
                "உருளைக்கிழங்கு மற்றும் தக்காளி தன்னார்வங்கள் மற்றும் குப்பைகளை அகற்றவும்",
                "சாதகமான வானிலையில் 7-10 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "நோய்க்கு சாதகமான நிலைமைகளுக்கான வானிலை முன்னறிவிப்புகளை கண்காணிக்கவும்"
            ],
            "hi": [
                "संक्रमित पौधों को तुरंत हटाकर नष्ट करें",
                "रोग के प्रकट होने से पहले निवारक रूप से फफूंदनाशक लगाएं",
                "सुरक्षात्मक नियंत्रण के लिए क्लोरोथैलोनिल या मैन्कोजेब का उपयोग करें",
                "उपचारात्मक कार्रवाई के लिए मेफेनोक्सम जैसे सिस्टमिक फफूंदनाशकों का उपयोग करें",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "वायु संचार में सुधार के लिए पौधों को सहारा दें",
                "आलू और टमाटर के स्वैच्छिक पौधों और मलबे को हटा दें",
                "अनुकूल मौसम के दौरान 7-10 दिनों के अंतराल पर फफूंदनाशक लगाएं",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "रोग-अनुकूल परिस्थितियों के लिए मौसम पूर्वानुमान की निगरानी करें"
            ]
        }
    },

    "Tomato___Leaf_Mold": {
        "name": {"en": "Tomato Leaf Mold", "ta": "தக்காளி இலை பழுப்பு", "hi": "टमाटर लीफ़ मोल्ड"},
        "description": {
            "en": "Olive-green fuzzy growth mainly on lower leaf surfaces in humid conditions.",
            "ta": "ஈரமான சூழலில் கீழ் இலை மேற்பரப்பில் ஓலிவ்-பச்சை பூஞ்சை வளர்ச்சி.",
            "hi": "आर्द्र स्थितियों में मुख्यतः नीचे की पत्ती सतह पर जैतून-हरे फज़ी विकास।"
        },
        "treatment": {
            "en": "Increase ventilation, remove infected leaves, apply fungicides if needed.",
            "ta": "காற்றோட்டத்தை அதிகரிக்கவும்; பாதிக்கப்பட்ட இலைகளை அகற்று; தேவையானால் பூஞ்சிக் கொல்லி.",
            "hi": "हवादार व्यवस्था बढ़ाएँ, संक्रमित पत्तियाँ हटाएँ और आवश्यक होने पर फफूंदनाशक।"
        },
        "spot_location": {
            "en": "Olive-green fuzzy patches on the underside of lower leaves.",
            "ta": "கீழ் இலைகள் கீழ்புறத்தில் ஓலிவ்-பச்சை மையமான மடிகள்.",
            "hi": "निचली पत्तियों की निचली सतह पर जैतून-हरे फज़ी पैच।"
        },
        "treatment_steps": {
            "en": [
                "Increase ventilation in greenhouse or growing area",
                "Remove severely infected leaves when first noticed",
                "Apply chlorothalonil or copper-based fungicides",
                "Avoid overhead irrigation to reduce humidity",
                "Space plants properly for good air circulation",
                "Use resistant varieties when available",
                "Apply fungicides to lower leaf surfaces for better coverage",
                "Monitor humidity levels and keep below 85%",
                "Remove crop debris after harvest",
                "Apply fungicides at 7-10 day intervals during humid weather"
            ],
            "ta": [
                "கிரீன்ஹவுஸ் அல்லது வளரும் பகுதியில் காற்றோட்டத்தை அதிகரிக்கவும்",
                "முதலில் கவனிக்கப்பட்ட時に கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றவும்",
                "குளோரோதலோனில் அல்லது தாமிரம் அடிப்படையிலான பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "நல்ல காற்றோட்டத்திற்கு தாவரங்களை சரியாக இடைவெளி விடவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "சிறந்த கவரேஜிற்கு கீழ் இலை மேற்பரப்புகளில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "ஈரப்பதம் அளவுகளை கண்காணித்து 85% க்கு கீழே வைக்கவும்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை அகற்றவும்",
                "ஈரமான வானிலையில் 7-10 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்"
            ],
            "hi": [
                "ग्रीनहाउस या बढ़ते क्षेत्र में वेंटिलेशन बढ़ाएं",
                "पहली बार देखे जाने पर गंभीर रूप से संक्रमित पत्तियों को हटा दें",
                "क्लोरोथैलोनिल या कॉपर-आधारित फफूंदनाशक लगाएं",
                "आर्द्रता कम करने के लिए ऊपर से सिंचाई से बचें",
                "अच्छे वायु संचार के लिए पौधों को ठीक से रिक्ति दें",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "बेहतर कवरेज के लिए निचली पत्ती की सतहों पर फफूंदनाशक लगाएं",
                "आर्द्रता स्तरों की निगरानी करें और 85% से नीचे रखें",
                "कटाई के बाद फसल के अवशेषों को हटा दें",
                "आर्द्र मौसम के दौरान 7-10 दिनों के अंतराल पर फफूंदनाशक लगाएं"
            ]
        }
    },

    "Tomato___Septoria_leaf_spot": {
        "name": {"en": "Septoria Leaf Spot", "ta": "செப்டோரியா இலை புள்ளி", "hi": "सेप्टोरिया लीफ स्पॉट"},
        "description": {
            "en": "Small circular spots with dark borders and light centers on leaves.",
            "ta": "கருப்பு எல்லையுடன் இலை மீது சிறு வட்டமான புள்ளிகள்.",
            "hi": "पत्तियों पर गहरे किनारों और हल्के केंद्र वाले छोटे गोल धब्बे।"
        },
        "treatment": {
            "en": "Remove debris, avoid overhead irrigation, apply fungicide if persistent.",
            "ta": "மருங்குகளை அகற்று; மேல்நீர் நீர்வழியை தவிர்க்கவும்; தொடர்ந்தால் பூஞ்சைக் கொல்லி.",
            "hi": "अवशेष हटाएँ, ऊपर से सिंचाई से बचें, लगातार होने पर फफूंदनाशक।"
        },
        "spot_location": {
            "en": "Small round spots on leaf surface, often lower canopy first.",
            "ta": "இலையின் மேற்பரப்பில் சிறு வட்டமான புள்ளிகள்; பொதுவாக கீழ்ப்பகுதியில் முதலில்.",
            "hi": "पत्ती की सतह पर छोटे गोल धब्बे, अक्सर पहले निचली आवरण पर।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy lower infected leaves",
                "Practice 2-3 year crop rotation with non-host crops",
                "Apply chlorothalonil or mancozeb fungicides",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Stake plants to improve air circulation",
                "Remove crop debris after harvest",
                "Apply fungicides at 7-10 day intervals during wet weather",
                "Use resistant varieties when available",
                "Monitor plants regularly, especially after rainfall",
                "Apply fungicides preventively before symptoms appear"
            ],
            "ta": [
                "கீழ் பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்",
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் 2-3 ஆண்டு பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "குளோரோதலோனில் அல்லது மான்கோசெப் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "காற்றோட்டத்தை மேம்படுத்த தாவரங்களை கட்டவும்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை அகற்றவும்",
                "ஈரமான வானிலையில் 7-10 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "தாவரங்களை தவறாமல் கண்காணிக்கவும், குறிப்பாக மழைக்குப் பிறகு",
                "அறிகுறிகள் தோன்றுவதற்கு முன் தடுப்பு முறையில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்"
            ],
            "hi": [
                "निचली संक्रमित पत्तियों को हटाकर नष्ट करें",
                "गैर-होस्ट फसलों के साथ 2-3 वर्ष का फसल चक्र अपनाएं",
                "क्लोरोथैलोनिल या मैन्कोजेब फफूंदनाशक लगाएं",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "वायु संचार में सुधार के लिए पौधों को सहारा दें",
                "कटाई के बाद फसल के अवशेषों को हटा दें",
                "गीले मौसम के दौरान 7-10 दिनों के अंतराल पर फफूंदनाशक लगाएं",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "नियमित रूप से पौधों की निगरानी करें, विशेष रूप से बारिश के बाद",
                "लक्षण दिखाई देने से पहले निवारक रूप से फफूंदनाशक लगाएं"
            ]
        }
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": {"en": "Tomato - Two-spotted Spider Mite", "ta": "இரு புள்ளி இலைக்கட்டி", "hi": "दो धब्बेदार मकड़ी काटने वाला (स्पाइडर माइट)"},
        "description": {
            "en": "Tiny mites causing stippling (small pale spots) and webbing on leaves.",
            "ta": "சிறிய குறுகிய புள்ளிகள் மற்றும் வலைப்புள்ளிகள்; இலைகள் மேல் பாதிக்கப்படும்.",
            "hi": "सूक्ष्म परजीवी जो पत्तियों पर छोटे हल्के धब्बे और जाले बनाते हैं।"
        },
        "treatment": {
            "en": "Use miticides, encourage predators (ladybirds), and avoid drought stress.",
            "ta": "மிடிசைடு பாவி; எதிரி பூச்சிகளை ஊக்கப்படுத்து; வறட்சி அழுத்தத்தை தவிர்த்து.",
            "hi": "माइटिसाइड लगाएँ, शिकारी (लेडीबग) प्रोत्साहित करें और सूखे तनाव से बचें।"
        },
        "spot_location": {
            "en": "Fine stippling on upper surfaces, webbing on undersides when severe.",
            "ta": "மேல் மேற்பரப்பில் நுண் புள்ளிகள்; கடுமையான போது கீழ்தர வலிகள்.",
            "hi": "ऊपरी सतह पर बारीक धब्बे, गंभीर होने पर नीचे जाल।"
        },
        "treatment_steps": {
            "en": [
                "Apply miticides when mites are first detected",
                "Use insecticidal soaps or horticultural oils",
                "Encourage natural predators like ladybugs and lacewings",
                "Avoid broad-spectrum insecticides that kill predators",
                "Use overhead watering to dislodge mites from plants",
                "Remove severely infested leaves when possible",
                "Apply miticides to lower leaf surfaces where mites feed",
                "Monitor plants regularly, especially during hot, dry weather",
                "Use reflective mulches to deter mites",
                "Maintain adequate plant moisture to reduce stress"
            ],
            "ta": [
                "மைட்டுகள் முதலில் கண்டறியப்பட்ட時に மைட்டிசைடுகளைப் பயன்படுத்தவும்",
                "பூச்சிக்கொல்லி சோப்புகள் அல்லது தோட்டக்கலை எண்ணெய்களைப் பயன்படுத்தவும்",
                "லேடிபக்ஸ் மற்றும் லேஸ்விங்ஸ் போன்ற இயற்கை வேட்டையாடுபவர்களை ஊக்குவிக்கவும்",
                "வேட்டையாடுபவர்களைக் கொல்லும் பரந்த-நிறமாலை பூச்சிக்கொல்லிகளைத் தவிர்க்கவும்",
                "தாவரங்களிலிருந்து மைட்டுகளை அகற்ற மேல்நோக்கி நீர்ப்பாசனத்தைப் பயன்படுத்தவும்",
                "முடிந்த時に கடுமையாக பாதிக்கப்பட்ட இலைகளை அகற்றவும்",
                "மைட்டுகள் உண்ணும் கீழ் இலை மேற்பரப்புகளில் மைட்டிசைடுகளைப் பயன்படுத்தவும்",
                "தாவரங்களை தவறாமல் கண்காணிக்கவும், குறிப்பாக சூடான, வறண்ட வானிலையில்",
                "மைட்டுகளை தடுக்க பிரதிபலிப்பு மல்ச்களைப் பயன்படுத்தவும்",
                "அழுத்தத்தைக் குறைக்க போதுமான தாவர ஈரப்பதத்தை பராமரிக்கவும்"
            ],
            "hi": [
                "माइट्स का पता चलने पर माइटिसाइड लगाएं",
                "कीटनाशक साबुन या बागवानी तेलों का उपयोग करें",
                "लेडीबग्स और लेसविंग्स जैसे प्राकृतिक शिकारियों को प्रोत्साहित करें",
                "ऐसे ब्रॉड-स्पेक्ट्रम कीटनाशकों से बचें जो शिकारियों को मारते हैं",
                "पौधों से माइट्स को हटाने के लिए ऊपर से पानी दें",
                "यदि संभव हो तो गंभीर रूप से संक्रमित पत्तियों को हटा दें",
                "निचली पत्ती की सतहों पर माइटिसाइड लगाएं जहां माइट्स भोजन करते हैं",
                "नियमित रूप से पौधों की निगरानी करें, विशेष रूप से गर्म, शुष्क मौसम के दौरान",
                "माइट्स को रोकने के लिए रिफ्लेक्टिव मल्च का उपयोग करें",
                "तनाव कम करने के लिए पर्याप्त पौधे की नमी बनाए रखें"
            ]
        }
    },

    "Tomato___Target_Spot": {
        "name": {"en": "Tomato Target Spot", "ta": "தக்காளி இலக்கு புள்ளி", "hi": "टमाटर टार्गेट स्पॉट"},
        "description": {
            "en": "Distinct concentric rings forming target-like lesions on leaves.",
            "ta": "வளையுரு வளையங்கள் உருவாக்கும் இலக்கு போலிய புள்ளிகள்.",
            "hi": "पत्तियों पर लक्षित जैसी समकेंद्र रिंग बनाते हुए धब्बे।"
        },
        "treatment": {
            "en": "Manage irrigation, remove diseased leaves, and use fungicides if required.",
            "ta": "நீர்வழியை கட்டுப்படுத்து; பாதிக்கப்பட்ட இலைகளை அகற்று; தேவையானால் பூஞ்சைக் கொல்லி.",
            "hi": "सिंचाई प्रबंधित करें, संक्रमित पत्तियाँ हटाएँ और आवश्यक पर फफूंदनाशक।"
        },
        "spot_location": {
            "en": "Target-like rings on leaf surface, often with yellow halos.",
            "ta": "இலையில் இலக்கு போன்ற வளையங்கள்; சில நேரங்களில் மஞ்சள் சுற்றுகள்.",
            "hi": "पत्ती सतह पर लक्ष्य जैसी रिंगें, अक्सर पीले हॉलो के साथ।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy infected leaves when first noticed",
                "Apply chlorothalonil or mancozeb fungicides",
                "Avoid overhead irrigation to reduce leaf wetness",
                "Practice 2-3 year crop rotation with non-host crops",
                "Stake plants to improve air circulation",
                "Use resistant varieties when available",
                "Apply fungicides at 7-10 day intervals during favorable conditions",
                "Remove crop debris after harvest",
                "Monitor plants regularly, especially during warm, humid weather",
                "Apply fungicides preventively before symptoms appear"
            ],
            "ta": [
                "முதலில் கவனிக்கப்பட்ட時に பாதிக்கப்பட்ட இலைகளை அகற்றி அழிக்கவும்",
                "குளோரோதலோனில் அல்லது மான்கோசெப் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "இலை ஈரப்பதத்தைக் குறைக்க மேல்நோக்கி நீர்ப்பாசனத்தைத் தவிர்க்கவும்",
                "ஹோஸ்ட் அல்லாத பயிர்களுடன் 2-3 ஆண்டு பயிர் சுழற்சியைப் பயிற்சி செய்யுங்கள்",
                "காற்றோட்டத்தை மேம்படுத்த தாவரங்களை கட்டவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "சாதகமான நிலைமைகளில் 7-10 நாள் இடைவெளியில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்",
                "அறுவடைக்குப் பிறகு பயிர் குப்பைகளை அகற்றவும்",
                "தாவரங்களை தவறாமல் கண்காணிக்கவும், குறிப்பாக சூடான, ஈரமான வானிலையில்",
                "அறிகுறிகள் தோன்றுவதற்கு முன் தடுப்பு முறையில் பூஞ்சைக் கொல்லிகளைப் பயன்படுத்தவும்"
            ],
            "hi": [
                "पहली बार देखे जाने पर संक्रमित पत्तियों को हटाकर नष्ट करें",
                "क्लोरोथैलोनिल या मैन्कोजेब फफूंदनाशक लगाएं",
                "पत्तियों की नमी कम करने के लिए ऊपर से सिंचाई से बचें",
                "गैर-होस्ट फसलों के साथ 2-3 वर्ष का फसल चक्र अपनाएं",
                "वायु संचार में सुधार के लिए पौधों को सहारा दें",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "अनुकूल परिस्थितियों के दौरान 7-10 दिनों के अंतराल पर फफूंदनाशक लगाएं",
                "कटाई के बाद फसल के अवशेषों को हटा दें",
                "नियमित रूप से पौधों की निगरानी करें, विशेष रूप से गर्म, आर्द्र मौसम के दौरान",
                "लक्षण दिखाई देने से पहले निवारक रूप से फफूंदनाशक लगाएं"
            ]
        }
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": {"en": "Tomato Yellow Leaf Curl Virus (TYLCV)", "ta": "தக்காளி மஞ்சள் இலை சுருண்ட வைரஸ்", "hi": "टमाटर येलो लीफ कर्ल वायरस"},
        "description": {
            "en": "Viral disease spread by whiteflies causing upward leaf curling and yellowing.",
            "ta": "வெள்ளைப் பூச்சிகளால் பரவும் வைரஸ்; இலைகள் மேலே முதுகிடை மற்றும் மஞ்சள் ஆகும்.",
            "hi": "सफेद मक्खियों द्वारा फैला वायरस जिससे पत्तियाँ ऊपर की ओर मुड़ जाती हैं और पीली हो जाती हैं।"
        },
        "treatment": {
            "en": "Control whiteflies, remove infected plants and use resistant varieties if available.",
            "ta": "வெள்ளைப் பூச்சிகளை கட்டுப்படுத்து; பாதிக்கப்பட்ட தாவரங்களை அகற்று; எதிர்ப்பு வகைகளை பாவி.",
            "hi": "सफेद मक्खियों का नियंत्रण करें, संक्रमित पौधे हटाएँ और प्रतिरोधी किस्में प्रयोग करें।"
        },
        "spot_location": {
            "en": "Leaf curling and yellowing rather than distinct spots; sometimes vein yellowing.",
            "ta": "தொகுதி தான்; சுருட்டும் மற்றும் மஞ்சள்; சில நேரங்களில் நரம்பு மஞ்சள்.",
            "hi": "विशिष्ट धब्बे की बजाय पत्ती का मरोड़ और पीला होना; कभी-कभी नसों का पीला होना।"
        },
        "treatment_steps": {
            "en": [
                "Remove and destroy infected plants immediately",
                "Control whitefly populations with insecticides",
                "Use reflective mulches to deter whiteflies",
                "Plant resistant varieties when available",
                "Use insect-proof screens in greenhouse production",
                "Monitor for whiteflies with yellow sticky traps",
                "Remove weed hosts that may harbor whiteflies",
                "Apply systemic insecticides for whitefly control",
                "Avoid planting new crops near infected fields",
                "Use UV-absorbing plastic in greenhouse production"
            ],
            "ta": [
                "பாதிக்கப்பட்ட தாவரங்களை உடனடியாக அகற்றி அழிக்கவும்",
                "பூச்சிக்கொல்லிகளுடன் வெள்ளை ஈ மக்கள்தொகையை கட்டுப்படுத்தவும்",
                "வெள்ளை ஈக்களை தடுக்க பிரதிபலிப்பு மல்ச்களைப் பயன்படுத்தவும்",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளை நடவு செய்யுங்கள்",
                "கிரீன்ஹவுஸ் உற்பத்தியில் பூச்சி-ஆதார வலைகளைப் பயன்படுத்தவும்",
                "மஞ்சள் ஒட்டும் பொறிகளுடன் வெள்ளை ஈக்களை கண்காணிக்கவும்",
                "வெள்ளை ஈக்களை வைத்திருக்கக்கூடிய களை ஹோஸ்ட்களை அகற்றவும்",
                "வெள்ளை ஈ கட்டுப்பாட்டிற்கு சிஸ்டமிக் பூச்சிக்கொல்லிகளைப் பயன்படுத்தவும்",
                "பாதிக்கப்பட்ட வயல்களுக்கு அருகில் புதிய பயிர்களை நடவு செய்வதைத் தவிர்க்கவும்",
                "கிரீன்ஹவுஸ் உற்பத்தியில் UV-உறிஞ்சும் பிளாஸ்டிக் பயன்படுத்தவும்"
            ],
            "hi": [
                "संक्रमित पौधों को तुरंत हटाकर नष्ट करें",
                "कीटनाशकों के साथ व्हाइटफ्लाई आबादी को नियंत्रित करें",
                "व्हाइटफ्लाइज़ को रोकने के लिए रिफ्लेक्टिव मल्च का उपयोग करें",
                "उपलब्ध होने पर प्रतिरोधी किस्में लगाएं",
                "ग्रीनहाउस उत्पादन में कीट-प्रूफ स्क्रीन का उपयोग करें",
                "पीले चिपचिपे जाल के साथ व्हाइटफ्लाइज़ की निगरानी करें",
                "उन खरपतवार होस्टों को हटाएं जो व्हाइटफ्लाइज़ को आश्रय दे सकते हैं",
                "व्हाइटफ्लाई नियंत्रण के लिए सिस्टमिक कीटनाशक लगाएं",
                "संक्रमित खेतों के पास नई फसलें न लगाएं",
                "ग्रीनहाउस उत्पादन में यूवी-अवशोषित प्लास्टिक का उपयोग करें"
            ]
        }
    },

    "Tomato___Tomato_mosaic_virus": {
        "name": {"en": "Tomato Mosaic Virus", "ta": "தக்காளி மோசைக் வைரஸ்", "hi": "टमाटर मोज़ेक वायरस"},
        "description": {
            "en": "Virus causing mosaic mottling and stunted growth; transmitted mechanically and by seed.",
            "ta": "மோசைக் மottle மற்றும் வளர்ச்சி குறைவு; இயந்திரமாகவும் விதைகளால் பரவும்.",
            "hi": "मोज़ेक पैटर्न और बौना विकास पैदा करने वाला वायरस; यांत्रिक और बीज द्वारा फैला।"
        },
        "treatment": {
            "en": "Use certified virus-free seed; remove infected plants and disinfect tools.",
            "ta": "வைரஸ்-இல்லாத விதைகளை பாவிக்கவும்; பாதிக்கப்பட்ட தாவரங்களை அகற்று; கருவிகளை சுத்தம் செய்.",
            "hi": "प्रमाणित वायरस-रहित बीज का उपयोग करें, संक्रमित पौधों को हटाएँ और उपकरणों को सैनिटाइज़ करें।"
        },
        "spot_location": {
            "en": "Mosaic mottling on leaves (light/dark patches), sometimes leaf distortion.",
            "ta": "இலைகளில் மாசைக் மottle (ஒளி/கருப்பு பகுதி), சில நேரங்களில் அழற்சி.",
            "hi": "पत्तियों पर मोज़ेक धब्बे (हल्का/गहरा पैच), कभी-कभी पत्ती विकृति।"
        },
        "treatment_steps": {
            "en": [
                "Use certified virus-free seed and transplants",
                "Remove and destroy infected plants immediately",
                "Disinfect tools with 10% bleach solution between plants",
                "Wash hands thoroughly before handling plants",
                "Control weed hosts that may harbor the virus",
                "Avoid smoking near tomato plants (tobacco mosaic virus)",
                "Use resistant varieties when available",
                "Remove volunteer tomato plants from previous seasons",
                "Practice good sanitation in greenhouse production",
                "Rotate tomatoes with non-host crops for 2-3 years"
            ],
            "ta": [
                "சான்றளிக்கப்பட்ட வைரஸ்-இல்லாத விதைகள் மற்றும் மாற்றுகளைப் பயன்படுத்தவும்",
                "பாதிக்கப்பட்ட தாவரங்களை உடனடியாக அகற்றி அழிக்கவும்",
                "தாவரங்களுக்கு இடையில் கருவிகளை 10% ப்ளீச் கரைசலுடன் கிருமி நீக்கம் செய்யுங்கள்",
                "தாவரங்களை கையாளும் முன் கைகளை முழுமையாக கழுவவும்",
                "வைரஸை வைத்திருக்கக்கூடிய களை ஹோஸ்ட்களை கட்டுப்படுத்தவும்",
                "தக்காளி தாவரங்களுக்கு அருகில் புகைபிடிப்பதைத் தவிர்க்கவும் (புகையிலை மோசைக் வைரஸ்)",
                "கிடைக்கும் போது எதிர்ப்பு வகைகளைப் பயன்படுத்தவும்",
                "முந்தைய பருவங்களிலிருந்து தன்னார்வ தக்காளி தாவரங்களை அகற்றவும்",
                "கிரீன்ஹவுஸ் உற்பத்தியில் நல்ல சுகாதாரத்தைப் பயிற்சி செய்யுங்கள்",
                "2-3 ஆண்டுகளுக்கு ஹோஸ்ட் அல்லாத பயிர்களுடன் தக்காளியை சுழற்றவும்"
            ],
            "hi": [
                "प्रमाणित वायरस-मुक्त बीज और पौध का उपयोग करें",
                "संक्रमित पौधों को तुरंत हटाकर नष्ट करें",
                "पौधों के बीच उपकरणों को 10% ब्लीच घोल से कीटाणुरहित करें",
                "पौधों को संभालने से पहले हाथों को अच्छी तरह धोएं",
                "उन खरपतवार होस्टों को नियंत्रित करें जो वायरस को आश्रय दे सकते हैं",
                "टमाटर के पौधों के पास धूम्रपान से बचें (तम्बाकू मोज़ेक वायरस)",
                "उपलब्ध होने पर प्रतिरोधी किस्मों का उपयोग करें",
                "पिछले मौसमों से स्वैच्छिक टमाटर के पौधों को हटा दें",
                "ग्रीनहाउस उत्पादन में अच्छी स्वच्छता का अभ्यास करें",
                "2-3 वर्षों के लिए गैर-होस्ट फसलों के साथ टमाटर को घुमाएं"
            ]
        }
    },

    "Tomato___healthy": {
        "name": {"en": "Healthy Tomato", "ta": "தக்காளி - ஆரோக்கியம்", "hi": "स्वस्थ टमाटर"},
        "description": {"en": "No visible disease", "ta": "நோய் இல்லை", "hi": "रोग नहीं"},
        "treatment": {"en": "Continue good care", "ta": "நல்ல பராமரிப்பை தொடரவும்", "hi": "अच्छी देखभाल जारी रखें"},
        "spot_location": {"en": "No spots", "ta": "புள்ளிகள் இல்லை", "hi": "कोई धब्बे नहीं"},
        "treatment_steps": {
            "en": [
                "Continue regular monitoring for pests and diseases",
                "Stake or cage plants for better support and air circulation",
                "Apply balanced fertilizer according to soil test",
                "Provide consistent moisture, especially during fruit set",
                "Monitor for common tomato pests like hornworms and aphids",
                "Test soil pH and maintain between 6.0-6.8",
                "Use mulch to conserve moisture and suppress weeds",
                "Prune suckers for indeterminate varieties if desired",
                "Harvest tomatoes at proper maturity for best flavor",
                "Rotate tomatoes with unrelated crops each year"
            ],
            "ta": [
                "பூச்சிகள் மற்றும் நோய்களுக்கு வழக்கமான கண்காணிப்பைத் தொடரவும்",
                "சிறந்த ஆதரவு மற்றும் காற்றோட்டத்திற்கு தாவரங்களை கட்டவும் அல்லது கூட்டு வைக்கவும்",
                "மண் சோதனைக்கு ஏற்ப சீரான உரத்தைப் பயன்படுத்தவும்",
                "பழம் அமைக்கும்期间 குறிப்பாக சீரான ஈரப்பதத்தை வழங்கவும்",
                "ஹார்ன்வார்ம்கள் மற்றும் அப்பிட்கள் போன்ற பொதுவான தக்காளி பூச்சிகளை கண்காணிக்கவும்",
                "மண் pH ஐ சோதித்து 6.0-6.8 க்கு இடையில் பராமரிக்கவும்",
                "ஈரப்பதத்தைப் பாதுகாக்க மற்றும் களைகளை அடக்க மல்ச் பயன்படுத்தவும்",
                "விரும்பினால் காலவரையற்ற வகைகளுக்கு சக்கர்களை கத்தரிக்கவும்",
                "சிறந்த சுவைக்கு சரியான முதிர்ச்சியில் தக்காளியை அறுவடை செய்யுங்கள்",
                "ஒவ்வொரு ஆண்டும் தொடர்பில்லாத பயிர்களுடன் தக்காளியை சுழற்றவும்"
            ],
            "hi": [
                "कीटों और रोगों की नियमित निगरानी जारी रखें",
                "बेहतर सहायता और वायु संचार के लिए पौधों को सहारा दें या पिंजरे में रखें",
                "मृदा परीक्षण के अनुसार संतुलित उर्वरक लगाएं",
                "फल लगने के दौरान विशेष रूप से लगातार नमी प्रदान करें",
                "हॉर्नवर्म्स और एफिड्स जैसे सामान्य टमाटर के कीटों की निगरानी करें",
                "मिट्टी का pH परीक्षण करें और 6.0-6.8 के बीच बनाए रखें",
                "नमी संरक्षण और खरपतवार दमन के लिए मल्च का उपयोग करें",
                "यदि वांछित हो तो अनिश्चित किस्मों के लिए सकर्स को छाँटें",
                "सर्वोत्तम स्वाद के लिए उचित परिपक्वता पर टमाटर की कटाई करें",
                "प्रत्येक वर्ष असंबंधित फसलों के साथ टमाटर को घुमाएं"
            ]
        }
    }
}
















# -----------------------------
# Helper: default info
# -----------------------------
def default_info_for(class_key):
    return {
        "name": {"en": class_key.replace("_", " "), "ta": class_key.replace("_", " "), "hi": class_key.replace("_", " ")},
        "description": {"en": "No additional info available.", "ta": "மேலும் தகவல் இல்லை.", "hi": "अतिरिक्त जानकारी उपलब्ध नहीं है।"},
        "treatment": {"en": "Consult local agricultural expert.", "ta": "உங்கள் விவசாய நிபுணரை அணுகவும்.", "hi": "स्थानीय कृषि विशेषज्ञ से संपर्क करें।"},
        "spot_location": {"en": "Not available", "ta": "கிடைக்கவில்லை", "hi": "उपलब्ध नहीं"}
    }

# -----------------------------
# Routes
# -----------------------------
# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "❌ No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "❌ No file selected"

    try:
        # -----------------------------
        # Image preprocessing
        # -----------------------------
        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)

        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # -----------------------------
        # Prediction
        # -----------------------------
        preds = model.predict(arr, verbose=0)

        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]) * 100)

        predicted_class = index_to_class.get(idx, "Unknown")

        # -----------------------------
        # Save image
        # -----------------------------
        os.makedirs("static", exist_ok=True)
        image_path = os.path.join("static", "uploaded.jpg")
        img.save(image_path)

        # -----------------------------
        # Get disease info
        # -----------------------------
        info = disease_info.get(predicted_class, default_info_for(predicted_class))

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=round(conf, 2),
            image_path=image_path,
            disease_info=info
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)