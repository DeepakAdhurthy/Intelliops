# telugu_translations.py
# Add this as a new file in your project

from typing import Dict, Optional

class TeluguTranslations:
    """Telugu language translations for the Organic Advisory System"""
    
    def __init__(self):
        self.translations = {
            # Navigation & Menu
            "home": "హోమ్",
            "dashboard": "డాష్‌బోర్డ్",
            "crop_analysis": "పంట విశ్లేషణ",
            "organic_solutions": "సేంద్రీయ పరిష్కారాలు",
            "traditional_practices": "సాంప్రదాయ పద్ధతులు",
            "community": "సంఘం",
            "consultations": "సంప్రదింపులు",
            "marketplace": "మార్కెట్‌ప్లేస్",
            "video_tutorials": "వీడియో ట్యుటోరియల్స్",
            "weather": "వాతావరణం",
            "seasonal_calendar": "కాలానుగుణ క్యాలెండర్",
            "profile": "ప్రొఫైల్",
            "settings": "సెట్టింగ్స్",
            
            # Common Actions
            "upload": "అప్‌లోడ్ చేయండి",
            "submit": "సమర్పించండి",
            "save": "సేవ్ చేయండి",
            "cancel": "రద్దు చేయండి",
            "delete": "తొలగించండి",
            "edit": "సవరించండి",
            "view": "చూడండి",
            "search": "వెతకండి",
            "filter": "ఫిల్టర్ చేయండి",
            "sort": "క్రమబద్ధీకరించండి",
            "share": "షేర్ చేయండి",
            "download": "డౌన్‌లోడ్ చేయండి",
            
            # Crop Analysis
            "upload_photo": "ఫోటో అప్‌లోడ్ చేయండి",
            "take_photo": "ఫోటో తీయండి",
            "analyzing": "విశ్లేషిస్తోంది...",
            "disease_detected": "వ్యాధి కనుగొనబడింది",
            "confidence": "విశ్వాసం",
            "treatment": "చికిత్స",
            "suggested_treatment": "సూచించిన చికిత్స",
            "severity": "తీవ్రత",
            "low": "తక్కువ",
            "medium": "మధ్యస్థ",
            "high": "అధిక",
            "very_high": "చాలా అధికం",
            
            # Diseases (common ones)
            "healthy": "ఆరోగ్యకరమైనది",
            "bacterial_blight": "బాక్టీరియల్ బ్లైట్",
            "leaf_spot": "ఆకు మచ్చ",
            "powdery_mildew": "పొడి బూజు",
            "rust": "తుప్పు",
            "wilt": "వాడిపోవుట",
            "rot": "కుళ్ళు",
            "mosaic": "మొజాయిక్",
            
            # Organic Solutions
            "neem_oil": "వేప నూనె",
            "vermicompost": "వర్మీకంపోస్ట్",
            "panchagavya": "పంచగవ్య",
            "jeevamrutham": "జీవామృతం",
            "ingredients": "పదార్థాలు",
            "preparation": "తయారీ",
            "application": "అప్లికేషన్",
            "success_rate": "విజయ రేటు",
            "cost_per_acre": "ఎకరాకు ఖర్చు",
            
            # Weather
            "temperature": "ఉష్ణోగ్రత",
            "humidity": "తేమ",
            "rainfall": "వర్షపాతం",
            "wind_speed": "గాలి వేగం",
            "forecast": "అంచనా",
            "today": "ఈరోజు",
            "tomorrow": "రేపు",
            "this_week": "ఈ వారం",
            "weather_alert": "వాతావరణ హెచ్చరిక",
            
            # Seasons
            "kharif": "ఖరీఫ్",
            "rabi": "రబీ",
            "summer": "వేసవి",
            "monsoon": "వర్షాకాలం",
            "winter": "శీతాకాలం",
            
            # Crops
            "rice": "వరి",
            "cotton": "పత్తి",
            "chili": "మిర్చి",
            "tomato": "టమాటో",
            "brinjal": "వంకాయ",
            "groundnut": "వేరుశనగ",
            "maize": "మొక్కజొన్న",
            "sugarcane": "చెరకు",
            "banana": "అరటి",
            "mango": "మామిడి",
            
            # Community
            "post": "పోస్ట్",
            "comment": "వ్యాఖ్య",
            "like": "ఇష్టం",
            "share": "షేర్",
            "ask_question": "ప్రశ్న అడగండి",
            "share_experience": "అనుభవం పంచుకోండి",
            "help_others": "ఇతరులకు సహాయం చేయండి",
            
            # Consultations
            "chat": "చాట్",
            "video_call": "వీడియో కాల్",
            "book_consultation": "సంప్రదింపు బుక్ చేయండి",
            "specialist": "నిపుణుడు",
            "online": "ఆన్‌లైన్",
            "offline": "ఆఫ్‌లైన్",
            
            # Marketplace
            "buy": "కొనుగోలు",
            "sell": "అమ్మకం",
            "price": "ధర",
            "quantity": "పరిమాణం",
            "add_to_cart": "కార్ట్‌కు జోడించండి",
            "checkout": "చెక్అవుట్",
            "order": "ఆర్డర్",
            "delivery": "డెలివరీ",
            
            # User Stats
            "crops_monitored": "పర్యవేక్షించిన పంటలు",
            "treatments_applied": "అన్వయించిన చికిత్సలు",
            "success_rate": "విజయ రేటు",
            "total_savings": "మొత్తం పొదుపు",
            "badges_earned": "సంపాదించిన బ్యాడ్జ్‌లు",
            
            # Messages
            "welcome": "స్వాగతం",
            "thank_you": "ధన్యవాదాలు",
            "success": "విజయం",
            "error": "లోపం",
            "loading": "లోడ్ అవుతోంది...",
            "no_data": "డేటా లేదు",
            "try_again": "మళ్ళీ ప్రయత్నించండి",
            
            # Chatbot Responses (Key phrases)
            "hello_response": "నమస్కారం! సేంద్రీయ సలహా వ్యవస్థకు స్వాగతం. నేను మీకు ఎలా సహాయం చేయగలను?",
            "disease_help": "వ్యాధి గుర్తింపు కోసం, దయచేసి పంట విశ్లేషణ విభాగంలో స్పష్టమైన ఫోటోను అప్‌లోడ్ చేయండి.",
            "treatment_help": "మేము 100+ నిరూపితమైన సేంద్రీయ చికిత్స పరిష్కారాలు కలిగి ఉన్నాము!",
            "weather_help": "వాతావరణ సమాచారం మరియు 7-రోజుల అంచనా చూడండి.",
            "specialist_help": "వ్యవసాయ నిపుణులతో కనెక్ట్ అవ్వండి - చాట్ లేదా వీడియో కాల్.",
            
            # Time periods
            "today": "ఈరోజు",
            "yesterday": "నిన్న",
            "last_week": "గత వారం",
            "last_month": "గత నెల",
            "this_year": "ఈ సంవత్సరం",
            
            # Status
            "active": "చురుకుగా",
            "inactive": "నిష్క్రియ",
            "pending": "పెండింగ్",
            "completed": "పూర్తయింది",
            "cancelled": "రద్దు చేయబడింది",
            "in_progress": "ప్రగతిలో ఉంది",
            
            # Common Phrases
            "view_details": "వివరాలు చూడండి",
            "learn_more": "మరింత తెలుసుకోండి",
            "get_started": "ప్రారంభించండి",
            "read_more": "మరింత చదవండి",
            "show_all": "అన్నీ చూపించు",
            "hide": "దాచు",
            "expand": "విస్తరించు",
            "collapse": "కుదించు",
        }
        
        # Chatbot intents in Telugu
        self.chatbot_telugu = {
            "greeting": {
                "patterns": [
                    "నమస్కారం", "హాయ్", "హలో", "శుభోదయం", "శుభ సాయంత్రం",
                    "ఎలా ఉన్నారు", "మీరు ఎలా ఉన్నారు"
                ],
                "responses": [
                    "నమస్కారం! సేంద్రీయ సలహా వ్యవస్థకు స్వాగతం. నేను మీకు ఎలా సహాయం చేయగలను?",
                    "హలో! నేను మీ వ్యవసాయ సహాయకుడిని. మీకు ఏమి తెలుసుకోవాలనుకుంటున్నారు?",
                    "నమస్తే! మీ పంటల గురించి నేను మీకు ఎలా సహాయం చేయగలను?",
                    "హలో! వ్యాధులు, చికిత్సలు లేదా వ్యవసాయ చిట్కాల గురించి నన్ను అడగండి!"
                ]
            },
            "disease_inquiry": {
                "patterns": [
                    "వ్యాధి", "పంట వ్యాధి", "అనారోగ్య పంట", "సమస్య",
                    "ఆకు మచ్చ", "బ్లైట్", "కుళ్ళు", "ఫంగల్ ఇన్ఫెక్షన్",
                    "పురుగు దాడి", "పసుపు ఆకులు", "వాడిపోతున్న పంట"
                ],
                "responses": [
                    "నేను పంట వ్యాధుల గుర్తింపులో మీకు సహాయం చేయగలను!\n\nఖచ్చితమైన రోగ నిర్ధారణ కోసం:\n1. 'పంట విశ్లేషణ' విభాగానికి వెళ్ళండి\n2. ప్రభావిత పంట యొక్క స్పష్టమైన ఫోటోను అప్‌లోడ్ చేయండి\n3. మా AI విశ్వాస స్కోర్‌తో వ్యాధిని గుర్తిస్తుంది\n4. మీకు సేంద్రీయ చికిత్స సిफార్సులు లభిస్తాయి\n\nమీ ప్రాంతంలో ఇటీవలి వ్యాధి నివేదికలను చూపించమంటారా?",
                    "వ్యాధి గుర్తింపు కోసం, దయచేసి పంట విశ్లేషణ విభాగంలో ఫోటో అప్‌లోడ్ చేయండి. ఫోటో:\n- బాగా వెలుతురు మరియు స్పష్టంగా ఉండాలి\n- ప్రభావిత ప్రాంతాలపై దృష్టి పెట్టాలి\n- ఆకులు, కాండాలు లేదా పండ్లను చూపించాలి\n\nమా సిస్టమ్ 50+ సాధారణ పంట వ్యాధులను తక్షణమే గుర్తించగలదు!"
                ]
            },
            "treatment_inquiry": {
                "patterns": [
                    "చికిత్స", "నివారణ", "పరిష్కారం", "పరిహారం",
                    "సేంద్రీయ చికిత్స", "పురుగుమందు", "నియంత్రణ"
                ],
                "responses": [
                    "మా వద్ద 100+ నిరూపితమైన సేంద్రీయ చికిత్స పరిష్కారాలు ఉన్నాయి!\n\n📚 సేంద్రీయ పరిష్కారాల లైబ్రరీ\n- వేప నూనె తయారీలు\n- వర్మీకంపోస్ట్ రెసిపీలు\n- సహజ పురుగుమందులు\n- విజయ రేటు: 85%+\n\n🌿 సాంప్రదాయ పద్ధతులు\n- గిరిజన వ్యవసాయ జ్ఞానం\n- పెద్దల ద్వారా ధృవీకరించబడింది\n- కాలం పరీక్షించిన పద్ధతులు\n\n🎥 వీడియో ట్యుటోరియల్స్\n- దశల వారీగా మార్గదర్శకాలు\n- తెలుగు & ఇంగ్లీష్‌లో\n\nమీరు ముందుగా ఏది అన్వేషించాలనుకుంటున్నారు?"
                ]
            }
        }
    
    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get Telugu translation for a key"""
        return self.translations.get(key, default or key)
    
    def translate(self, text: str, language: str = "telugu") -> str:
        """Translate text based on language preference"""
        if language.lower() in ["telugu", "te", "తెలుగు"]:
            return self.get(text.lower().replace(" ", "_"), text)
        return text
    
    def get_chatbot_response(self, intent: str, language: str = "telugu") -> Optional[str]:
        """Get chatbot response in specified language"""
        if language.lower() in ["telugu", "te", "తెలుగు"]:
            if intent in self.chatbot_telugu:
                import random
                return random.choice(self.chatbot_telugu[intent]["responses"])
        return None
    
    def format_number(self, number: float, language: str = "telugu") -> str:
        """Format numbers according to language"""
        if language.lower() in ["telugu", "te", "తెలుగు"]:
            # Indian numbering system
            return f"₹{number:,.0f}"
        return f"₹{number:,.2f}"


# Helper function to use in routes
def get_translator():
    """Get translator instance"""
    return TeluguTranslations()


# Middleware to detect user language preference
async def get_user_language(user: dict) -> str:
    """Get user's language preference"""
    return user.get("language_preference", "telugu")


# Response wrapper with language support
def localized_response(data: dict, user_language: str = "telugu") -> dict:
    """Wrap response with language-specific translations"""
    translator = get_translator()
    
    # Add common translations
    data["_translations"] = {
        "upload": translator.get("upload"),
        "submit": translator.get("submit"),
        "cancel": translator.get("cancel"),
        "success": translator.get("success"),
        "error": translator.get("error"),
    }
    
    return data