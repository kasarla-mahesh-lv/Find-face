const twilio = require("twilio");
const VoiceResponse = twilio.twiml.VoiceResponse;
const voiceConfig = {
  te: "Polly.Aditi",
  en: "Polly.Joanna",
  hi: "Google.hi-IN-Wavenet-A"
};
const languageConfig = {
  en: {
    welcome: "Welcome to Luvetha Restaurant.",
    menu: "Press 1 for table reservation. Press 2 to place a takeaway order. Press 3 for order status. Press 4 for restaurant location and timings. Press 5 to speak with our staff. Press 9 to repeat the menu.",
    reservation: "You selected table reservation. Our reservation team will contact you shortly.",
    order: "You selected takeaway order. Our staff will assist you shortly.",
    status: "Please keep your order number ready. Our team will assist you shortly.",
    location: "Luvetha Restaurant is located on Sarojini Devi Road. We are open from 10 AM to 11 PM.",
    connect: "Connecting you to our restaurant staff.",
    invalid: "Invalid selection. Please try again.",
    noInput: "We did not receive any input."
  },

  te: {
    welcome: "లువేతా రెస్టారెంట్‌కు స్వాగతం.",
    menu: "టేబుల్ రిజర్వేషన్ కోసం 1 నొక్కండి. టేక్ అవే ఆర్డర్ కోసం 2 నొక్కండి. ఆర్డర్ స్థితి కోసం 3 నొక్కండి. చిరునామా మరియు సమయాల కోసం 4 నొక్కండి. మా సిబ్బందితో మాట్లాడడానికి 5 నొక్కండి. మెనూను మళ్లీ వినడానికి 9 నొక్కండి.",
    reservation: "మీరు టేబుల్ రిజర్వేషన్‌ను ఎంచుకున్నారు. మా బృందం త్వరలో మిమ్మల్ని సంప్రదిస్తుంది.",
    order: "మీరు టేక్ అవే ఆర్డర్‌ను ఎంచుకున్నారు. మా సిబ్బంది మీకు సహాయం చేస్తారు.",
    status: "దయచేసి మీ ఆర్డర్ నంబర్ సిద్ధంగా ఉంచండి.",
    location: "లువేతా రెస్టారెంట్ సరోజిని దేవి రోడ్‌లో ఉంది. ఉదయం 10 గంటల నుండి రాత్రి 11 గంటల వరకు తెరిచి ఉంటుంది.",
    connect: "మా రెస్టారెంట్ సిబ్బందికి కాల్‌ను కనెక్ట్ చేస్తున్నాము.",
    invalid: "తప్పు ఎంపిక. దయచేసి మళ్లీ ప్రయత్నించండి.",
    noInput: "మాకు ఎలాంటి ఇన్‌పుట్ రాలేదు."
  },

  hi: {
    welcome: "लुवेथा रेस्टोरेंट में आपका स्वागत है।",
    menu: "टेबल रिजर्वेशन के लिए 1 दबाएं। टेकअवे ऑर्डर के लिए 2 दबाएं। ऑर्डर स्टेटस के लिए 3 दबाएं। लोकेशन और टाइमिंग के लिए 4 दबाएं। हमारे स्टाफ से बात करने के लिए 5 दबाएं। मेनू दोबारा सुनने के लिए 9 दबाएं।",
    reservation: "आपने टेबल रिजर्वेशन चुना है। हमारी टीम जल्द ही आपसे संपर्क करेगी।",
    order: "आपने टेकअवे ऑर्डर चुना है। हमारा स्टाफ आपकी मदद करेगा।",
    status: "कृपया अपना ऑर्डर नंबर तैयार रखें।",
    location: "लुवेथा रेस्टोरेंट सरोजिनी देवी रोड पर स्थित है और सुबह 10 बजे से रात 11 बजे तक खुला रहता है।",
    connect: "आपकी कॉल हमारे स्टाफ से जोड़ी जा रही है।",
    invalid: "गलत चयन। कृपया फिर से प्रयास करें।",
    noInput: "हमें कोई इनपुट प्राप्त नहीं हुआ।"
  }
};

function getLanguageFromDigit(digit) {
  if (digit === "1") return "te";
  if (digit === "2") return "en";
  if (digit === "3") return "hi";
  return null;
}

async function incomingCall(request, reply) {

  const twiml = new VoiceResponse();

  const gather = twiml.gather({
    numDigits: 1,
    action: "/handle-language",
    method: "POST",
    timeout: 7
  });

  gather.say(
    { voice: "Polly.Aditi" },
    "Welcome to Luvetha Restaurant."
  );

  gather.say(
    { voice: "Polly.Aditi" },
    "For Telugu press 1. For English press 2. For Hindi press 3."
  );

  twiml.say("We did not receive any input.");
  twiml.redirect("/voice");

  reply.type("text/xml");
  return reply.send(twiml.toString());
}

async function handleLanguage(request, reply) {

  const twiml = new VoiceResponse();

  const digit = request.body?.Digits;
  const lang = getLanguageFromDigit(digit);

  if (!lang) {

    twiml.say("Invalid selection.");
    twiml.redirect("/voice");
    reply.type("text/xml");
    return reply.send(twiml.toString());
  }
  const voice = voiceConfig[lang];
  const text = languageConfig[lang];
  const gather = twiml.gather({
    numDigits: 1,
    action: `/handle-service?lang=${lang}`,
    method: "POST",
    timeout: 7
  });
  gather.say({ voice }, text.menu);
  twiml.say({ voice }, text.noInput);
  reply.type("text/xml");
  return reply.send(twiml.toString());
}
async function handleService(request, reply) {
  const twiml = new VoiceResponse();
  const digit = request.body?.Digits;
  const lang = request.query?.lang || "en";
  const voice = voiceConfig[lang];
  const text = languageConfig[lang];
  if (digit === "1") {
    twiml.say({ voice }, text.reservation);
    twiml.hangup();
  }
  else if (digit === "2") {
    twiml.say({ voice }, text.order);
    twiml.hangup();

  }

  else if (digit === "3") {
    twiml.say({ voice }, text.status);
    twiml.hangup();
  }
  else if (digit === "4") {
    twiml.say({ voice }, text.location);
    twiml.hangup();
  }
  else if (digit === "5") {
    twiml.say({ voice }, text.connect);
    twiml.dial("+919999999999");
  }
  else if (digit === "9") {
    twiml.redirect(`/handle-language`);
  }
  else {
    twiml.say({ voice }, text.invalid);
    twiml.hangup();
  }
  reply.type("text/xml");
  return reply.send(twiml.toString());
}
module.exports = {
  incomingCall,
  handleLanguage,
  handleService
};