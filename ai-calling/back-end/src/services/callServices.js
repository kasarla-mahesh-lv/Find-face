require('dotenv').config();

const twilio = require('twilio');

const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken  = process.env.TWILIO_AUTH_TOKEN;

const client = new twilio(accountSid, authToken);

const makeCall = async ({ to, from_number }) => {
  const call = await client.calls.create({
    to:   to,
    from: from_number,
    url:  'http://demo.twilio.com/docs/voice.xml',
  });
  return call;
};

const getCallStatus = async (call_sid) => {
  const call = await client.calls(call_sid).fetch();
  return {
    sid:      call.sid,
    status:   call.status,
    duration: call.duration,
    to:       call.to,
    from:     call.from,
  };
};

const hangupCall = async (call_sid) => {
  const call = await client.calls(call_sid).update({
    status: 'completed'
  });
  return call;
};

const getRecording = async (call_sid) => {
  const recordings = await client.recordings.list({
    callSid: call_sid
  });
  return recordings;
};

module.exports = { makeCall, getCallStatus, hangupCall, getRecording };