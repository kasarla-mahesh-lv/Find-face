const callServices = require('../services/callServices');

module.exports = async (fastify) => {

  fastify.post('/make', async (request, reply) => {
    try {
      const { to } = request.body;
      if (!to)
        return reply.code(400).send({ success: false, error: 'to number is required' });

      const call = await callServices.makeCall({
        to,
        from_number: process.env.TWILIO_PHONE_NUMBER,
      });
      return reply.send({
        success:  true,
        call_sid: call.sid,
        status:   call.status,
        message:  'Call initiated!'
      });
    } catch (err) {
      return reply.code(500).send({ success: false, error: err.message });
    }
  });

  fastify.get('/status/:call_sid', async (request, reply) => {
    try {
      const data = await callServices.getCallStatus(request.params.call_sid);
      return reply.send({ success: true, data });
    } catch (err) {
      return reply.code(500).send({ success: false, error: err.message });
    }
  });

  fastify.post('/hangup/:call_sid', async (request, reply) => {
    try {
      const call = await callServices.hangupCall(request.params.call_sid);
      return reply.send({ success: true, message: 'Call ended', status: call.status });
    } catch (err) {
      return reply.code(500).send({ success: false, error: err.message });
    }
  });

  fastify.get('/recording/:call_sid', async (request, reply) => {
    try {
      const recordings = await callServices.getRecording(request.params.call_sid);
      return reply.send({ success: true, data: recordings });
    } catch (err) {
      return reply.code(500).send({ success: false, error: err.message });
    }
  });

  fastify.post('/status', async (request, reply) => {
    const { CallSid, CallStatus, CallDuration } = request.body;
    console.log(`Call ${CallSid} → ${CallStatus} (${CallDuration}s)`);
    return reply.send('<Response/>');
  });

};