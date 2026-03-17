const controller = require("../controllers/callController")

async function callRoutes(fastify) {

  fastify.post("/incoming-call", controller.incomingCall)

  fastify.post("/ivr-input", controller.ivrInput)

}

module.exports = callRoutes