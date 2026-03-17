const {
  incomingCall,
  handleLanguage,
  handleService
} = require("../controllers/ivrController");

async function ivrRoutes(fastify) {
  fastify.post("/voice", incomingCall);
  fastify.post("/handle-language", handleLanguage);
  fastify.post("/handle-service", handleService);
}
module.exports = ivrRoutes;