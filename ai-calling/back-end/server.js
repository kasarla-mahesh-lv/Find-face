require("dotenv").config();
const Fastify = require("fastify");
const formbody = require("@fastify/formbody");
const ivrRoutes = require("./src/routes/ivrRoutes");
const fastify = Fastify({
  logger: true
});
fastify.register(formbody);
fastify.get("/", async (request, reply) => {
  return {
    message: "Luvetha Restaurant"
  };
});
fastify.post("/test", async (request, reply) => {
  const { name } = request.body;
  return {
    message: "API working successfully",
    name: name
  };
});
fastify.register(ivrRoutes, { prefix: "/ivr" });
const start = async () => {
  try {
    await fastify.listen({
      port: process.env.PORT || 5000,
      host: "0.0.0.0"
    }); 
    console.log(
      `Server running on http://localhost:${process.env.PORT || 5000}`
    );
  } catch (error) {
    fastify.log.error(error);
    process.exit(1);
  }
};
start();