require("dotenv").config();

const Fastify = require("fastify");
const formbody = require("@fastify/formbody");
const ivrRoutes = require("./src/routes/ivrRoutes");

const fastify = Fastify({
  logger: true,
});

fastify.register(formbody);

fastify.get("/", async () => ({
  message: "Luvetha Restaurant",
}));

fastify.post("/test", async (request) => {
  const { name } = request.body || {};

  return {
    message: "API working successfully",
    name,
  };
});

fastify.register(ivrRoutes, { prefix: "/ivr" });

async function start() {
  try {
    const port = Number(process.env.PORT) || 5000;

    await fastify.listen({
      port,
      host: "0.0.0.0",
    });

    console.log(`Server running on http://localhost:${port}`);
  } catch (error) {
    fastify.log.error(error);
    process.exit(1);
  }
}

start();
